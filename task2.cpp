#include <iostream>
#include <complex>
#include "omp.h"
#include <cmath>
#include "time.h"
#include "sys/time.h"
#include <stdio.h>
#include <stdlib.h>
#include "/home/dlichman/tf/include/mpi.h"
#include "/home/dlichman/tf/include/mpi-ext.h"
#include <signal.h>

using namespace std;

typedef complex<double> complexd;

unsigned int rank_change = 0;
bool requires_extra_work = false;
double module = 0;

static void handle_failure(MPI_Comm* pcomm, int* perr, ...) {
    MPI_Comm comm = *pcomm;
    int err = *perr;
    char errstr[MPI_MAX_ERROR_STRING];
    int i, rank, size, nf, len, eclass;
    MPI_Group group_c, group_f;
    int *ranks_gc, *ranks_gf;

    MPI_Error_class(err, &eclass);
    if( MPIX_ERR_PROC_FAILED != eclass ) {
        MPI_Abort(comm, err);
    }

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MPIX_Comm_failure_ack(comm);
    MPIX_Comm_failure_get_acked(comm, &group_f);
    MPI_Group_size(group_f, &nf);
    MPI_Error_string(err, errstr, &len);
    printf("Rank %d / %d: Notified of error %s. %d found dead\n",
           rank, size, errstr, nf);

    ranks_gf = (int*)malloc(nf * sizeof(int));
    ranks_gc = (int*)malloc(nf * sizeof(int));
    MPI_Comm_group(comm, &group_c);
    for(i = 0; i < nf; i++)
        ranks_gf[i] = i;
    MPI_Group_translate_ranks(group_f, nf, ranks_gf,
                              group_c, ranks_gc);
    for(i = 0; i < nf; i++) {
        if (rank_change == ranks_gc[i]) {
            requires_extra_work = true;
        }
    }
}

void write(char *f, complexd *B, int n, int rank, int size) {
    MPI_File file;
    if (MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                      MPI_INFO_NULL, &file)) {
        if (!rank)
            printf("Error opening file %s\n", f);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
    }
    if (rank == 0) {
        MPI_File_write(file, &n, 1, MPI_INT, MPI_STATUS_IGNORE);

    }
    unsigned long long index = 1LLU << n;
    unsigned seg_size = index / size;
    double d[2];
    MPI_File_seek(file, sizeof(int) + 2 * seg_size * rank * sizeof(double), MPI_SEEK_SET);
    for (std::size_t i = 0; i < seg_size; ++i) {
        d[0] = B[i].real();
        d[1] = B[i].imag();
        MPI_File_write(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&file);
}

complexd *generate_condition(unsigned long long seg_size, int rank, int initial_generation) {
    auto *A = new complexd[seg_size];
    double sqr = 0;
    unsigned int seed = time(nullptr) + rank;
#pragma omp parallel shared(A) reduction(+: sqr)
    {
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < seg_size; i++) {
            A[i].real((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
            A[i].imag((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
            sqr += abs(A[i] * A[i]);
        }
    }
    if (initial_generation == true) {
        MPI_Reduce(&sqr, &module, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            module = sqrt(module);
        }
        MPI_Bcast(&module, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    #pragma omp parallel shared(A, module)
        {
    #pragma omp for schedule(static)
            for (std::size_t i = 0; i < seg_size; i++)
                A[i] /= module;
        }

    return A;
}

void
OneQubitEvolution(complexd *buf_zone, complexd U[2][2], unsigned int n, unsigned int k, complexd *recv_zone, int rank,
                  int size, char *f) {
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    unsigned first_index = rank * seg_size;
    unsigned rank_change = first_index ^(1u << (k - 1));
    rank_change /= seg_size;

    printf("RANK %d has a change-neighbor %d\n", rank, rank_change);
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 3) {
        raise(SIGKILL);
    }

    MPI_Sendrecv(buf_zone, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 0, recv_zone, seg_size, MPI_DOUBLE_COMPLEX,
                 rank_change, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (requires_extra_work) {
        requires_extra_work = false;
        recv_zone = generate_condition(seg_size, rank, false);
        printf("Rank 2 is ready to do extra work\n");
    }
    if (rank > rank_change) { //Got data somewhere from left
#pragma omp parallel shared(recv_zone, buf_zone, U)
        {
#pragma omp for schedule(static)
            for (int i = 0; i < seg_size; i++) {
                recv_zone[i] = U[1][0] * recv_zone[i] + U[1][1] * buf_zone[i];
            }
        }
    } else {
        if(rank == 2) {
            printf("Rank 2 is doing extra work\n");
        }
#pragma omp parallel shared(recv_zone, buf_zone, U)
        {
#pragma omp for schedule(static)
            for (int i = 0; i < seg_size; i++) {
                recv_zone[i] = U[0][0] * buf_zone[i] + U[0][1] * recv_zone[i];
            }
        }
    }
}


int main(int argc, char **argv) {
    bool file_read = false;
    bool test_flag = false;
    MPI_Errhandler errh;
    char *input, *output, *test_file;
    unsigned k, n;
    for (int i = 1; i < argc; i++) {
        string option(argv[i]);

        if (option.compare("n") == 0) {
            n = atoi(argv[++i]);
        }

        if ((option.compare("k") == 0)) {
            k = atoi(argv[++i]);
        }

        if ((option.compare("file_read") == 0)) {
            input = argv[++i];
            file_read = true;
        }
    }

    MPI_Init(&argc, &argv);
    int rank;
    int size;


    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm_create_errhandler(handle_failure,
                               &errh);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD,
                            errh);
    MPI_Barrier(MPI_COMM_WORLD);


    unsigned long long index = 1LLU << n;
    unsigned long long seg_size = index / size;

    unsigned first_index = rank * seg_size;
    rank_change = first_index ^(1u << (k - 1));
    rank_change /= seg_size;

    complexd *V;

    V = generate_condition(seg_size, rank, true);

    struct timeval start, stop;

    complexd U[2][2];
    U[0][0] = 1 / sqrt(2);
    U[0][1] = 1 / sqrt(2);
    U[1][0] = 1 / sqrt(2);
    U[1][1] = -1 / sqrt(2);

    auto *recv_buf = new complexd[seg_size];
    double begin = MPI_Wtime();

    OneQubitEvolution(V, U, n, k, recv_buf, rank, size, input);
    double end = MPI_Wtime();
    std::cout << "The process took " << end - begin << " seconds to run." << std::endl;

    MPI_Finalize();
}