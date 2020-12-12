#include <iostream>
#include <complex>
#include "omp.h"
#include <cmath>
#include "time.h"
#include "sys/time.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpi/mpi.h"
#include <unistd.h>

#define eps 0.01
using namespace std;

typedef complex<double> complexd;

complexd *read(char *f, unsigned int *n, int rank, int size) {
    MPI_File file;
    if (MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_RDONLY, MPI_INFO_NULL,
                      &file)) {
        if (!rank)
            printf("Error opening file %s\n", f);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
    }
    if (!rank)
        MPI_File_read(file, n, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD); //Рассылаем степень
    unsigned long long index = 1LLU << *n;
    cout << size << *n << endl;
    unsigned seg_size = index / size;
    auto *A = new complexd[seg_size];

    double d[2];
    MPI_File_seek(file, sizeof(int) + 2 * seg_size * rank * sizeof(double),
                  MPI_SEEK_SET);
    for (std::size_t i = 0; i < seg_size; ++i) {
        MPI_File_read(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
        A[i].real(d[0]);
        A[i].imag(d[1]);
    }
    MPI_File_close(&file);
    return A;
}

complexd *generate_condition(unsigned long long seg_size, int rank) {
    auto *A = new complexd[seg_size];
    double sqr = 0, module;
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
    MPI_Reduce(&sqr, &module, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        module = sqrt(module);
    }
    MPI_Bcast(&module, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


#pragma omp parallel shared(A, module)
    {
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < seg_size; i++)
            A[i] /= module;
    }
    return A;
}


void check_situation(bool* robustness, bool* extra_load, bool* send_ack, int rank, int size, unsigned rank_change) {
    MPI_Allgather(send_ack, 1, MPI_INT, robustness, 1, MPI_INT, MPI_COMM_WORLD);
    for (int j = 0; j < size; j++) {
        if (!robustness[j] and j == rank_change) {
            *extra_load = true;
        }
    }
}

void
OneQubitEvolution(complexd *buf_zone, complexd U[2][2], unsigned int n, unsigned int k, complexd *recv_zone, int rank,
                  int size, bool extra_load) {
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    unsigned first_index = rank * seg_size;
    unsigned int rank_change = first_index ^(1u << (k - 1));
    rank_change /= seg_size;


    if (rank != rank_change) {
        if (!extra_load) {
            MPI_Sendrecv(buf_zone, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 0, recv_zone, seg_size,
                         MPI_DOUBLE_COMPLEX,
                         rank_change, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
#pragma omp parallel shared(recv_zone, buf_zone, U)
            {
#pragma omp for schedule(static)
                for (int i = 0; i < seg_size; i++) {
                    recv_zone[i] = U[0][0] * buf_zone[i] + U[0][1] * recv_zone[i];
                }
            }
        }
    } else {
        unsigned shift = (int) log2(seg_size) - k;
        unsigned pow = 1u << (shift);
#pragma omp parallel shared(recv_zone, buf_zone, U)
        {
#pragma omp for schedule(static)
            for (std::size_t i = 0; i < seg_size; i++) {
                unsigned i0 = i & ~pow;
                unsigned i1 = i | pow;
                unsigned iq = (i & pow) >> shift;
                recv_zone[i] = U[iq][0] * buf_zone[i0] + U[iq][1] * buf_zone[i1];
            }
        }
    }
}

int main(int argc, char **argv) {
    bool file_read = false;
    char *input;
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

    /*Failure control parameters
     * robustness - monitoring processes with no response
     * extra_load - if we should do extra task due to a neighbour failure
     * send_ack - always true, send a proof that a process is alive*/
    bool robustness[size];
    bool extra_load = false;
    bool send_ack[1] = {true};

    unsigned long long index = 1LLU << n;
    unsigned long long seg_size = index / size;
    unsigned first_index = rank * seg_size;
    unsigned rank_change = first_index ^(1u << (k - 1));
    rank_change /= seg_size;

    /* This control point needs nothing to be done, as there are no computations or reading files yet */
    check_situation(robustness, &extra_load, send_ack, rank, size, rank_change);

    complexd *V;

    //control point

    if (!file_read) {
        V = generate_condition(seg_size, rank);
    } else {
        V = read(input, &n, rank, size);
    }

    auto *recv_buf = new complexd[seg_size]; //buffer to receive a message from neighbour

    check_situation(robustness, &extra_load, send_ack, rank, size, rank_change);
    if (extra_load) {
        MPI_File file;
        if (MPI_File_open(MPI_COMM_WORLD, input, MPI_MODE_RDONLY, MPI_INFO_NULL,
                          &file)) {
            if (!rank)
                printf("Error opening file %s\n", input);
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
        }
        MPI_File_seek(file, sizeof(int) + 2 * seg_size * rank * sizeof(double),
                      MPI_SEEK_SET);
        double d[2];
        recv_buf = new complexd[seg_size];
        for (std::size_t i = 0; i < seg_size; ++i) {
            MPI_File_read(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
            recv_buf[i].real(d[0]);
            recv_buf[i].imag(d[1]);
        }
    }

    complexd U[2][2];
    U[0][0] = 1 / sqrt(2);
    U[0][1] = 1 / sqrt(2);
    U[1][0] = 1 / sqrt(2);
    U[1][1] = -1 / sqrt(2);

    double begin = MPI_Wtime();
    OneQubitEvolution(V, U, n, k, recv_buf, rank, size, extra_load);

    //check if Evolution needs to be redone
    check_situation(robustness, &extra_load, send_ack, rank, size, rank_change);
    if (extra_load) {
        MPI_File file;
        if (MPI_File_open(MPI_COMM_WORLD, input, MPI_MODE_RDONLY, MPI_INFO_NULL,
                          &file)) {
            if (!rank)
                printf("Error opening file %s\n", input);
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
        }
        MPI_File_seek(file, sizeof(int) + 2 * seg_size * rank * sizeof(double),
                      MPI_SEEK_SET);
        double d[2];
        recv_buf = new complexd[seg_size];
        for (std::size_t i = 0; i < seg_size; ++i) {
            MPI_File_read(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
            recv_buf[i].real(d[0]);
            recv_buf[i].imag(d[1]);
        }
    }

    if (extra_load) {
        OneQubitEvolution(V, U, n, k, recv_buf, rank, size, extra_load);
    }

    double end = MPI_Wtime();

    std::cout << "Clear evolution took " << end - begin << " seconds to run." << std::endl;

    MPI_Finalize();
    delete[] V;
}
