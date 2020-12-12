#include <iostream>
#include "mpich/mpi.h"


int main(int argc, char** argv) {
    /*Initialization variables*/
    MPI_Init(&argc, &argv);
    int DIM_SIZE = 4;
    int NDIMS = 2;
    const int dims[2] = {DIM_SIZE, DIM_SIZE};
    const int periods[2] = {false, false};
    int coord[2];
    int rank;
    int size;

    /*Create a cartesian grid*/
    MPI_Comm TRANSPUTER_MATRIX;
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, false, &TRANSPUTER_MATRIX);
    MPI_Comm_rank(TRANSPUTER_MATRIX, &rank);
    MPI_Comm_size(TRANSPUTER_MATRIX, &size);
    MPI_Cart_coords(TRANSPUTER_MATRIX, rank, 2, coord);
    printf("Rank %d coordinates are %d %d\n", rank, coord[0], coord[1]);
    fflush(stdout);
    MPI_Barrier(TRANSPUTER_MATRIX);

    if (rank == 0) {
        /*Generate values*/
        int storage[10000];   //a storage of messages in the buffer of (0,0) process
        int displaces[size];  //array of displaces
        int current_displace = 0;
        displaces[0] = 0;

        for (int i = 1; i < 16; i++){
            int gap_size = std::rand() % 100;
            displaces[i] = current_displace + gap_size;
            current_displace += i;
        }

        for (int i = 1; i < 16; i++) {
            for (int j = 0; j < i; j++) {
                storage[displaces[i] + j] = std::rand() % 100;
            }
        }

        /*Send to processes*/
        for (int i = 1; i < 16; i++) {
            MPI_Send(storage + displaces[i], i, MPI_INT, i, 0, TRANSPUTER_MATRIX);
        }

    } else {
        int storage[rank];
        MPI_Recv(storage, rank, MPI_INT, 0, MPI_ANY_TAG, TRANSPUTER_MATRIX, MPI_STATUS_IGNORE);
        for (int i = 0; i < rank; i++) {
            printf("Process num %d has element %d\n", rank, storage[i]);
            //std::cout<<"process num " << rank <<" has element " <<storage[i] <<std::endl;
        }
    }
    MPI_Barrier(TRANSPUTER_MATRIX);
}
