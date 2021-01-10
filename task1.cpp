#include <iostream>
#include "mpich/mpi.h"
#include <utility>
#include <array>
#include <vector>
#include <set>


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
    int total_size = 120;
    std::vector<std::set<int>> active_proc;
    std::set<int> set1;
    std::set<int> set2;
    std::set<int> set3;
    std::set<int> set4;
    std::set<int> set5;
    std::set<int> set6;
    set1.insert(0);

    set2.insert(1);
    set2.insert(4);

    set3.insert(2);
    set3.insert(5);
    set3.insert(8);

    set4.insert(3);
    set4.insert(6);
    set4.insert(9);
    set4.insert(12);

    set5.insert(7);
    set5.insert(10);
    set5.insert(13);

    set6.insert(11);
    set6.insert(14);

    active_proc.push_back(set1);
    active_proc.push_back(set2);
    active_proc.push_back(set3);
    active_proc.push_back(set4);
    active_proc.push_back(set5);
    active_proc.push_back(set6);



    // std:: pair ( ELEMENTS TO THE RIGHT | ELEMENTS TO THE DOWN ) - according to proposed strategy
    std::array<std::pair<int, int>, 16> routing = {std::pair<int, int>(49,71),
                                               std::pair<int, int>(29,19),
                                               std::pair<int, int>(12,15),
                                               std::pair<int, int>(0,9),
                                               std::pair<int, int>(18,49),
                                               std::pair<int, int>(14,18),
                                               std::pair<int, int>(8,15),
                                               std::pair<int, int>(0,10),
                                               std::pair<int, int>(18,23),
                                               std::pair<int, int>(15,12),
                                               std::pair<int, int>(9,11),
                                               std::pair<int, int>(0,8),
                                               std::pair<int, int>(11,0),
                                               std::pair<int, int>(10,0),
                                               std::pair<int, int>(7,0),
                                                   std::pair<int, int>(10,0)};

    std::array<std::pair<int, int>, 16> providers = {std::pair<int, int>(-1,-1),
                                                    std::pair<int, int>(0,-1),
                                                     std::pair<int, int>(1,-1),
                                                     std::pair<int, int>(2,-1),
                                                     std::pair<int, int>(-1,0),
                                                     std::pair<int, int>(4,1),
                                                     std::pair<int, int>(5,2),
                                                     std::pair<int, int>(6,3),
                                                     std::pair<int, int>(-1,4),
                                                     std::pair<int, int>(8,5),
                                                     std::pair<int, int>(9,6),
                                                     std::pair<int, int>(10,7),
                                                     std::pair<int, int>(-1,8),
                                                     std::pair<int, int>(12,9),
                                                     std::pair<int, int>(13,10),
                                                     std::pair<int, int>(14,11)};

    std::array<std::pair<int, int>, 16> adressers = {std::pair<int, int>(1,4),
                                                     std::pair<int, int>(2,5),
                                                     std::pair<int, int>(3,6),
                                                     std::pair<int, int>(-1,7),
                                                     std::pair<int, int>(5,8),
                                                     std::pair<int, int>(6,9),
                                                     std::pair<int, int>(7,10),
                                                     std::pair<int, int>(-1,11),
                                                     std::pair<int, int>(9,12),
                                                     std::pair<int, int>(10,13),
                                                     std::pair<int, int>(11,14),
                                                     std::pair<int, int>(-1,15),
                                                     std::pair<int, int>(13,-1),
                                                     std::pair<int, int>(14,-1),
                                                     std::pair<int, int>(15,-1),
                                                     std::pair<int, int>(-1,-1)};


    /*Create a cartesian grid*/
    MPI_Comm TRANSPUTER_MATRIX;
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, false, &TRANSPUTER_MATRIX);
    MPI_Comm_rank(TRANSPUTER_MATRIX, &rank);
    MPI_Comm_size(TRANSPUTER_MATRIX, &size);
    MPI_Cart_coords(TRANSPUTER_MATRIX, rank, 2, coord);

    const int current_rank = rank;

    fflush(stdout);
    MPI_Barrier(TRANSPUTER_MATRIX);

    int buffer_size = rank + routing[rank].first + routing[rank].second;

    // BUFFER : ( PERSONAL ELEMENTS | ELEMENTS TO THE RIGHT | ELEMENTS TO THE DOWN)
    int *buffer = new int[buffer_size];

    MPI_Barrier(TRANSPUTER_MATRIX);

    //STEPS

    for (auto it = active_proc.begin(); it!= active_proc.end(); ++it) {
        std::set<int> current_active = *it;
        std::set<int> receivers;
        if (it + 1 != active_proc.end()){
            receivers = *(it + 1);
        } else {
            receivers = std::set<int>{15};
        }
        auto search_active = current_active.find(current_rank);

        if (search_active != current_active.end()) { //do parallel send
            if (adressers[rank].first != -1) {
                MPI_Send(buffer + rank, routing[rank].first, MPI_INT, adressers[rank].first, 0, TRANSPUTER_MATRIX);
            }
            if (adressers[rank].second != -1) {
                MPI_Send(buffer + rank + routing[rank].first, routing[rank].second, MPI_INT, adressers[rank].second, 0, TRANSPUTER_MATRIX);
            }
        }

        auto search_receiver = receivers.find(current_rank);

        if (search_receiver != receivers.end()) {
            if (providers[rank].first != -1) {
                MPI_Recv(buffer, buffer_size/2 , MPI_INT, providers[rank].first, MPI_ANY_TAG, TRANSPUTER_MATRIX, nullptr);
            }
            if (providers[rank].second != -1) {
                MPI_Recv(buffer + buffer_size / 2, buffer_size + 1  / 2, MPI_INT, providers[rank].second, MPI_ANY_TAG,TRANSPUTER_MATRIX, nullptr);
            }
        }
        MPI_Barrier(TRANSPUTER_MATRIX);
    }

    MPI_Barrier(TRANSPUTER_MATRIX);
}
