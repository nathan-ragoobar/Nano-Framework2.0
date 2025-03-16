#ifndef NANO_DISTRIBUTED_COMM_MPI_WRAPPER_HPP
#define NANO_DISTRIBUTED_COMM_MPI_WRAPPER_HPP

#include <mpi.h>
#include <stdexcept>
#include <vector>

namespace nano_distributed {
namespace comm {

class MPIWrapper {
public:
    MPIWrapper(int& argc, char**& argv) {
        if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
            throw std::runtime_error("MPI Initialization failed");
        }
    }

    ~MPIWrapper() {
        MPI_Finalize();
    }

    static int getRank() {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }

    static int getSize() {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }

    static void send(const void* data, int count, MPI_Datatype datatype, int dest, int tag) {
        MPI_Send(data, count, datatype, dest, tag, MPI_COMM_WORLD);
    }

    static void receive(void* data, int count, MPI_Datatype datatype, int source, int tag) {
        MPI_Recv(data, count, datatype, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    static void broadcast(void* data, int count, MPI_Datatype datatype, int root) {
        MPI_Bcast(data, count, datatype, root, MPI_COMM_WORLD);
    }

    static void gather(const void* sendbuf, int sendcount, void* recvbuf, int recvcount, MPI_Datatype datatype, int root) {
        MPI_Gather(sendbuf, sendcount, datatype, recvbuf, recvcount, datatype, root, MPI_COMM_WORLD);
    }

    static void scatter(const void* sendbuf, int sendcount, void* recvbuf, int recvcount, MPI_Datatype datatype, int root) {
        MPI_Scatter(sendbuf, sendcount, datatype, recvbuf, recvcount, datatype, root, MPI_COMM_WORLD);
    }
};

} // namespace comm
} // namespace nano_distributed

#endif // NANO_DISTRIBUTED_COMM_MPI_WRAPPER_HPP