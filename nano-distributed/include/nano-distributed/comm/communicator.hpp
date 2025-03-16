#ifndef NANO_DISTRIBUTED_COMMUNICATOR_HPP
#define NANO_DISTRIBUTED_COMMUNICATOR_HPP

#include <string>
#include <vector>
#include "mpi_wrapper.hpp"
#include "tcp_socket.hpp"

class Communicator {
public:
    enum class CommType {
        MPI,
        TCP
    };

    Communicator(CommType type);
    ~Communicator();

    void send(const std::string& message, int destination);
    std::string receive(int source);

    void broadcast(const std::string& message);
    std::vector<std::string> gather(const std::string& message, int root);

private:
    CommType comm_type_;
    MPIWrapper mpi_wrapper_;
    TCPSocket tcp_socket_;
};

#endif // NANO_DISTRIBUTED_COMMUNICATOR_HPP