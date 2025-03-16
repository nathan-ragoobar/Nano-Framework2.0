#include <gtest/gtest.h>
#include "nano-distributed/comm/mpi_wrapper.hpp"
#include "nano-distributed/comm/tcp_socket.hpp"
#include "nano-distributed/comm/communicator.hpp"

TEST(MPIWrapperTests, InitializationFinalization) {
    ASSERT_NO_THROW(nano::comm::MPIWrapper::Init());
    ASSERT_NO_THROW(nano::comm::MPIWrapper::Finalize());
}

TEST(TCPSocketTests, Connection) {
    nano::comm::TCPSocket server_socket;
    nano::comm::TCPSocket client_socket;

    ASSERT_NO_THROW(server_socket.Bind("127.0.0.1", 8080));
    ASSERT_NO_THROW(server_socket.Listen());
    
    ASSERT_NO_THROW(client_socket.Connect("127.0.0.1", 8080));
    ASSERT_NO_THROW(client_socket.Send("Hello", 5));

    char buffer[10];
    ASSERT_NO_THROW(server_socket.Accept());
    ASSERT_NO_THROW(server_socket.Receive(buffer, 10));
    ASSERT_STREQ(buffer, "Hello");
}

TEST(CommunicatorTests, MessageSending) {
    nano::comm::Communicator communicator;

    const std::string message = "Test Message";
    std::string received_message;

    ASSERT_NO_THROW(communicator.Send(message, 1));
    ASSERT_NO_THROW(communicator.Receive(received_message, 1));
    ASSERT_EQ(message, received_message);
}