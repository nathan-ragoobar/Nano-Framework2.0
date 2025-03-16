#ifndef NANO_DISTRIBUTED_COMM_TCP_SOCKET_HPP
#define NANO_DISTRIBUTED_COMM_TCP_SOCKET_HPP

#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

class TCPSocket {
public:
    TCPSocket() : sockfd(-1) {}

    ~TCPSocket() {
        if (sockfd != -1) {
            close(sockfd);
        }
    }

    bool create() {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        return sockfd != -1;
    }

    bool bind(int port) {
        sockaddr_in server_addr;
        std::memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(port);
        return ::bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) != -1;
    }

    bool listen(int backlog = 5) {
        return ::listen(sockfd, backlog) != -1;
    }

    bool accept(TCPSocket& client_socket) {
        sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        client_socket.sockfd = ::accept(sockfd, (struct sockaddr*)&client_addr, &client_len);
        return client_socket.sockfd != -1;
    }

    bool connect(const std::string& host, int port) {
        sockaddr_in server_addr;
        std::memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);
        inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr);
        return ::connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) != -1;
    }

    ssize_t send(const void* buffer, size_t length) {
        return ::send(sockfd, buffer, length, 0);
    }

    ssize_t receive(void* buffer, size_t length) {
        return ::recv(sockfd, buffer, length, 0);
    }

private:
    int sockfd;
};

#endif // NANO_DISTRIBUTED_COMM_TCP_SOCKET_HPP