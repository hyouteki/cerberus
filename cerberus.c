#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#include "cerberus.h"
#include <lodge.h>

void listen_server(Server *server) {
	int server_socket = socket(AF_INET, SOCK_STREAM, 0);
	if (server_socket < 0) {
	    lodge_fatal("could not instantiate socket");
	}

	struct sockaddr_in server_addr;
	server_addr.sin_family = AF_INET;
	server_addr.sin_addr.s_addr = INADDR_ANY;
	server_addr.sin_port = htons(server->port);

	if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
		close(server_socket);
	    lodge_fatal("could not bind socket");
	}

	if (listen(server_socket, server->max_concurrent_conns) < 0) {
		close(server_socket);
		lodge_fatal("could not listen socket");
	}
	lodge_info("server listening on port %d", server->port);
	lodge_info("maximum concurrent connections: %ld", server->max_concurrent_conns);

	int client_socket;
	struct sockaddr_in client_addr;
	socklen_t addr_len = sizeof(client_addr);
	while (1) {
		client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &addr_len);
		if (client_socket < 0) {
			lodge_error("could not accept client connection");
			continue;
		}
	    lodge_info("accepted client connection");

		close(client_socket);
	}

	close(server_socket);
}

int main() {
	return 0;
}
