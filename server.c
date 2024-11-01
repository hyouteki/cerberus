#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#include <lodge.h>

#define PORT 3000
#define CONNS 5

int main() {
	int server_socket = socket(AF_INET, SOCK_STREAM, 0);
	if (server_socket < 0) {
	    ExitFatal("could not instantiate socket");
	}

	struct sockaddr_in server_addr;
	server_addr.sin_family = AF_INET;
	server_addr.sin_addr.s_addr = INADDR_ANY;
	server_addr.sin_port = htons(PORT);

	if (bind(server_socket, (struct socker_addr *)&server_addr, sizeof(server_addr)) < 0) {
		close(server_socket);
		ExitFatal("could not bind socket");
	}

	if (listen(server_socket, CONNS) < 0) {
		close(server_socket);
		ExitFatal("could not listen socket");
	}
	LogInfo("server listening on port %ld", PORT);
	LogInfo("maximum concurrent connections: %ld", CONNS);

	int client_socket;
	struct sockaddr_in client_addr;
	size_t addr_len = sizeof(client_addr);
	while (1) {
		client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &addr_len);
		if (client_socket < 0) {
			LogFatal("could not accept client connection");
			continue;
		}
		LogInfo("accepted client connection");

		close(client_socket);
	}

	close(server_socket);
	return 0;
}
