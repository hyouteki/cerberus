#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <arpa/inet.h>

#include "cerberus.h"
#include <lodge.h>
#include <mnist_dataloader.h>

#define ImagesFilePath "malpractice/mnist/train-images-idx3-ubyte"
#define LabelsFilePath "malpractice/mnist/train-labels-idx1-ubyte"

static void handle_received_string_data(Server *server, void *data, size_t data_size) {
	char *str = (char *)data;
	str[data_size] = 0;
	lodge_info("received data: %s", str);
}

void handle_client(Server *server, int client_socket) {
	void *buffer = (void *)malloc(server->params.max_data_buffer_size);
	size_t bytes_received = recv(client_socket, buffer, server->params.max_data_buffer_size, 0);
	if (bytes_received < 0) {
		lodge_error("could not receive data from client socket '%d'", client_socket);
		return;
	}
	server->handle_received_data(server, buffer, bytes_received);
}

void instantiate_server(Server *server, Data *data) {
	// Partitioning data
	server->data_array = n_partition_data(data, server->max_clients);
	lodge_info("data partitioned into '%lu' chunks", server->max_clients);

	pid_t pid = fork();
    if (pid < 0) {
		lodge_fatal("could not create new process for server");
    } else if (pid > 0) {
		// Parent process
		lodge_debug("sleeping for 2 seconds for server to start...");
		sleep(2);
		return;
	}
	server->pid = getpid();

   	int server_socket = socket(AF_INET, SOCK_STREAM, 0);
	if (server_socket < 0) {
	    lodge_fatal("could not instantiate server socket");
	}
	
	struct sockaddr_in server_addr;
	server_addr.sin_family = AF_INET;
	server_addr.sin_addr.s_addr = INADDR_ANY;
	server_addr.sin_port = htons(server->params.port);

	if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
		close(server_socket);
	    lodge_fatal("could not bind socket");
	}

	if (listen(server_socket, server->params.max_concurrent_conns) < 0) {
		close(server_socket);
		lodge_fatal("could not listen socket");
	}
	lodge_info("server listening on port %d", server->params.port);
	lodge_info("maximum concurrent connections: %ld", server->params.max_concurrent_conns);

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
		server->handle_client(server, client_socket);
		close(client_socket);
	}

	close(server_socket);
}

void close_server(Server *server) {
	if (kill(server->pid, SIGKILL) == 0) {
        lodge_info("server with pid '%d' closed successfully", server->pid);
    } else {
		lodge_fatal("failed to close server");
    }
}

void send_data_to_server(Client *client, void *data, size_t data_size) {
	int client_socket = socket(AF_INET, SOCK_STREAM, 0);
	if (client_socket < 0) {
		// Individual client is less important than the server,
		// thus the program does not terminate upon client socket instantiate failure.
	    lodge_error("could not instantiate client socket");
		return;
	}

	struct sockaddr_in server_addr;
	server_addr.sin_family = AF_INET;
	server_addr.sin_port = htons(client->server_params->port);
	// Connect to localhost for testing
	inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

	if (connect(client_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        close(client_socket);
		lodge_error("could not connect to server socket");
		return;
    }

	send(client_socket, data, data_size, 0);
	close(client_socket);
}

void register_client_to_server(Client *client, Server *server) {
	if (server->num_clients == server->max_clients) {
		lodge_error("max number of clients already reached");
		return;
	}
	// Assigned ID and Data
	client->client_id = server->num_clients++;
	client->data = server->data_array+client->client_id;
	lodge_info("client with ID '%lu' resgistered and received data chunk", client->client_id);
}

int main() {
	ServerNetworkParams params = {.port=3000, .max_concurrent_conns=5, .max_data_buffer_size=65535};
	Server server = {.params=params, .handle_received_data=&handle_received_string_data,
					 .handle_client=&handle_client, .instantiate=&instantiate_server,
					 .close=close_server, .max_clients=params.max_concurrent_conns};

	server.instantiate(&server, mnist_dataloader(ImagesFilePath, LabelsFilePath));
	
	Client client = {.server_params=&params, .send_data_to_server=send_data_to_server,
		.register_to_server = &register_client_to_server};

	char *data = "Hello Server, this is client";
	client.register_to_server(&client, &server);
	client.send_data_to_server(&client, (void *)data, strlen(data)+1);

	// Waiting for 4 seconds for server to receive the data then kill server
	sleep(4);
	server.close(&server);
	return 0;
}
