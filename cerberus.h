#include <stdio.h>
#include <stdlib.h>

#include <malpractice.h>

typedef struct Node {
	Data *data;
	Model *model;
	
	void (*train)(struct Node *node);
	void (*test)(struct Node *node);
} Node;

typedef struct ServerParams {
    int port;
	size_t max_concurrent_conns;
	size_t max_data_buffer_size;
} ServerParams;

typedef struct Client {
	Node node;
	int client_id;
	// This is pointer as many clients will refer to the same server.
	ServerParams *server_params;

	void (*send_data_to_server)(struct Client *client, void *data, size_t data_size);
} Client;

typedef struct ClientList {
	Client client;
	struct ClientList *next;
} ClientList;

typedef struct Server {
	Node node;
	ClientList *client_list;
	size_t num_clients;
	ServerParams params;

	void (*handle_received_data)(struct Server *server, void *data, size_t data_size);
	void (*handle_client)(struct Server *server, int client_socket);
	void (*instantiate_server)(struct Server *server);
} Server;

void handle_received_data(Server *server, void *data, size_t data_size);
void handle_client(Server *server, int client_socket);
void instantiate_server(Server *server);
void send_data_to_server(Client *client, void *data, size_t data_size);
