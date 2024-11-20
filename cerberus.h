#ifndef CERBERUS_H_
#define CERBERUS_H_

#include <stdio.h>
#include <stdlib.h>

#include <malpractice.h>

// Forward declarations
struct Client;
typedef struct Client Client;
struct Server;
typedef struct Server Server;

typedef struct ServerNetworkParams {
    int port;
	size_t max_concurrent_conns;
	size_t max_data_buffer_size;
} ServerNetworkParams;

typedef enum {
	ServerState_DataDistribution,
	ServerState_ModelTraining,
} ServerState;

struct Client {
	int client_id;
	Data *data;
	Model *model;
	// NOTE: This is pointer as many clients will refer to the same server
	ServerNetworkParams *server_params;

	void (*train)(Client *client);
	void (*test)(Client *client);
	void (*register_to_server)(Client *client, Server *server);
	void (*send_data_to_server)(Client *client, void *data, size_t data_size);
};

typedef struct ClientList {
	Client client;
	struct ClientList *next;
} ClientList;

struct Server {
	ClientList *client_list;
	size_t num_clients;
	size_t max_clients;
	Data *data_array;
	ServerNetworkParams params;
	size_t pid;
	ServerState state;

	void (*handle_received_data)(Server *server, void *data, size_t data_size);
	void (*handle_client)(Server *server, int client_socket);
	void (*instantiate)(Server *server, Data *data);
	void (*close)(Server *server);
};

void handle_received_data(Server *server, void *data, size_t data_size);
void handle_client(Server *server, int client_socket);
void instantiate_server(Server *server, Data *data);
void close_server(Server *server);
void send_data_to_server(Client *client, void *data, size_t data_size);

void register_client_to_server(Client *client, Server *server);

#endif // CERBERUS_H_
