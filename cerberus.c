#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <arpa/inet.h>

#include "cerberus.h"
#include "config_parser.h"
#include "malpractice.h"
#include <lodge.h>
#include <mnist_dataloader.h>

#define ConfigFilePath "params.config"

// Default values 
static char *images_file_path = "malpractice/mnist/train-images-idx3-ubyte";
static char *labels_file_path = "malpractice/mnist/train-labels-idx1-ubyte";
static size_t input_size = 784, hidden_size = 128, output_size = 10;
static size_t port = 4000, num_clients = 3, global_epochs = 10;
static float learning_rate = 0.01f;
static size_t local_epochs = 10;
static int log_train_metrics = 1;

static Server *server = NULL;

int server_handle_client(Server *server, int client_socket) {
	void *buffer = (void *)malloc(server->params.max_data_buffer_size);
	size_t bytes_received = recv(client_socket, buffer, server->params.max_data_buffer_size, 0);
	if (bytes_received < 0) {
		lodge_error("could not receive data from client socket '%d'", client_socket);
		return 0;
	}
	Model *client_model = (Model *)buffer;
    if (!server->model) {
        server->model = clone_model(client_model);
        lodge_debug("initialized server model from client model");
    } else {
        add_inplace_model(server->model, client_model);
        /* lodge_debug("added client model weights to server (global) model"); */
    }
    free(buffer);

	server->num_received_models++;
	if (server->num_received_models == server->max_clients) {
		normalize_inplace_model(server->model, server->max_clients);
		ClientList *itr = server->client_list;
		while (itr) {
			// NOTE: For the time being model is being updated using the client ptr
			itr->client->model = clone_model(server->model);
			itr->client->train_signal = 1;
			itr = itr->next;
		}
		test(&server->data_array[0], server->model);
		server->num_received_models = 0;
		server->global_epochs_trained++;
	}

	return server->global_epochs_trained == global_epochs;
}

static void *server_instantiate(void *param) {
    Server *server = (Server *)param;
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
	    /* lodge_info("accepted client connection"); */
		int ret = server_handle_client(server, client_socket);
		close(client_socket);
		if (ret) {
			break;
		}
	}

	close(server_socket);
}

void server_constructor(Server *server, Data *data) {
	server->num_received_models = 0;
	server->model = NULL;
	server->client_list = NULL;
	server->num_clients = 0;
	server->global_epochs_trained = 0;
	server->data_array = n_partition_data(data, server->max_clients);
	lodge_info("data partitioned into '%lu' chunks", server->max_clients);
    pthread_mutex_init(&server->model_mutex_lock, NULL);
    pthread_create(&server->pid, NULL, server_instantiate, (void *)server);
}

void server_destructor(Server *server) {
    pthread_mutex_destroy(&server->model_mutex_lock);
	pthread_join(server->pid, NULL);
	lodge_info("server closed successfully", server->pid);
}

void client_list_push(ClientList **client_list, Client *client) {
    ClientList *node = (ClientList *)malloc(sizeof(ClientList));
    node->client = client;
    node->next = *client_list;
    *client_list = node;
}

void client_send_data(Client *client, void *data, size_t data_size) {
	int client_socket = socket(AF_INET, SOCK_STREAM, 0);
	if (client_socket < 0) {
		// Individual client is less important than the server,
		// thus the program shall not terminate upon client socket instantiate failure.
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

static void *client_instantiate(void *param) {
	Client *client = (Client *)param;
	pthread_mutex_lock(&server->model_mutex_lock);
	if (server->num_clients == server->max_clients) {
		lodge_error("max number of clients already reached");
		return NULL;
	}
	// Assigned ID and Data
	client->client_id = server->num_clients++;
	client->data = server->data_array+client->client_id;
    pthread_mutex_unlock(&server->model_mutex_lock);
	client->train_signal = 1;
	char client_prefix[50];
	sprintf(client_prefix, "Client %d =>", client->client_id);
	client->model_params = (Parameters){.learning_rate=learning_rate, .epochs=local_epochs,
										.log_train_metrics=log_train_metrics, .train_prefix=client_prefix};
	client_list_push(&server->client_list, client);
	lodge_info("client with ID '%lu' resgistered and received data chunk", client->client_id);
	describe_data(client->data);
	while (server->global_epochs_trained < global_epochs) {
		client_train(client);
	}
}

void client_register(Client *client) {
	pthread_create(&client->pid, NULL, client_instantiate, (void *)client);
}

void client_train(Client *client) {
	lodge_debug("client %d %lu %d", client->client_id, server->global_epochs_trained, client->train_signal);
	if (server->global_epochs_trained >= global_epochs) return;
	if (!client->train_signal) return;
	train(client->data, client->model_params, client->model);
	client->train_signal = 0;
	client_send_data(client, (void *)client->model, sizeof(Model));
}

void client_destructor(Client *client) {
	pthread_join(client->pid, NULL);
	lodge_info("client '%d' closed successfully", client->client_id);
}

static void load_and_update_config(ConfigList *list) {
    update_config(list, "images_file_path", &images_file_path, ConfigValType_String);
    update_config(list, "labels_file_path", &labels_file_path, ConfigValType_String);
    update_config(list, "input_size", &input_size, ConfigValType_Int);
    update_config(list, "hidden_size", &hidden_size, ConfigValType_Int);
    update_config(list, "output_size", &output_size, ConfigValType_Int);
    update_config(list, "port", &port, ConfigValType_Int);
    update_config(list, "num_clients", &num_clients, ConfigValType_Int);
    update_config(list, "global_epochs", &global_epochs, ConfigValType_Int);
    update_config(list, "learning_rate", &learning_rate, ConfigValType_Float);
    update_config(list, "local_epochs", &local_epochs, ConfigValType_Int);
    update_config(list, "log_train_metrics", &log_train_metrics, ConfigValType_Int);
}

static void parse_config_if_exists() {
	ConfigList list;
    init_config_list(&list);
    parse_config(ConfigFilePath, &list);
	print_config_list(&list);
    load_and_update_config(&list);
}

int main() {
    lodge_set_log_level(LOG_INFO);
	parse_config_if_exists();

	ServerNetworkParams params = {.port=port, .max_concurrent_conns=num_clients, .max_data_buffer_size=65535};
	server = (Server *)malloc(sizeof(Server));
	server->params = params;
	server->max_clients = params.max_concurrent_conns;

	server_constructor(server, mnist_dataloader(images_file_path, labels_file_path));

	Client clients[num_clients];

	for (size_t i = 0; i < num_clients; ++i) {
		clients[i] = (Client){.server_params = &params};
		clients[i].model = initialize_model(input_size, hidden_size, output_size, Model_Init_Random);
		describe_model(clients[i].model);
		client_register(&clients[i]);
	}

	for (size_t i = 0; i < num_clients; ++i) {
		client_destructor(&clients[i]);
	}
	server_destructor(server);
	return 0;
}
