#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <arpa/inet.h>
#include <math.h>
#include <pthread.h>

#include "cerberus.h"
#include "config_parser.h"
#include "malpractice.h"
#include "lodge.h"
#include "mnist_dataloader.h"

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

// ---- BEGIN HELPER FUNCTION IMPLEMENTATIONS FOR DOG ----
// NOTE: Make sure these functions conform to the Model structure in malpractice.h.

Model *initialize_empty_model_like(Model *src) {
    Model *m = (Model *)malloc(sizeof(Model));
    m->input_size = src->input_size;
    m->hidden_size = src->hidden_size;
    m->output_size = src->output_size;
    m->tech = Model_Init_Zeroes;
    size_t ih_size = m->input_size * m->hidden_size;
    size_t ho_size = m->hidden_size * m->output_size;
    m->input_hidden_weights = zero_initialize_fvec(ih_size);
    m->hidden_output_weights = zero_initialize_fvec(ho_size);
    return m;
}

void model_subtraction(Model *A, Model *B, Model *dest) {
    for (size_t i = 0; i < A->input_hidden_weights->size; i++) {
        dest->input_hidden_weights->vals[i] = A->input_hidden_weights->vals[i] - B->input_hidden_weights->vals[i];
    }
    for (size_t i = 0; i < A->hidden_output_weights->size; i++) {
        dest->hidden_output_weights->vals[i] = A->hidden_output_weights->vals[i] - B->hidden_output_weights->vals[i];
    }
}

void scale_and_add_model(Model *A, Model *G, float alpha, Model *Out) {
    for (size_t i = 0; i < A->input_hidden_weights->size; i++) {
        Out->input_hidden_weights->vals[i] = A->input_hidden_weights->vals[i] + alpha*G->input_hidden_weights->vals[i];
    }
    for (size_t i = 0; i < A->hidden_output_weights->size; i++) {
        Out->hidden_output_weights->vals[i] = A->hidden_output_weights->vals[i] + alpha*G->hidden_output_weights->vals[i];
    }
}

float model_norm(Model *A) {
    double sum = 0.0;
    for (size_t i = 0; i < A->input_hidden_weights->size; i++) {
        float v = A->input_hidden_weights->vals[i];
        sum += v*v;
    }
    for (size_t i = 0; i < A->hidden_output_weights->size; i++) {
        float v = A->hidden_output_weights->vals[i];
        sum += v*v;
    }
    return (float)sqrt(sum);
}

float model_distance(Model *A, Model *B) {
    double sum = 0.0;
    for (size_t i = 0; i < A->input_hidden_weights->size; i++) {
        float diff = A->input_hidden_weights->vals[i] - B->input_hidden_weights->vals[i];
        sum += diff*diff;
    }
    for (size_t i = 0; i < A->hidden_output_weights->size; i++) {
        float diff = A->hidden_output_weights->vals[i] - B->hidden_output_weights->vals[i];
        sum += diff*diff;
    }
    return (float)sqrt(sum);
}

void free_model(Model *m) {
    if (m) {
        if (m->input_hidden_weights) deinitialize_fvec(m->input_hidden_weights);
        if (m->hidden_output_weights) deinitialize_fvec(m->hidden_output_weights);
        free(m);
    }
}

// ---- END HELPER FUNCTIONS ----

int server_handle_client(Server *server, int client_socket) {
    void *buffer = (void *)malloc(server->params.max_data_buffer_size);
    size_t bytes_received = recv(client_socket, buffer, server->params.max_data_buffer_size, 0);
    if (bytes_received < 0) {
        lodge_error("could not receive data from client socket '%d'", client_socket);
        free(buffer);
        return 0;
    }
    Model *client_model = (Model *)buffer;
    if (!server->model) {
        server->model = clone_model(client_model);
        lodge_debug("initialized server model from client model");
    } else {
        add_inplace_model(server->model, client_model);
    }
    free(buffer);

    server->num_received_models++;
    if (server->num_received_models == server->max_clients) {
        normalize_inplace_model(server->model, server->max_clients);
        Model *avg_model = clone_model(server->model);

        Model *old_model = NULL;
        if (server->global_epochs_trained > 0 || server->is_initialized) {
            old_model = clone_model(server->model_before_agg);
        } else {
            old_model = clone_model(avg_model);
        }

        if (!server->is_initialized) {
            server->initial_model = clone_model(avg_model);
            server->r_max = 0.0f;
            server->sum_grad_norm_sq = 0.0f;
            server->r_epsilon = 1e-6f;

            // g_0 = x_0 - avg_model
            Model *g_0 = initialize_empty_model_like(avg_model);
            model_subtraction(server->initial_model, avg_model, g_0);
            float g_norm = model_norm(g_0);
            server->sum_grad_norm_sq = g_norm*g_norm;
            float eta_0 = server->r_epsilon / g_norm;

            // x_1 = x_0 - eta_0*g_0
            Model *updated_model = initialize_empty_model_like(server->initial_model);
            scale_and_add_model(server->initial_model, g_0, -eta_0, updated_model);
            free_model(server->model);
            server->model = updated_model;

            server->is_initialized = 1;
            free_model(g_0);
        } else {
            // g_t = x_{t-1} - avg_model
            Model *g_t = initialize_empty_model_like(server->model);
            model_subtraction(server->model_before_agg, avg_model, g_t);
            float g_norm = model_norm(g_t);

            server->sum_grad_norm_sq += g_norm*g_norm;

            float dist_from_init = model_distance(server->model_before_agg, server->initial_model);
            if (dist_from_init > server->r_max) {
                server->r_max = dist_from_init;
            }

            float eta_t = server->r_max / sqrtf(server->sum_grad_norm_sq);

            // x_t = x_{t-1} - eta_t*g_t
            Model *updated_model = initialize_empty_model_like(server->model_before_agg);
            scale_and_add_model(server->model_before_agg, g_t, -eta_t, updated_model);
            free_model(server->model);
            server->model = updated_model;

            free_model(g_t);
        }

        // Broadcast updated model to clients
        ClientList *itr = server->client_list;
        while (itr) {
            free_model(itr->client->model);
            itr->client->model = clone_model(server->model);
            itr->client->train_signal = 1;
            itr = itr->next;
        }

        test(&server->data_array[0], server->model);
        server->num_received_models = 0;
        server->global_epochs_trained++;

        if (server->model_before_agg) free_model(server->model_before_agg);
        server->model_before_agg = clone_model(server->model);

        free_model(avg_model);
        free_model(old_model);
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
        int ret = server_handle_client(server, client_socket);
        close(client_socket);
        if (ret) {
            break;
        }
    }

    close(server_socket);
    return NULL;
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

    // DOG-related initializations
    server->initial_model = NULL;
    server->model_before_agg = NULL;
    server->r_max = 0.0f;
    server->sum_grad_norm_sq = 0.0f;
    server->r_epsilon = 1e-6f;
    server->is_initialized = 0;
}

void server_destructor(Server *server) {
    pthread_mutex_destroy(&server->model_mutex_lock);
    pthread_join(server->pid, NULL);
    lodge_info("server closed successfully");
    if (server->initial_model) free_model(server->initial_model);
    if (server->model_before_agg) free_model(server->model_before_agg);
    if (server->model) free_model(server->model);
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
        lodge_error("could not instantiate client socket");
        return;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(client->server_params->port);
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
        pthread_mutex_unlock(&server->model_mutex_lock);
        return NULL;
    }
    client->client_id = server->num_clients++;
    client->data = server->data_array+client->client_id;
    pthread_mutex_unlock(&server->model_mutex_lock);

    client->train_signal = 1;
    char client_prefix[50];
    sprintf(client_prefix, "Client %lu =>", client->client_id);
    client->model_params = (Parameters){.learning_rate=learning_rate, .epochs=local_epochs,
                                        .log_train_metrics=log_train_metrics, .train_prefix=client_prefix};
    client_list_push(&server->client_list, client);
    lodge_info("client with ID '%lu' registered and received data chunk", client->client_id);
    describe_data(client->data);
    while (server->global_epochs_trained < global_epochs) {
        client_train(client);
    }
    return NULL;
}

void client_register(Client *client) {
    pthread_create(&client->pid, NULL, client_instantiate, (void *)client);
}

void client_train(Client *client) {
    lodge_debug("client %lu %lu %d", client->client_id, server->global_epochs_trained, client->train_signal);
    if (server->global_epochs_trained >= global_epochs) return;
    if (!client->train_signal) return;
    train(client->data, client->model_params, client->model);
    client->train_signal = 0;
    client_send_data(client, (void *)client->model, sizeof(Model));
}

void client_destructor(Client *client) {
    pthread_join(client->pid, NULL);
    lodge_info("client '%lu' closed successfully", client->client_id);
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
    free(server);
    return 0;
}
