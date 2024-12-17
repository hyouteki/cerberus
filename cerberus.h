#ifndef CERBERUS_H_
#define CERBERUS_H_

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "malpractice.h" // Ensure this is included to have Model, Data, etc.

typedef struct ServerNetworkParams {
    int port;
    size_t max_concurrent_conns;
    size_t max_data_buffer_size;
} ServerNetworkParams;

typedef struct Client {
    int client_id;
    Data *data;
    Model *model;
    pthread_t pid;
    Parameters model_params;
    int train_signal;
    ServerNetworkParams *server_params;
} Client;

typedef struct ClientList {
    Client *client;
    struct ClientList *next;
} ClientList;

typedef struct Server {
    ClientList *client_list;
    size_t num_clients;
    size_t max_clients;
    Data *data_array;
    Model *model;
    size_t num_received_models;
    ServerNetworkParams params;
    pthread_t pid;
    size_t global_epochs_trained;
    pthread_mutex_t model_mutex_lock;

    // Additional fields for DOG step size integration:
    Model *initial_model;       // To store the initial global model (x_0)
    Model *model_before_agg;    // To store the previous global model (x_{t-1})
    float r_max;                // To store the maximum radius (max distance from x_0)
    float sum_grad_norm_sq;     // To store the sum of squared "gradient" norms
    float r_epsilon;            // Initial small movement parameter (r_ϵ)
    int is_initialized;         // Flag to indicate if DOG was initialized
} Server;

// Existing functions remain the same
void client_list_push(ClientList **client_list, Client *client);

void server_handle_received_data(Server *server, void *data, size_t data_size);
void server_constructor(Server *server, Data *data);
void server_destructor(Server *server);

void client_send_data(Client *client, void *data, size_t data_size);
void client_register(Client *client);
void client_train(Client *client);
void client_destructor(Client *client);

#endif // CERBERUS_H_
