#include <stdio.h>
#include <stdlib.h>

#include <malpractice.h>

typedef struct Node {
	Model *model;
	
	void (*synchronize)(struct Node *);
	void (*train)(struct Node *);
	void (*test)(struct Node *);
} Node;

typedef struct Client {
	Node node;
	int client_id;
} Client;

typedef struct ClientList {
	Client client;
	struct ClientList *next;
} ClientList;

typedef struct Server {
	Node node;
	ClientList *client_list;
	size_t num_clients;
} Server;
