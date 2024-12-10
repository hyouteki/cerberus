#ifndef CERBERUS_CONFIG_H_
#define CERBERUS_CONFIG_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <lodge.h>

#define MaxKeyLen 100
#define MaxValLen 100
#define MaxConfigs 100

typedef enum {
    ConfigValType_Int,
    ConfigValType_String,
    ConfigValType_Float
} ConfigValType;

typedef struct {
    char key[MaxKeyLen];
    ConfigValType type;
    union {
        int int_value;
        char string_value[MaxValLen];
        float float_value;
    } value;
} Config;

typedef struct {
    Config configs[MaxConfigs];
    int len;
} ConfigList;

void init_config_list(ConfigList *list);
int add_config(ConfigList *list, const char *key, const char *value);
Config *get_config(ConfigList *list, const char *key);
void update_config(ConfigList *list, const char *key, void *variable, ConfigValType type);
void parse_config(const char *filename, ConfigList *list);
void print_config_value(Config *config);
void print_config_list(ConfigList *list);

void init_config_list(ConfigList *list) {
    list->len = 0;
}

static int is_float(const char *str) {
    int decimal_point_count = 0;
    while (*str) {
        if (*str == '.') {
            decimal_point_count++;
        } else if (!isdigit(*str) && *str != '-') {
            return 0;
        }
        str++;
    }
    return decimal_point_count == 1;
}

int add_config(ConfigList *list, const char *key, const char *value) {
    lodge_assert(list->len < MaxConfigs, "config list max size is reached");

    char *endptr;
    long num_value = strtol(value, &endptr, 10);
    Config *config = &list->configs[list->len];
    strncpy(config->key, key, MaxKeyLen);
    if (*endptr == '\0') {
        config->type = ConfigValType_Int;
        config->value.int_value = (int)num_value;
    } else if (is_float(value)) {
        config->type = ConfigValType_Float;
        config->value.float_value = strtof(value, NULL);
    } else {
        config->type = ConfigValType_String;
        strncpy(config->value.string_value, value, MaxValLen);
    }

    list->len++;
    return 0;
}

Config *get_config(ConfigList *list, const char *key) {
    for (int i = 0; i < list->len; i++) {
        if (strcmp(list->configs[i].key, key) == 0) {
            return &list->configs[i];
        }
    }
    return NULL;
}

void update_config(ConfigList *list, const char *key, void *variable, ConfigValType type) {
    Config *config = get_config(list, key);
    if (!config) return;

    switch (type) {
        case ConfigValType_String:
            *(char **)variable = strdup(config->value.string_value);
            break;
        case ConfigValType_Int:
            *(int *)variable = config->value.int_value;
            break;
        case ConfigValType_Float:
            *(float *)variable = config->value.float_value;
            break;
    }
}

void parse_config(const char *filename, ConfigList *list) {
    FILE *file = fopen(filename, "r");
    lodge_assert(file, "could not open config file");

    char line[200];
    while (fgets(line, sizeof(line), file)) {
        char key[MaxKeyLen];
        char value[MaxValLen];
        if (line[0] == '\0' || line[0] == '#') {
            continue;
        }
        if (sscanf(line, "%s = %s", key, value) == 2) {
            add_config(list, key, value);
        }
    }

    fclose(file);
}

void print_config_value(Config *config) {
    if (!config) {
		lodge_error("config key not found");
		return;
	}
	if (config->type == ConfigValType_Int) {
		lodge_info("%s: Int = %d", config->key, config->value.int_value);
	} else if (config->type == ConfigValType_String) {
		lodge_info("%s: Str = \"%s\"", config->key, config->value.string_value);
	} else if (config->type == ConfigValType_Float) {
		lodge_info("%s: Float = %f", config->key, config->value.float_value);
	}    
}

void print_config_list(ConfigList *list) {
	for (size_t i = 0; i < list->len; ++i) {
		print_config_value(&list->configs[i]);
	}
}

#endif // CERBERUS_CONFIG_H_
