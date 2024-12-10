#include "config_parser.h"

int main() {
    ConfigList config_list;
    init_config_list(&config_list);

    const char *filename = "params.config";
    parse_config(filename, &config_list);
    print_config_list(&config_list);

    return 0;
}
