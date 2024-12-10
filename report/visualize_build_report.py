import re
import matplotlib.pyplot as plt

NUM_CLIENTS = 3


def parse_client_accuracy_from_file(filepath):
    client_accuracies = {i: [] for i in range(NUM_CLIENTS)}
    total_epochs, num_local_epochs = 0, 0
    start = True
    regex = r"Client (\d+) => Epoch: (\d+), Accuracy: (\d+\.\d+)"
    with open(filepath, "r") as file:
        for line in file:
            if match := re.search(regex, line.strip()):
                client_id = int(match.group(1))
                accuracy = float(match.group(3))
                client_accuracies[client_id].append(accuracy)
                if client_id == 0:
                    total_epochs += 1
                    if start:
                        num_local_epochs += 1
            if "Test Accuracy" in line.strip():
                start = False
    return num_local_epochs, range(1, total_epochs+1), client_accuracies


def plot_client_accuracy(num_local_epochs, epochs, client_accuracies):
    client_colors = ["b", "g", "r"]
    plt.figure(figsize=(10, 6))
    for client_id, accuracies in client_accuracies.items():
        plt.plot(epochs, accuracies,
                 label=f"Client {client_id}", linestyle="-",
                 color=client_colors[client_id])
    for i in epochs:
        if i % 10 != 0:
            continue
        if (i//10) % 2 == 1:
            plt.axvspan(i, i+num_local_epochs, facecolor="lightgray", alpha=1)

    plt.title("Accuracy over Epochs for Each Client")
    plt.xlabel("Total Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()


def parse_global_accuracy_from_file(filepath):
    accuracies = []
    regex = r"Test Accuracy: (\d+\.\d+)"
    with open(filepath, "r") as file:
        for line in file:
            if match := re.search(regex, line.strip()):
                accuracy = float(match.group(1))
                accuracies.append(accuracy)
    return accuracies


def plot_global_accuracy(accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(accuracies)), accuracies, linestyle="-", color="b")

    plt.title("Global Accuracy over Global Epochs")
    plt.xlabel("Global Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()


filepath = "../build/build_report.txt"
plot_client_accuracy(*parse_client_accuracy_from_file(filepath))
plot_global_accuracy(parse_global_accuracy_from_file(filepath))
