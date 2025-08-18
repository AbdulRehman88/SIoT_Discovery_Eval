import os
import random
import pickle

import networkx as nx
from networkx.readwrite import gpickle as nx_gpickle  # ✅ Avoids conflict with any variable named 'gpickle'

import yaml
from src.load_graphs import load_config, load_graph


def assign_services_and_trust(G, service_classes, trust_key="trust"):
    for node in G.nodes():
        G.nodes[node]['service'] = random.choice(service_classes)

    for u, v, data in G.edges(data=True):
        if trust_key not in data:
            data[trust_key] = round(random.uniform(0.5, 1.0), 3)
    return G


def save_graph(G, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(G, f)
    print(f"✅ Saved processed graph to: {output_path}")


def main(dataset_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(current_dir, "..", "config.yaml"))
    config = load_config(config_path)

    service_classes = config["service_classes"]
    processed_dir = os.path.join(config["base_dir"], config["data_paths"]["processed_dir"])
    os.makedirs(processed_dir, exist_ok=True)

    G = load_graph(dataset_name, config)
    G = assign_services_and_trust(G, service_classes)

    out_file = os.path.join(processed_dir, f"{dataset_name}.gpickle")
    save_graph(G, out_file)


if __name__ == "__main__":
    main("BitcoinAlpha")
