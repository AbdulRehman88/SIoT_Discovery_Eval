import os
import networkx as nx
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_graph(dataset_name, config):
    dataset = config["datasets"][dataset_name]
    data_dir = os.path.join(config["base_dir"], config["data_paths"]["raw_dir"])
    file_path = os.path.join(data_dir, dataset["file"])
    fmt = dataset["format"]
    directed = dataset.get("directed", False)
    has_weight = dataset.get("has_weight", False)
    skip_header = dataset.get("skip_header", False)
    delimiter = dataset.get("delimiter", ",")

    G = nx.DiGraph() if directed else nx.Graph()

    try:
        if fmt == "pajek":
            G = nx.read_pajek(file_path)
            G = nx.Graph(G) if not directed else nx.DiGraph(G)

        elif fmt == "edgelist":
            G = nx.read_edgelist(file_path, create_using=G)

        elif fmt in ["weighted_edgelist_csv", "edgelist_custom", "edgelist_manual"]:
            with open(file_path, "r") as f:
                if skip_header:
                    next(f)
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('%'):
                        continue  # Skip comments and blank lines
                    parts = line.split(delimiter)

                    try:
                        u, v = parts[0], parts[1]
                        if has_weight and len(parts) >= 3:
                            weight = float(parts[2])
                            G.add_edge(u, v, trust=weight)
                        else:
                            G.add_edge(u, v)
                    except Exception as e:
                        raise ValueError(f"Failed to parse line: {line.strip()} — {e}")
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    except Exception as e:
        raise RuntimeError(f"Error loading graph '{dataset_name}' from {file_path}: {e}")

    return G
