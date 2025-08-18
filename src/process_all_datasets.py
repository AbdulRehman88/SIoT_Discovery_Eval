# src/process_all_datasets.py

import os
import traceback
from src.load_graphs import load_config, load_graph
from assign_services_and_trust import assign_services_and_trust, save_graph

def main():
    # Resolve and load config.yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(current_dir, "..", "config.yaml"))
    config = load_config(config_path)

    # Read parameters from config
    service_classes = config["service_classes"]
    base_dir = config["base_dir"]
    processed_dir = os.path.join(base_dir, config["data_paths"]["processed_dir"])
    os.makedirs(processed_dir, exist_ok=True)

    # Loop through datasets
    for dataset_name in config["datasets"]:
        print(f"\n🔄 Processing: {dataset_name}")
        try:
            G = load_graph(dataset_name, config)
            G = assign_services_and_trust(G, service_classes)
            out_file = os.path.join(processed_dir, f"{dataset_name}.gpickle")
            save_graph(G, out_file)
        except Exception as e:
            print(f"❌ Failed to process {dataset_name}: {e}")
            # Optional: uncomment below to print full traceback
            # traceback.print_exc()

if __name__ == "__main__":
    main()
