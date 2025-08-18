import os
import pandas as pd
import networkx as nx
import pickle

def preprocess_fb_forum(input_path, output_path):
    print(f"🔍 Reading data from: {input_path}")

    # Read source and target columns from edge list
    df = pd.read_csv(input_path, header=None, names=["source", "target", "timestamp"])
    df = df[["source", "target"]]

    print(f"✅ Loaded {len(df)} edges.")

    # Create undirected graph and assign trust = 1.0 to all edges
    G = nx.Graph()
    for u, v in df.values:
        G.add_edge(u, v, trust=1.0)  # Assign fixed trust

    print(f"✅ Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"✅ All edges assigned trust = 1.0")

    # Save the graph as a pickle file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(G, f)

    print(f"📦 Saved preprocessed graph to: {output_path}")

if __name__ == "__main__":
    input_path = r"D:\Dr_Abdul_Rehman\MyPycharm\August 02 2025\SIoT_Discovery_Eval\data\FB_Forum\fb-forum.edges"
    output_path = r"D:\Dr_Abdul_Rehman\MyPycharm\August 02 2025\SIoT_Discovery_Eval\data\processed\FB_Forum.gpickle"
    preprocess_fb_forum(input_path, output_path)
