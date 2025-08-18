# plot_trace.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def plot_trace(trace_dir: str, out_name: str = "trace_vis"):
    nodes_csv = os.path.join(trace_dir, "trace_nodes.csv")
    edges_csv = os.path.join(trace_dir, "trace_edges.csv")
    meta_jsonl = os.path.join(trace_dir, "trace.jsonl")

    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    with open(meta_jsonl, "r") as f:
        meta = json.loads(f.readline())

    # build subgraph
    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        G.add_node(int(r["node"]), role=r["role"], hop=int(r["hop"]), trust=None if pd.isna(r["trust"]) else float(r["trust"]))
    for _, r in edges.iterrows():
        G.add_edge(int(r["src"]), int(r["dst"]),
                   hop=int(r["hop"]),
                   trust=float(r["trust"]),
                   passes_alpha=bool(r["passes_alpha"]),
                   is_on_path=bool(r["is_on_path"]))

    # layout: layered by hop
    hop_layers = {}
    for n, data in G.nodes(data=True):
        hop_layers.setdefault(data["hop"], []).append(n)
    pos = {}
    y_gap, x_gap = 1.4, 1.4
    for yi, hop in enumerate(sorted(hop_layers.keys())):
        xs = hop_layers[hop]
        for xi, n in enumerate(xs):
            pos[n] = (xi * x_gap, -yi * y_gap)

    # colors
    node_colors = []
    for n, d in G.nodes(data=True):
        if d["role"] == "requestor":
            node_colors.append("#E74C3C")  # red
        elif d["role"] == "provider":
            node_colors.append("#2ECC71")  # green
        else:
            node_colors.append("#3498DB")  # blue

    plt.figure(figsize=(12, 8))
    # draw edges that failed alpha (faded)
    bad_edges = [(u, v) for u, v, d in G.edges(data=True) if not d["passes_alpha"]]
    nx.draw_networkx_edges(G, pos, edgelist=bad_edges, alpha=0.15, arrows=True, width=1.0)

    # draw edges that passed alpha (normal)
    good_edges = [(u, v) for u, v, d in G.edges(data=True) if d["passes_alpha"]]
    nx.draw_networkx_edges(G, pos, edgelist=good_edges, alpha=0.45, arrows=True, width=1.6)

    # highlight on-path edges
    path_edges = [(u, v) for u, v, d in G.edges(data=True) if d["is_on_path"]]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, alpha=0.95, arrows=True, width=2.8)

    # nodes + labels
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=450, linewidths=1.2, edgecolors="white")
    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    # optional edge labels: trust
    edge_labels = {(u, v): f'{d["trust"]:.2f}' for u, v, d in G.edges(data=True) if d["passes_alpha"]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)

    title = (f'{meta["dataset"]} — {meta["strategy"].capitalize()}  |  '
             f'α={meta["alpha"]}, K={meta["top_k"]}, MAX_HOPS={meta["max_hops"]}  |  '
             f'hops={meta["hops"]}, visited={meta["visited_count"]}  |  '
             f'{"SUCCESS" if meta["success"] else "FAIL"}')
    plt.title(title)
    plt.axis("off")
    out_png = os.path.join(trace_dir, f"{out_name}.png")
    out_pdf = os.path.join(trace_dir, f"{out_name}.pdf")
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
