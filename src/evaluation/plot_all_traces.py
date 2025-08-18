# plot_all_traces.py
import os
import sys
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx

ROOT = r"D:\Dr_Abdul_Rehman\MyPycharm\August 02 2025\SIoT_Discovery_Eval"
TRACES_DIR = os.path.join(ROOT, "results", "traces")

# ---- styling ----
NODE_SIZE = 450
ARROW_SIZE = 12
FONT_SIZE = 9

COLORS = {
    "requestor": "#d62728",     # red
    "provider":  "#2ca02c",     # green
    "intermediate": "#1f77b4",  # blue
}
EDGE_COLOR_PASS   = "#4c78a8"   # darker for edges that pass α
EDGE_COLOR_NOPASS = "#c7c7c7"   # light grey for edges failing α
EDGE_COLOR_PATH   = "#ff7f0e"   # orange for edges on discovered path

def read_trace(trace_dir):
    nodes_path = os.path.join(trace_dir, "trace_nodes.csv")
    edges_path = os.path.join(trace_dir, "trace_edges.csv")
    meta_path  = os.path.join(trace_dir, "trace.jsonl")

    if not (os.path.isfile(nodes_path) and os.path.isfile(edges_path)):
        print(f"[skip] {trace_dir}: missing trace_nodes.csv or trace_edges.csv")
        return None

    nodes = {}
    with open(nodes_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nid = str(row["node"])
            nodes[nid] = {
                "role": row.get("role", "intermediate"),
                "hop": int(float(row.get("hop") or 0)),
                "trust": (None if row.get("trust") in ("", "None", None) else float(row["trust"])),
            }

    edges = []
    with open(edges_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            e = {
                "src": str(row["src"]),
                "dst": str(row["dst"]),
                "hop": int(float(row.get("hop") or 0)),
                "trust": float(row.get("trust") or 0.0),
                "passes_alpha": str(row.get("passes_alpha", "False")).lower() == "true",
                "is_on_path":   str(row.get("is_on_path", "False")).lower() == "true",
            }
            edges.append(e)

    meta = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                import json
                meta = json.loads(f.readline().strip())
        except Exception:
            pass

    return {"nodes": nodes, "edges": edges, "meta": meta}


def draw_trace(trace_dir, trace):
    nodes = trace["nodes"]
    edges = trace["edges"]
    meta  = trace.get("meta", {})

    # build directed graph from visited nodes/edges
    G = nx.DiGraph()
    for n, attrs in nodes.items():
        G.add_node(n, **attrs)
    for e in edges:
        G.add_edge(e["src"], e["dst"], **e)

    # layout: spring on this small subgraph
    # seed for reproducibility
    pos = nx.spring_layout(G, seed=7, k=None, iterations=100)

    # figure
    plt.figure(figsize=(9, 6))

    # draw non-path edges: first those that fail alpha (light), then passing (darker)
    fail_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("is_on_path") and not d.get("passes_alpha")]
    pass_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("is_on_path") and d.get("passes_alpha")]

    nx.draw_networkx_edges(G, pos, edgelist=fail_edges, edge_color=EDGE_COLOR_NOPASS,
                           style="dotted", arrows=True, arrowstyle="-|>", arrowsize=ARROW_SIZE, alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=pass_edges, edge_color=EDGE_COLOR_PASS,
                           style="solid", arrows=True, arrowstyle="-|>", arrowsize=ARROW_SIZE, alpha=0.8)

    # draw path edges on top (highlight)
    path_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("is_on_path")]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=EDGE_COLOR_PATH,
                           width=2.5, arrows=True, arrowstyle="-|>", arrowsize=ARROW_SIZE)

    # draw nodes by role
    roles = defaultdict(list)
    for n, d in G.nodes(data=True):
        roles[d.get("role", "intermediate")].append(n)

    for role, nodes_list in roles.items():
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes_list, node_color=COLORS.get(role, COLORS["intermediate"]),
            node_size=NODE_SIZE, linewidths=0.8, edgecolors="black"
        )

    # labels: show node id; optionally add hop for requestor/provider/intermediates
    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=FONT_SIZE)

    # annotate trust on PATH edges only (keeps figure readable)
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        if d.get("is_on_path"):
            edge_labels[(u, v)] = f'{d.get("trust", 0):.2f}'
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=FONT_SIZE-1, label_pos=0.5)

    # title
    title_bits = []
    if meta.get("dataset"):   title_bits.append(str(meta["dataset"]))
    if meta.get("strategy"):  title_bits.append(f"strategy={meta['strategy']}")
    if meta.get("alpha") is not None:     title_bits.append(f"α={meta['alpha']}")
    if meta.get("top_k") is not None:     title_bits.append(f"K={meta['top_k']}")
    if meta.get("max_hops") is not None:  title_bits.append(f"H={meta['max_hops']}")
    if meta.get("success") is not None:   title_bits.append(f"success={meta['success']}")
    if meta.get("hops") not in (None, 0): title_bits.append(f"hops={meta['hops']}")
    if meta.get("visited_count") is not None: title_bits.append(f"visited={meta['visited_count']}")

    plt.title(" | ".join(title_bits), fontsize=11)
    plt.axis("off")
    plt.tight_layout()

    # save beside CSVs
    png_path = os.path.join(trace_dir, "trace_vis.png")
    pdf_path = os.path.join(trace_dir, "trace_vis.pdf")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")


def main():
    if not os.path.isdir(TRACES_DIR):
        print(f"[ERROR] traces dir not found: {TRACES_DIR}")
        sys.exit(1)

    made = 0
    for name in sorted(os.listdir(TRACES_DIR)):
        d = os.path.join(TRACES_DIR, name)
        if not os.path.isdir(d):
            continue
        trace = read_trace(d)
        if trace is None:
            continue
        try:
            draw_trace(d, trace)
            made += 1
        except Exception as e:
            print(f"[warn] failed to plot {name}: {e}")

    print(f"Done. Plotted {made} trace(s).")


if __name__ == "__main__":
    main()
