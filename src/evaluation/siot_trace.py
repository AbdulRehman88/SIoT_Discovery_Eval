# siot_trace.py
import json, csv, random, math, os, argparse, pickle, sys
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, List, Tuple, Optional
import networkx as nx

@dataclass
class StepEdge:
    src: int
    dst: int
    hop: int
    trust: float
    passes_alpha: bool
    is_on_path: bool = False

@dataclass
class TraceSummary:
    dataset: str
    strategy: str        # "greedy" or "fallback"
    alpha: float
    top_k: int
    max_hops: int
    requestor_id: int
    provider_id: Optional[int]
    success: bool
    hops: int
    visited_count: int

class TraceRecorder:
    def __init__(self):
        self.nodes: Dict[int, Dict[str, Any]] = {}
        self.edges: List[StepEdge] = []

    def add_node(self, node: int, role: str, hop: int, trust: Optional[float] = None):
        # role in {"requestor","provider","intermediate"}
        if node not in self.nodes:
            self.nodes[node] = {"node": node, "role": role, "hop": hop, "trust": trust}
        else:
            # keep earliest hop; update role to provider if discovered
            self.nodes[node]["hop"] = min(self.nodes[node]["hop"], hop)
            if role == "provider":
                self.nodes[node]["role"] = "provider"
            if trust is not None:
                self.nodes[node]["trust"] = trust

    def add_edge(self, src: int, dst: int, hop: int, trust: float, passes_alpha: bool):
        self.edges.append(StepEdge(src=src, dst=dst, hop=hop, trust=trust, passes_alpha=passes_alpha))

    def mark_path(self, path_nodes: List[int]):
        path_set = set()
        for i in range(len(path_nodes)-1):
            path_set.add((path_nodes[i], path_nodes[i+1]))
        for e in self.edges:
            if (e.src, e.dst) in path_set:
                e.is_on_path = True

    def save(self, out_dir: str, summary: TraceSummary):
        os.makedirs(out_dir, exist_ok=True)
        # nodes
        with open(os.path.join(out_dir, "trace_nodes.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["node","role","hop","trust"])
            w.writeheader()
            for v in self.nodes.values():
                w.writerow(v)
        # edges
        with open(os.path.join(out_dir, "trace_edges.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["src","dst","hop","trust","passes_alpha","is_on_path"])
            w.writeheader()
            for e in self.edges:
                w.writerow(asdict(e))
        # jsonl summary
        with open(os.path.join(out_dir, "trace.jsonl"), "w") as f:
            f.write(json.dumps(asdict(summary)) + "\n")

def default_rating_fn(G: nx.Graph, u: int, v: int) -> float:
    """Fallback if you don’t plug your trust rating; uses edge attr 'trust' then 'weight' or 1.0."""
    data = G.get_edge_data(u, v) or {}
    w = data.get("trust", data.get("weight", 1.0))
    try:
        return float(w)
    except Exception:
        return 1.0

def greedy_discovery_with_trace(
    G: nx.Graph,
    start: int,
    is_target_fn: Callable[[int], bool],
    rating_fn: Callable[[nx.Graph, int, int], float],
    class_match_fn: Callable[[int], bool],
    alpha: float,
    top_k: int,
    max_hops: int,
    strategy: str,              # "greedy" or "fallback"
    rng: random.Random,
    recorder: TraceRecorder
) -> Tuple[bool, Optional[int], List[int], int, int]:
    """
    Returns: (success, provider_id, path_nodes, hops, visited_count)
    """
    visited = set()
    frontier: List[int] = [start]
    parents: Dict[int, Optional[int]] = {start: None}
    hop_of: Dict[int, int] = {start: 0}
    recorder.add_node(start, role="requestor", hop=0, trust=None)

    while frontier:
        u = frontier.pop(0)
        visited.add(u)
        hop_u = hop_of[u]

        # stop if hop limit reached (don’t expand further)
        if hop_u >= max_hops:
            continue

        # candidate neighbors, ranked by rating descending, filtered by class and alpha
        neighs = []
        for v in G.neighbors(u):
            if v in visited or v in frontier:
                continue
            if not class_match_fn(v):
                continue
            score = rating_fn(G, u, v)
            passes = (score >= alpha)
            recorder.add_edge(u, v, hop_u+1, score, passes_alpha=passes)
            if passes:
                neighs.append((v, score))

        # sort by trust score
        neighs.sort(key=lambda x: x[1], reverse=True)

        # if none pass alpha and strategy == "fallback", relax to the top_k of all neighbors (still class-matched)
        if not neighs and strategy.lower() == "fallback":
            tmp = []
            for v in G.neighbors(u):
                if v in visited or v in frontier:
                    continue
                if not class_match_fn(v):
                    continue
                score = rating_fn(G, u, v)
                tmp.append((v, score))
            tmp.sort(key=lambda x: x[1], reverse=True)
            neighs = tmp[:top_k]

        # restrict breadth
        neighs = neighs[:top_k]

        # push to frontier
        for v, sc in neighs:
            parents[v] = u
            hop_of[v] = hop_u + 1
            frontier.append(v)
            role = "intermediate"
            if is_target_fn(v):
                # reconstruct path and finish
                path = [v]
                cur = v
                while parents[cur] is not None:
                    cur = parents[cur]
                    path.append(cur)
                path.reverse()
                hops = hop_of[v]
                # record nodes
                for i, pn in enumerate(path):
                    recorder.add_node(
                        pn,
                        role=("provider" if i == len(path)-1 else ("requestor" if i==0 else "intermediate")),
                        hop=i, trust=None
                    )
                recorder.mark_path(path)
                return True, v, path, hops, len(visited) + len(frontier)

            recorder.add_node(v, role=role, hop=hop_of[v], trust=sc)

    return False, None, [], math.inf, len(visited)

def trace_one_run(
    G: nx.Graph,
    dataset: str,
    strategy: str,
    alpha: float,
    top_k: int = 6,
    max_hops: int = 6,
    seed: int = 42,
    out_dir: str = "./trace_output",
    rating_fn: Callable[[nx.Graph, int, int], float] = default_rating_fn,
    class_match_fn: Callable[[int], bool] = lambda v: True,
    target_picker: Optional[Callable[[nx.Graph, random.Random], int]] = None,
):
    rng = random.Random(seed)
    # pick a requestor with degree>0
    candidates = [n for n, d in G.degree() if d > 0]
    if not candidates:
        raise RuntimeError("Graph has no nodes with degree > 0.")
    start = rng.choice(candidates)

    # pick a provider (target) distinct from start; prefer far reachable node if possible
    if target_picker is None:
        reachable = []
        for n in candidates:
            try:
                if nx.has_path(G, start, n):
                    reachable.append((n, nx.shortest_path_length(G, start, n)))
            except nx.NetworkXNoPath:
                continue
        if reachable:
            reachable.sort(key=lambda t: t[1])  # by distance
            target = reachable[-1][0] if reachable[-1][0] != start else (reachable[-2][0] if len(reachable) > 1 else start)
        else:
            # fall back: any other node
            others = [n for n in candidates if n != start]
            target = rng.choice(others) if others else start
    else:
        target = target_picker(G, rng)

    is_target_fn = lambda v: v == target

    recorder = TraceRecorder()
    success, provider, path, hops, visited_count = greedy_discovery_with_trace(
        G, start, is_target_fn, rating_fn, class_match_fn,
        alpha=alpha, top_k=top_k, max_hops=max_hops,
        strategy=strategy, rng=rng, recorder=recorder
    )

    summary = TraceSummary(
        dataset=dataset, strategy=strategy, alpha=alpha, top_k=top_k, max_hops=max_hops,
        requestor_id=start, provider_id=provider, success=success,
        hops=(hops if success else 0), visited_count=visited_count
    )
    recorder.save(out_dir, summary)
    return summary

# ----------------------- CLI MAIN -----------------------

def load_graph(graph_path: str) -> nx.Graph:
    ext = os.path.splitext(graph_path)[1].lower()
    if ext in [".gpickle", ".pickle", ".pkl"]:
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        return G
    elif ext in [".edgelist", ".csv", ".txt"]:
        # naive CSV/edgelist loader: try src,dst,weight
        try:
            G = nx.read_edgelist(graph_path, delimiter=",", data=[("weight", float)], create_using=nx.DiGraph())
        except Exception:
            G = nx.read_edgelist(graph_path, data=[("weight", float)], create_using=nx.DiGraph())
        return G
    else:
        raise ValueError(f"Unsupported graph format: {ext}")

def rating_from_edge(G: nx.Graph, u: int, v: int) -> float:
    data = G.get_edge_data(u, v) or {}
    val = data.get("trust", data.get("weight", 1.0))
    try:
        return float(val)
    except Exception:
        return 1.0

def main():
    parser = argparse.ArgumentParser(description="Trace a single SIoT discovery run and save path files.")
    parser.add_argument("--graph", required=True, help="Path to graph (.gpickle recommended).")
    parser.add_argument("--dataset", required=True, help="Dataset name label (for titles/files).")
    parser.add_argument("--strategy", default="greedy", choices=["greedy", "fallback"], help="Discovery strategy.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Trust threshold α.")
    parser.add_argument("--top_k", type=int, default=6, help="Top-K neighbors to expand per hop.")
    parser.add_argument("--max_hops", type=int, default=6, help="Max hops (depth) limit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out_dir", required=True, help="Output directory to write trace files.")

    args = parser.parse_args()

    G = load_graph(args.graph)

    # Ensure node ids are hashable and consistent
    # (NetworkX supports str/int; we keep as-is)
    summary = trace_one_run(
        G=G,
        dataset=args.dataset,
        strategy=args.strategy,
        alpha=args.alpha,
        top_k=args.top_k,
        max_hops=args.max_hops,
        seed=args.seed,
        out_dir=args.out_dir,
        rating_fn=rating_from_edge,
        class_match_fn=lambda v: True
    )

    print(f"[OK] Saved trace to: {args.out_dir}")
    print(f"     -> trace_nodes.csv, trace_edges.csv, trace.jsonl")
    print(f"Summary: {summary}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
