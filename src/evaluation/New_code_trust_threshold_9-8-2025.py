# siot_trust_sensitivity_topk_progress_LCC.py
# Trust-based Top-K Greedy SIoT discovery + fallback.
# MAX_HOPS=6, TOP_K=6, TRIALS_PER_ALPHA=100.
# Parallel trials, global + per-dataset progress bars, CSV checkpointing,
# logs to file, and plots for AvgVisited and LCC ratio.

import os
import yaml
import time
import pickle
import logging
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==============================
# USER KNOBS
# ==============================
SEED = 42
MAX_HOPS = 6             # <= your claim
TOP_K = 6                # <= expand only top-6 trusted neighbors
TRIALS_PER_ALPHA = 100   # final runs per α
WORKERS = max(4, (os.cpu_count() or 8) // 2)
TRIAL_TIMEOUT_SEC = None
LCC_RATIO_GATE = 0.50    # below this we "carry forward" metrics but still record LCCRatio

CONFIG_PATH = "D:/Dr_Abdul_Rehman/MyPycharm/August 02 2025/SIoT_Discovery_Eval/config.yaml"
RESULTS_BASENAME = "trust_threshold_sensitivity_top6_trials100"
LOG_FILENAME = f"{RESULTS_BASENAME}_run.log"
# ==============================


# ---------- Logging ----------
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, LOG_FILENAME)

    logger = logging.getLogger("siot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("=== SIoT Top-K Greedy Discovery Run ===")
    logger.info(f"TOP_K={TOP_K}, MAX_HOPS={MAX_HOPS}, TRIALS_PER_ALPHA={TRIALS_PER_ALPHA}, "
                f"WORKERS={WORKERS}, TIMEOUT={TRIAL_TIMEOUT_SEC}, LCC_GATE={LCC_RATIO_GATE}")
    logger.info(f"Logging to: {log_path}")
    return logger, log_path


# ---------- I/O ----------
def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_partial(greedy_rows, fallback_rows, greedy_csv, fallback_csv):
    pd.DataFrame(greedy_rows).to_csv(greedy_csv, index=False)
    pd.DataFrame(fallback_rows).to_csv(fallback_csv, index=False)


# ---------- Graph utilities ----------
def filter_graph_by_alpha(G, alpha, trust_default=1.0):
    Gf = G.copy()
    to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get("trust", trust_default) < alpha]
    Gf.remove_edges_from(to_remove)
    return Gf


def largest_cc_nodes_undirected(G):
    GU = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    comp_gen = nx.connected_components(GU)
    try:
        return max(comp_gen, key=len)
    except ValueError:
        return set()


# ---------- Core algorithms ----------
def _sorted_neighbors_by_trust(G, u):
    nbrs = []
    for v in G.neighbors(u):
        trust = G[u][v].get("trust", 0.0)
        nbrs.append((v, trust))
    nbrs.sort(key=lambda t: (-t[1], t[0]))
    return nbrs


def run_greedy_discovery_topk(G, source, alpha=0.5, service_label="target_service", max_hops=6, top_k=6):
    visited = set([source])
    q = deque([(source, 0)])
    path_trace = []
    nodes_expanded = 0

    while q:
        node, depth = q.popleft()
        nodes_expanded += 1
        path_trace.append(node)
        if depth > max_hops:
            break

        if G.nodes[node].get("service") == service_label:
            return {"success": True, "hops": depth, "path": path_trace,
                    "visited_count": nodes_expanded, "used_fallback": False}

        added = 0
        for v, trust in _sorted_neighbors_by_trust(G, node):
            if v in visited:
                continue
            if trust >= alpha:
                visited.add(v)
                q.append((v, depth + 1))
                added += 1
                if added >= top_k:
                    break

    return {"success": False, "hops": 0, "path": path_trace,
            "visited_count": nodes_expanded, "used_fallback": False}


def run_fallback_discovery_topk(G, source, alpha=0.5, service_label="target_service", max_hops=6, top_k=6):
    visited = set([source])
    q = deque([(source, 0)])
    path_trace = []
    nodes_expanded = 0
    fallback_used = False

    while q:
        node, depth = q.popleft()
        nodes_expanded += 1
        path_trace.append(node)
        if depth > max_hops:
            break

        if G.nodes[node].get("service") == service_label:
            return {"success": True, "hops": depth, "path": path_trace,
                    "visited_count": nodes_expanded, "used_fallback": fallback_used}

        nbrs = _sorted_neighbors_by_trust(G, node)

        added = 0
        for v, trust in nbrs:
            if v in visited:
                continue
            if trust >= alpha:
                visited.add(v)
                q.append((v, depth + 1))
                added += 1
                if added >= top_k:
                    break

        if added == 0 and not fallback_used:
            fallback_used = True
            added2 = 0
            for v, _trust in nbrs:
                if v in visited:
                    continue
                visited.add(v)
                q.append((v, depth + 1))
                added2 += 1
                if added2 >= top_k:
                    break

    return {"success": False, "hops": 0, "path": path_trace,
            "visited_count": nodes_expanded, "used_fallback": fallback_used}


# ---------- Trial worker ----------
def _one_trial_worker(pickled_G, alpha, alpha_idx, run_idx, service_label, use_fallback, max_hops, top_k):
    import pickle as _pickle
    import numpy as _np
    import time as _time

    G = _pickle.loads(pickled_G)
    run_rng = _np.random.default_rng(SEED + alpha_idx * 1000 + run_idx)

    nodes = list(G.nodes)
    if not nodes:
        return {"success": False, "hops": 0, "used_fallback": False, "visited": 0, "elapsed": 0.0}

    num_targets = min(max(1, int(0.05 * len(nodes))), 5)
    targets = set(run_rng.choice(nodes, num_targets, replace=False))
    for n in targets:
        G.nodes[n]["service"] = service_label

    sources = [n for n in nodes if n not in targets]
    if not sources:
        return {"success": False, "hops": 0, "used_fallback": False, "visited": 0, "elapsed": 0.0}
    src = run_rng.choice(sources)

    start = _time.time()
    try:
        if use_fallback:
            res = run_fallback_discovery_topk(G, src, alpha=alpha, service_label=service_label,
                                              max_hops=max_hops, top_k=top_k)
        else:
            res = run_greedy_discovery_topk(G, src, alpha=alpha, service_label=service_label,
                                            max_hops=max_hops, top_k=top_k)
    except Exception:
        res = {"success": False, "hops": 0, "used_fallback": use_fallback, "visited": 0}
    elapsed = _time.time() - start

    return {
        "success": bool(res.get("success", False)),
        "hops": int(res.get("hops", 0) or 0),
        "used_fallback": bool(res.get("used_fallback", False)),
        "visited": int(res.get("visited_count", 0) or 0),
        "elapsed": float(elapsed)
    }


# ---------- Evaluation ----------
def evaluate_graph_parallel(G, alpha, alpha_idx, service_label="target_service", use_fallback=False,
                            workers=WORKERS, trials=TRIALS_PER_ALPHA, trial_timeout=TRIAL_TIMEOUT_SEC,
                            max_hops=MAX_HOPS, top_k=TOP_K):
    successes = 0
    hops_list, time_list, visited_list = [], [], []
    fb_needed, fb_used = 0, 0

    pickled_G = pickle.dumps(G, protocol=pickle.HIGHEST_PROTOCOL)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_one_trial_worker, pickled_G, alpha, alpha_idx, run_idx,
                      service_label, use_fallback, max_hops, top_k)
            for run_idx in range(trials)
        ]
        for fut in as_completed(futures):
            try:
                res = fut.result(timeout=trial_timeout) if trial_timeout else fut.result()
            except Exception:
                res = {"success": False, "hops": 0, "used_fallback": False, "visited": 0, "elapsed": 0.0}

            if res.get("used_fallback"):
                fb_needed += 1
                if res.get("success"):
                    fb_used += 1

            if res.get("success"):
                successes += 1
                hops_list.append(res.get("hops", 0))

            time_list.append(res.get("elapsed", 0.0))
            visited_list.append(res.get("visited", 0))

    success_rate = round(successes / trials if trials else 0.0, 3)
    avg_hops = round(float(np.mean(hops_list)) if hops_list else 0.0, 2)
    avg_time = round(float(np.mean(time_list)) if time_list else 0.0, 4)
    avg_visited = round(float(np.mean(visited_list)) if visited_list else 0.0, 2)

    return {
        "SuccessRate": success_rate,
        "AvgHops": avg_hops,
        "AvgTime": avg_time,
        "AvgVisited": avg_visited,
        "Trials": int(trials),
        "FallbackUsed": int(fb_used),
        "FallbackNeeded": int(fb_needed)
    }


# ---------- Plotting ----------
def _plot_combined_metric(df, metric, out_path, ylabel, title):
    plt.figure(figsize=(12, 6))
    for ds, g in df.groupby("Dataset"):
        g = g.sort_values("Alpha")
        plt.plot(g["Alpha"], g[metric], marker="o", label=f"{ds}")
    plt.xlabel("Trust Threshold α")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_plots(results_dir, greedy_csv, fallback_csv):
    # AvgVisited (use greedy by default; fallback similar)
    gdf = pd.read_csv(greedy_csv)
    fdf = pd.read_csv(fallback_csv)

    # Plot AvgVisited (greedy)
    out1 = os.path.join(results_dir, "combined_avg_visited_vs_alpha.png")
    _plot_combined_metric(gdf, "AvgVisited", out1, "Average Nodes Visited",
                          "Average Nodes Visited vs Trust Threshold α (Greedy)")

    # Plot LCC ratio (identical across greedy/fallback; take from greedy)
    out2 = os.path.join(results_dir, "combined_lcc_ratio_vs_alpha.png")
    _plot_combined_metric(gdf, "LCCRatio", out2, "Largest Connected Component Ratio",
                          "LCC Ratio vs Trust Threshold α")


# ---------- Main ----------
def main():
    config = load_config()
    base = config["base_dir"]
    processed_dir = os.path.join(base, config["data_paths"]["processed_dir"])
    results_dir = os.path.join(base, "results")
    os.makedirs(results_dir, exist_ok=True)

    logger, log_path = setup_logger(results_dir)

    greedy_csv = os.path.join(results_dir, f"{RESULTS_BASENAME}_greedy.csv")
    fallback_csv = os.path.join(results_dir, f"{RESULTS_BASENAME}_fallback.csv")

    alphas = [round(a, 1) for a in np.arange(0.1, 1.0, 0.1)]
    datasets = list(config["datasets"])

    greedy_rows, fallback_rows = [], []

    logger.info(f"Output CSVs:\n  {greedy_csv}\n  {fallback_csv}\n")

    # Global progress over datasets × α (we tick once per α; modes run inside)
    total_alpha_modes = len(datasets) * len(alphas)
    global_bar = tqdm(total=total_alpha_modes, desc="Global progress (datasets×α)", unit="α", leave=True)

    for dataset_name in tqdm(datasets, desc="Datasets", unit="ds", leave=False):
        graph_path = os.path.join(processed_dir, f"{dataset_name}.gpickle")
        if not os.path.exists(graph_path):
            msg = f"⚠️  {dataset_name} missing. Skipping."
            logger.info(msg); tqdm.write(msg)
            # still advance the global bar for this dataset's alphas
            global_bar.update(len(alphas))
            continue

        with open(graph_path, "rb") as f:
            G_original = pickle.load(f)

        n0, m0 = G_original.number_of_nodes(), G_original.number_of_edges()
        header = f"=== {dataset_name}: nodes={n0}, edges={m0} ==="
        logger.info(header); tqdm.write("\n" + header)

        ds_bar = tqdm(alphas, desc=f"{dataset_name}: α sweep", unit="α", leave=False)

        last_greedy, last_fallback = None, None
        for idx, alpha in enumerate(ds_bar):
            start_alpha = time.time()

            # 1) Trust filter
            G_alpha = filter_graph_by_alpha(G_original, alpha)

            # 2) Largest CC nodes and ratio (always compute & store)
            lcc_nodes = largest_cc_nodes_undirected(G_alpha)
            lcc_ratio = len(lcc_nodes) / max(1, n0)

            if not lcc_nodes or lcc_ratio < LCC_RATIO_GATE:
                # Carry-forward previous metrics but record LCCRatio
                msg = (f"⚠️  [{dataset_name}] α={alpha:.1f}: "
                       f"{'no components' if not lcc_nodes else 'poor connectivity'} "
                       f"(LCC={lcc_ratio:.2f}) — carry forward.")
                logger.info(msg); tqdm.write(msg)

                g = last_greedy or {"SuccessRate": 0.0, "AvgHops": 0.0, "AvgTime": 0.0,
                                    "AvgVisited": 0.0, "Trials": 0, "FallbackUsed": 0, "FallbackNeeded": 0}
                f = last_fallback or {"SuccessRate": 0.0, "AvgHops": 0.0, "AvgTime": 0.0,
                                      "AvgVisited": 0.0, "Trials": 0, "FallbackUsed": 0, "FallbackNeeded": 0}

                g_row = {"Dataset": dataset_name, "Alpha": alpha, "LCCRatio": round(lcc_ratio, 3), **g}
                f_row = {"Dataset": dataset_name, "Alpha": alpha, "LCCRatio": round(lcc_ratio, 3), **f}
                greedy_rows.append(g_row); fallback_rows.append(f_row)
                save_partial(greedy_rows, fallback_rows, greedy_csv, fallback_csv)
                global_bar.update(1)
                continue

            # 3) Evaluate on LCC
            G_lcc = G_alpha.subgraph(lcc_nodes).copy()

            t0 = time.time()
            greedy_res = evaluate_graph_parallel(G_lcc, alpha, idx, service_label="target_service",
                                                 use_fallback=False, workers=WORKERS,
                                                 trials=TRIALS_PER_ALPHA, trial_timeout=TRIAL_TIMEOUT_SEC,
                                                 max_hops=MAX_HOPS, top_k=TOP_K)
            t1 = time.time()
            fallback_res = evaluate_graph_parallel(G_lcc, alpha, idx, service_label="target_service",
                                                   use_fallback=True, workers=WORKERS,
                                                   trials=TRIALS_PER_ALPHA, trial_timeout=TRIAL_TIMEOUT_SEC,
                                                   max_hops=MAX_HOPS, top_k=TOP_K)
            t2 = time.time()

            line = (f"[{dataset_name}] α={alpha:.1f}  "
                    f"greedy SR={greedy_res['SuccessRate']:.3f}, hops={greedy_res['AvgHops']:.2f}, "
                    f"visited={greedy_res['AvgVisited']:.1f}, time={t1 - t0:.1f}s  |  "
                    f"fallback SR={fallback_res['SuccessRate']:.3f}, hops={fallback_res['AvgHops']:.2f}, "
                    f"visited={fallback_res['AvgVisited']:.1f}, time={t2 - t1:.1f}s  "
                    f"(LCC={lcc_ratio:.2f}, α elapsed {time.time() - start_alpha:.1f}s)")
            logger.info(line); tqdm.write(line)

            g_row = {"Dataset": dataset_name, "Alpha": alpha, "LCCRatio": round(lcc_ratio, 3), **greedy_res}
            f_row = {"Dataset": dataset_name, "Alpha": alpha, "LCCRatio": round(lcc_ratio, 3), **fallback_res}
            greedy_rows.append(g_row); fallback_rows.append(f_row)
            last_greedy, last_fallback = g_row, f_row

            save_partial(greedy_rows, fallback_rows, greedy_csv, fallback_csv)
            global_bar.update(1)

        ds_bar.close()

    global_bar.close()
    logger.info(f"\n✅ Completed. Results saved to:\n → {greedy_csv}\n → {fallback_csv}")

    # Make the two new combined plots
    make_plots(results_dir, greedy_csv, fallback_csv)
    logger.info(f"Additional plots saved to: {results_dir}")
    print(f"\nLog file: {os.path.join(results_dir, LOG_FILENAME)}")


if __name__ == "__main__":
    main()
