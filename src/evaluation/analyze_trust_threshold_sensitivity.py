# import os
# import yaml
# import numpy as np
# import pandas as pd
# import networkx as nx
# from tqdm import tqdm
# from time import time
# import pickle
#
# from src.discovery.greedy_discovery import run_greedy_discovery  # Must be defined in your codebase
#
# def load_config(config_path="D:/Dr_Abdul_Rehman/MyPycharm/August 02 2025/SIoT_Discovery_Eval/config.yaml"):
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"❌ Config file not found at: {config_path}")
#     with open(config_path, "r") as f:
#         return yaml.safe_load(f)
#
# def filter_graph_by_alpha(G, alpha):
#     """Remove edges with trust below alpha."""
#     G_filtered = G.copy()
#     edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get("trust", 1.0) < alpha]
#     G_filtered.remove_edges_from(edges_to_remove)
#     return G_filtered
#
# def assign_target_services(G, service_label="target_service", fraction=0.05):
#     """Randomly assign a target service label to a small fraction of nodes."""
#     nodes = list(G.nodes)
#     if not nodes:
#         return G
#     num_targets = max(1, int(fraction * len(nodes)))
#     num_targets = min(num_targets, len(nodes))
#     target_nodes = np.random.choice(nodes, num_targets, replace=False)
#     for node in target_nodes:
#         G.nodes[node]["service"] = service_label
#     return G
#
# def evaluate_graph(G, alpha, service_label="target_service", num_runs=100):
#     """Run greedy discovery multiple times and average the results."""
#     success_rates = []
#     avg_hops = []
#     avg_times = []
#
#     for _ in range(num_runs):
#         G_temp = assign_target_services(G.copy(), service_label=service_label, fraction=0.05)
#
#         successful = 0
#         total_hops = []
#         total_time = []
#
#         nodes = list(G_temp.nodes)
#         source_pool = [n for n in nodes if G_temp.nodes[n].get("service") != service_label]
#         if not source_pool:
#             continue
#         num_trials = 100
#         source_nodes = np.random.choice(source_pool, num_trials, replace=True)
#         for src in source_nodes:
#             try:
#                 start = time()
#                 result = run_greedy_discovery(G_temp, source=src, alpha=alpha, service_label=service_label)
#                 end = time()
#                 if result.get("success"):
#                     successful += 1
#                     total_hops.append(result["hops"])
#                     total_time.append(end - start)
#             except Exception:
#                 continue
#
#         run_success = successful / len(source_nodes)
#         run_hops = np.mean(total_hops) if total_hops else 0
#         run_time = np.mean(total_time) if total_time else 0
#
#         success_rates.append(run_success)
#         avg_hops.append(run_hops)
#         avg_times.append(run_time)
#
#     # Final stable results
#     final_success = np.mean(success_rates)
#     final_hops = np.mean(avg_hops)
#     final_time = np.mean(avg_times)
#     return final_success, final_hops, final_time
#
# def main():
#     config = load_config()
#     processed_dir = os.path.join(config["base_dir"], config["data_paths"]["processed_dir"])
#     output_path = os.path.join(config["base_dir"], "results", "trust_threshold_sensitivity.csv")
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#     alphas = [round(a, 1) for a in np.arange(0.1, 1.0, 0.1)]
#     results = []
#
#     for dataset_name in config["datasets"]:
#         graph_path = os.path.join(processed_dir, f"{dataset_name}.gpickle")
#         if not os.path.exists(graph_path):
#             print(f"⚠️ Skipping {dataset_name}: file not found.")
#             continue
#
#         print(f"\n🔍 Evaluating {dataset_name}...")
#         with open(graph_path, "rb") as f:
#             G_original = pickle.load(f)
#
#         last_valid_result = None
#         last_valid_alpha = None
#
#         for alpha in alphas:
#             G_alpha = filter_graph_by_alpha(G_original, alpha)
#
#             # Connectivity check
#             G_temp = G_alpha if isinstance(G_alpha, nx.Graph) else G_alpha.to_undirected()
#             # Safely convert to undirected graph if it's directed
#             if isinstance(G_alpha, nx.DiGraph):
#                 G_temp = G_alpha.to_undirected()
#             else:
#                 G_temp = G_alpha
#
#             components = list(nx.connected_components(G_temp))
#             largest_cc = max(components, key=len)
#             cc_ratio = len(largest_cc) / G_temp.number_of_nodes()
#             largest_cc = max(components, key=len) if components else set()
#             lcc_ratio = len(largest_cc) / G_original.number_of_nodes() if G_original.number_of_nodes() > 0 else 0
#
#             # Handle disconnected graphs
#             if lcc_ratio < 0.5:
#                 print(f"⚠️ Skipping α={alpha} for {dataset_name} — poor connectivity (LCC={lcc_ratio:.2f})")
#                 if last_valid_result:
#                     results.append({
#                         "Dataset": dataset_name,
#                         "Alpha": alpha,
#                         **last_valid_result
#                     })
#                 else:
#                     results.append({
#                         "Dataset": dataset_name,
#                         "Alpha": alpha,
#                         "SuccessRate": 0.0,
#                         "AvgHops": 0.0,
#                         "AvgTime": 0.0
#                     })
#                 continue
#
#             # Evaluate only on largest connected component
#             G_lcc = G_alpha.subgraph(largest_cc).copy()
#             G_lcc = assign_target_services(G_lcc, service_label="target_service", fraction=0.05)
#
#             # Optional debug
#             if dataset_name == "FB_Forum":
#                 print(f"\n🔎 DEBUG — {dataset_name} @ α={alpha}")
#                 print(f"   → Nodes: {G_lcc.number_of_nodes()}, Edges: {G_lcc.number_of_edges()}")
#                 print(f"   → Largest CC size ratio: {lcc_ratio:.2f}")
#                 trust_vals = [d["trust"] for _, _, d in G_lcc.edges(data=True) if "trust" in d]
#                 if trust_vals:
#                     print(f"   → Trust stats: min={min(trust_vals):.3f}, max={max(trust_vals):.3f}, count={len(trust_vals)}")
#
#             # Evaluate
#             success, hops, time_cost = evaluate_graph(G_lcc, alpha)
#             result_dict = {
#                 "SuccessRate": round(success, 3),
#                 "AvgHops": round(hops, 2),
#                 "AvgTime": round(time_cost, 4)
#             }
#             results.append({
#                 "Dataset": dataset_name,
#                 "Alpha": alpha,
#                 **result_dict
#             })
#             last_valid_result = result_dict
#             last_valid_alpha = alpha
#
#     df = pd.DataFrame(results)
#     df.to_csv(output_path, index=False)
#     print(f"\n✅ Sensitivity analysis completed. Results saved to: {output_path}")
#
# if __name__ == "__main__":
#     main()





#================================= New code for both our greedy approach and fallback ===========

# import os
# import yaml
# import numpy as np
# import pandas as pd
# import networkx as nx
# import pickle
# from time import time
# from tqdm import tqdm
#
# from src.discovery.greedy_discovery import run_greedy_discovery, run_fallback_discovery
#
#
# def load_config(config_path="D:/Dr_Abdul_Rehman/MyPycharm/August 02 2025/SIoT_Discovery_Eval/config.yaml"):
#     with open(config_path, "r") as f:
#         return yaml.safe_load(f)
#
#
# def filter_graph_by_alpha(G, alpha):
#     G_filtered = G.copy()
#     G_filtered.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d.get("trust", 1.0) < alpha])
#     return G_filtered
#
#
# def assign_target_services(G, label="target_service", fraction=0.05):
#     nodes = list(G.nodes)
#     num_targets = max(1, int(fraction * len(nodes)))
#     target_nodes = np.random.choice(nodes, num_targets, replace=False)
#     for node in target_nodes:
#         G.nodes[node]["service"] = label
#     return G
#
#
# def evaluate_graph(G, alpha, service_label="target_service", use_fallback=False, num_trials=100):
#     success_list, hop_list, time_list = [], [], []
#
#     for _ in range(100):  # 100 runs for statistical reliability
#         G_copy = assign_target_services(G.copy(), label=service_label, fraction=0.05)
#         sources = [n for n in G_copy.nodes if G_copy.nodes[n].get("service") != service_label]
#         if not sources:
#             continue
#         chosen_sources = np.random.choice(sources, num_trials, replace=True)
#
#         for src in chosen_sources:
#             try:
#                 start = time()
#                 result = run_fallback_discovery(G_copy, src, alpha, service_label) if use_fallback \
#                     else run_greedy_discovery(G_copy, src, alpha, service_label)
#                 end = time()
#                 if result["success"]:
#                     success_list.append(1)
#                     hop_list.append(result["hops"])
#                     time_list.append(end - start)
#             except:
#                 continue
#
#     return np.mean(success_list), np.mean(hop_list), np.mean(time_list)
#
#
# def main():
#     config = load_config()
#     base = config["base_dir"]
#     processed_dir = os.path.join(base, config["data_paths"]["processed_dir"])
#     results_dir = os.path.join(base, "results")
#     os.makedirs(results_dir, exist_ok=True)
#
#     greedy_csv = os.path.join(results_dir, "trust_threshold_sensitivity_greedy.csv")
#     fallback_csv = os.path.join(results_dir, "trust_threshold_sensitivity_fallback.csv")
#
#     alphas = [round(a, 1) for a in np.arange(0.1, 1.0, 0.1)]
#     greedy_results = []
#     fallback_results = []
#
#     for dataset_name in config["datasets"]:
#         graph_path = os.path.join(processed_dir, f"{dataset_name}.gpickle")
#         if not os.path.exists(graph_path):
#             print(f"⚠️ {dataset_name} missing. Skipping.")
#             continue
#
#         with open(graph_path, "rb") as f:
#             G_original = pickle.load(f)
#
#         last_greedy, last_fallback = None, None
#
#         for alpha in alphas:
#             G_alpha = filter_graph_by_alpha(G_original, alpha)
#             G_temp = G_alpha.to_undirected() if isinstance(G_alpha, nx.DiGraph) else G_alpha
#             components = list(nx.connected_components(G_temp))
#
#             if not components:
#                 continue
#
#             largest_cc = max(components, key=len)
#             lcc_ratio = len(largest_cc) / G_original.number_of_nodes()
#
#             if lcc_ratio < 0.5:
#                 print(f"⚠️ α={alpha} for {dataset_name}: poor connectivity (LCC={lcc_ratio:.2f})")
#                 greedy_results.append({
#                     "Dataset": dataset_name, "Alpha": alpha,
#                     **(last_greedy if last_greedy else {"SuccessRate": 0.0, "AvgHops": 0.0, "AvgTime": 0.0})
#                 })
#                 fallback_results.append({
#                     "Dataset": dataset_name, "Alpha": alpha,
#                     **(last_fallback if last_fallback else {"SuccessRate": 0.0, "AvgHops": 0.0, "AvgTime": 0.0})
#                 })
#                 continue
#
#             G_lcc = G_alpha.subgraph(largest_cc).copy()
#
#             # Run Greedy
#             s_g, h_g, t_g = evaluate_graph(G_lcc, alpha, use_fallback=False)
#             greedy_row = {"Dataset": dataset_name, "Alpha": alpha,
#                           "SuccessRate": round(s_g, 3), "AvgHops": round(h_g, 2), "AvgTime": round(t_g, 4)}
#             greedy_results.append(greedy_row)
#             last_greedy = greedy_row
#
#             # Run Fallback
#             s_f, h_f, t_f = evaluate_graph(G_lcc, alpha, use_fallback=True)
#             fallback_row = {"Dataset": dataset_name, "Alpha": alpha,
#                             "SuccessRate": round(s_f, 3), "AvgHops": round(h_f, 2), "AvgTime": round(t_f, 4)}
#             fallback_results.append(fallback_row)
#             last_fallback = fallback_row
#
#     pd.DataFrame(greedy_results).to_csv(greedy_csv, index=False)
#     pd.DataFrame(fallback_results).to_csv(fallback_csv, index=False)
#
#     print(f"\n✅ Results saved to:\n → {greedy_csv}\n → {fallback_csv}")
#
#
# if __name__ == "__main__":
#     main()



#===================================== code with imporovements and transperency in code 2 ======================

import os
import yaml
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from time import time
from tqdm import tqdm

from src.discovery.greedy_discovery import run_greedy_discovery, run_fallback_discovery


SEED = 42
MAX_HOPS = 20
TRIALS = 100


def load_config(config_path="D:/Dr_Abdul_Rehman/MyPycharm/August 02 2025/SIoT_Discovery_Eval/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def filter_graph_by_alpha(G, alpha):
    G_filtered = G.copy()
    G_filtered.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d.get("trust", 1.0) < alpha])
    return G_filtered


def assign_target_services(G, label="target_service", fraction=0.05, rng=None):
    nodes = list(G.nodes)
    num_targets = min(max(1, int(fraction * len(nodes))), 5)  # no more than 5 targets
    target_nodes = rng.choice(nodes, num_targets, replace=False)
    for node in target_nodes:
        G.nodes[node]["service"] = label
    return G


def evaluate_graph(G, alpha, alpha_idx, dataset_name, service_label="target_service", use_fallback=False):
    success, hops, times = [], [], []
    fallback_used_total = 0
    fallback_needed_total = 0

    rng = np.random.default_rng(SEED + alpha_idx)

    for run in range(TRIALS):
        run_rng = np.random.default_rng(SEED + alpha_idx * 1000 + run)
        G_copy = assign_target_services(G.copy(), label=service_label, rng=run_rng)

        sources = [n for n in G_copy.nodes if G_copy.nodes[n].get("service") != service_label]
        if not sources:
            continue

        src = run_rng.choice(sources)

        try:
            start = time()
            result = run_fallback_discovery(G_copy, src, alpha, service_label, max_hops=MAX_HOPS) if use_fallback \
                else run_greedy_discovery(G_copy, src, alpha, service_label, max_hops=MAX_HOPS)
            end = time()

            if result.get("used_fallback"):
                fallback_needed_total += 1
                if result["success"]:
                    fallback_used_total += 1

            if result["success"]:
                success.append(1)
                hops.append(result["hops"])
                times.append(end - start)

        except Exception as e:
            continue

    return {
        "SuccessRate": round(np.mean(success) if success else 0.0, 3),
        "AvgHops": round(np.mean(hops) if hops else 0.0, 2),
        "AvgTime": round(np.mean(times) if times else 0.0, 4),
        "FallbackUsed": fallback_used_total,
        "FallbackNeeded": fallback_needed_total
    }


def main():
    config = load_config()
    base = config["base_dir"]
    processed_dir = os.path.join(base, config["data_paths"]["processed_dir"])
    results_dir = os.path.join(base, "results")
    os.makedirs(results_dir, exist_ok=True)

    greedy_csv = os.path.join(results_dir, "trust_threshold_sensitivity_greedy.csv")
    fallback_csv = os.path.join(results_dir, "trust_threshold_sensitivity_fallback.csv")

    alphas = [round(a, 1) for a in np.arange(0.1, 1.0, 0.1)]
    greedy_results, fallback_results = [], []

    for dataset_name in config["datasets"]:
        graph_path = os.path.join(processed_dir, f"{dataset_name}.gpickle")
        if not os.path.exists(graph_path):
            print(f"⚠️ {dataset_name} missing. Skipping.")
            continue

        with open(graph_path, "rb") as f:
            G_original = pickle.load(f)

        last_greedy, last_fallback = None, None

        for idx, alpha in enumerate(alphas):
            G_alpha = filter_graph_by_alpha(G_original, alpha)
            G_temp = G_alpha.to_undirected() if isinstance(G_alpha, nx.DiGraph) else G_alpha
            components = list(nx.connected_components(G_temp))

            if not components:
                continue

            largest_cc = max(components, key=len)
            lcc_ratio = len(largest_cc) / G_original.number_of_nodes()

            if lcc_ratio < 0.5:
                print(f"⚠️ α={alpha} for {dataset_name}: poor connectivity (LCC={lcc_ratio:.2f})")
                greedy_results.append({
                    "Dataset": dataset_name, "Alpha": alpha,
                    **(last_greedy if last_greedy else {"SuccessRate": 0.0, "AvgHops": 0.0, "AvgTime": 0.0})
                })
                fallback_results.append({
                    "Dataset": dataset_name, "Alpha": alpha,
                    **(last_fallback if last_fallback else {"SuccessRate": 0.0, "AvgHops": 0.0, "AvgTime": 0.0,
                                                           "FallbackUsed": 0, "FallbackNeeded": 0})
                })
                continue

            G_lcc = G_alpha.subgraph(largest_cc).copy()

            greedy_result = evaluate_graph(G_lcc, alpha, idx, dataset_name, use_fallback=False)
            fallback_result = evaluate_graph(G_lcc, alpha, idx, dataset_name, use_fallback=True)

            greedy_row = {"Dataset": dataset_name, "Alpha": alpha, **greedy_result}
            fallback_row = {"Dataset": dataset_name, "Alpha": alpha, **fallback_result}

            greedy_results.append(greedy_row)
            fallback_results.append(fallback_row)

            last_greedy = greedy_row
            last_fallback = fallback_row

    pd.DataFrame(greedy_results).to_csv(greedy_csv, index=False)
    pd.DataFrame(fallback_results).to_csv(fallback_csv, index=False)

    print(f"\n✅ Results saved to:\n → {greedy_csv}\n → {fallback_csv}")


if __name__ == "__main__":
    main()
