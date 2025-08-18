# import networkx as nx
#
# def run_greedy_discovery(G, source, alpha=0.5, service_label="target_service"):
#     """
#     Greedy class-based SIoT service discovery.
#
#     Parameters:
#     - G (networkx.Graph or DiGraph): The input SIoT network.
#     - source (str): The starting node.
#     - alpha (float): Trust threshold (0.1 to 0.9).
#     - service_label (str): The service we are trying to discover.
#
#     Returns:
#     - dict: {
#         "success": True/False,
#         "hops": Number of hops taken,
#         "path": List of visited nodes
#     }
#     """
#     visited = set()
#     path = []
#     queue = [(source, 0)]  # (node, depth)
#
#     while queue:
#         current_node, depth = queue.pop(0)
#         visited.add(current_node)
#         path.append(current_node)
#
#         # Check if node offers the target service
#         if G.nodes[current_node].get("service") == service_label:
#             return {
#                 "success": True,
#                 "hops": depth,
#                 "path": path
#             }
#
#         # Explore neighbors sorted by descending trust
#         neighbors = sorted(
#             G.neighbors(current_node),
#             key=lambda x: G[current_node][x].get("trust", 0),
#             reverse=True
#         )
#
#         for neighbor in neighbors:
#             if neighbor not in visited:
#                 trust_val = G[current_node][neighbor].get("trust", 0)
#                 if trust_val >= alpha:
#                     queue.append((neighbor, depth + 1))
#
#     return {
#         "success": False,
#         "hops": 0,
#         "path": path
#     }



# ============================================= Modified code for our approach + fallback ===============

# import networkx as nx
#
# def run_greedy_discovery(G, source, alpha=0.5, service_label="target_service"):
#     visited = set()
#     path = []
#     queue = [(source, 0)]  # (node, depth)
#
#     while queue:
#         current_node, depth = queue.pop(0)
#         visited.add(current_node)
#         path.append(current_node)
#
#         if G.nodes[current_node].get("service") == service_label:
#             return {"success": True, "hops": depth, "path": path}
#
#         neighbors = sorted(
#             G.neighbors(current_node),
#             key=lambda x: G[current_node][x].get("trust", 0),
#             reverse=True
#         )
#
#         for neighbor in neighbors:
#             if neighbor not in visited:
#                 trust_val = G[current_node][neighbor].get("trust", 0)
#                 if trust_val >= alpha:
#                     queue.append((neighbor, depth + 1))
#
#     return {"success": False, "hops": 0, "path": path}
#
#
# def run_fallback_discovery(G, source, alpha=0.5, service_label="target_service"):
#     """
#     Greedy service discovery with fallback (no alpha filter on fallback).
#     First explores neighbors with trust ≥ alpha, then allows all trust edges if stuck.
#
#     Returns:
#     - dict: {
#         "success": True/False,
#         "hops": Number of hops taken,
#         "path": List of visited nodes
#     }
#     """
#     visited = set()
#     path = []
#     queue = [(source, 0)]
#     fallback_used = False
#
#     while queue:
#         current_node, depth = queue.pop(0)
#         visited.add(current_node)
#         path.append(current_node)
#
#         if G.nodes[current_node].get("service") == service_label:
#             return {"success": True, "hops": depth, "path": path}
#
#         neighbors = sorted(G.neighbors(current_node),
#                            key=lambda x: G[current_node][x].get("trust", 0),
#                            reverse=True)
#
#         added = False
#         for neighbor in neighbors:
#             if neighbor not in visited:
#                 trust_val = G[current_node][neighbor].get("trust", 0)
#                 if trust_val >= alpha:
#                     queue.append((neighbor, depth + 1))
#                     added = True
#
#         # Fallback if no neighbors meet alpha
#         if not added and not fallback_used:
#             fallback_used = True
#             for neighbor in neighbors:
#                 if neighbor not in visited:
#                     queue.append((neighbor, depth + 1))
#
#     return {"success": False, "hops": 0, "path": path}
#
#
# def run_discovery(G, source, alpha=0.5, service_label="target_service", strategy="greedy", top_k=3):
#     if strategy == "greedy":
#         return run_greedy_discovery(G, source, alpha, service_label)
#     elif strategy == "top_k":
#         return run_fallback_discovery(G, source, alpha, service_label, top_k)
#     else:
#         raise ValueError(f"Unsupported strategy: {strategy}")



#============================= Code after improvements in the code 2 with transperency =====================


import networkx as nx

def run_greedy_discovery(G, source, alpha=0.5, service_label="target_service", max_hops=20):
    visited = set()
    path = []
    queue = [(source, 0)]  # (node, depth)

    while queue:
        current_node, depth = queue.pop(0)

        if depth > max_hops:
            break

        visited.add(current_node)
        path.append(current_node)

        if G.nodes[current_node].get("service") == service_label:
            return {
                "success": True,
                "hops": depth,
                "path": path,
                "used_fallback": False
            }

        neighbors = sorted(
            G.neighbors(current_node),
            key=lambda x: G[current_node][x].get("trust", 0),
            reverse=True
        )

        for neighbor in neighbors:
            if neighbor not in visited:
                trust_val = G[current_node][neighbor].get("trust", 0)
                if trust_val >= alpha:
                    queue.append((neighbor, depth + 1))

    return {
        "success": False,
        "hops": 0,
        "path": path,
        "used_fallback": False
    }


def run_fallback_discovery(G, source, alpha=0.5, service_label="target_service", max_hops=20):
    """
    Greedy service discovery with fallback: first tries trust ≥ alpha,
    then falls back to any trust value if stuck.
    """
    visited = set()
    path = []
    queue = [(source, 0)]
    fallback_used = False

    while queue:
        current_node, depth = queue.pop(0)

        if depth > max_hops:
            break

        visited.add(current_node)
        path.append(current_node)

        if G.nodes[current_node].get("service") == service_label:
            return {
                "success": True,
                "hops": depth,
                "path": path,
                "used_fallback": fallback_used
            }

        neighbors = sorted(
            G.neighbors(current_node),
            key=lambda x: G[current_node][x].get("trust", 0),
            reverse=True
        )

        added = False
        for neighbor in neighbors:
            if neighbor not in visited:
                trust_val = G[current_node][neighbor].get("trust", 0)
                if trust_val >= alpha:
                    queue.append((neighbor, depth + 1))
                    added = True

        # Fallback: allow any edge if no neighbor meets trust threshold
        if not added and not fallback_used:
            fallback_used = True
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

    return {
        "success": False,
        "hops": 0,
        "path": path,
        "used_fallback": fallback_used
    }


def run_discovery(G, source, alpha=0.5, service_label="target_service", strategy="greedy", max_hops=20):
    if strategy == "greedy":
        return run_greedy_discovery(G, source, alpha, service_label, max_hops)
    elif strategy == "top_k":
        return run_fallback_discovery(G, source, alpha, service_label, max_hops)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
