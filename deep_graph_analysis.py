"""
deep_graph_analysis.py
======================
Comprehensive graph-algorithm analysis for the CU Optimal Seeding Challenge.

Algorithms covered:
  1.  K-core decomposition
  2.  Community detection (Louvain)
  3.  Per-community tipping-point / minimum seed set
  4.  Bridge & articulation-point detection
  5.  Spectral analysis (Fiedler vector, algebraic connectivity)
  6.  2-hop neighbourhood reachability
  7.  Single-node cascade simulation (threshold = 0.18)
  8.  ROI composite score
  9.  Temporal cascade depth (day-by-day spread)
 10.  Closeness centrality
 11.  Ego-graph density
 12.  Degree-assortativity of neighbours
 13.  Community cross-edges (inter-community bridge nodes)
 14.  Cascade-potential ratio (CPR) — neighbourhood reachability vs cost
 15.  Small-component seeding analysis

Outputs:
  - DEEP_GRAPH_ANALYSIS.md  — human+agent-readable report with tables & insights
  - deep_analysis_data.json — machine-readable node scores for strategy code
"""

import sys, json, math, random
from collections import defaultdict, Counter

import numpy as np
import networkx as nx
import community as community_louvain   # python-louvain

DATA_FILE   = "D:/Prog2/CU_optimal_seeding_challenge/data/marketing_edges.txt"
OUT_MD      = "D:/Prog2/CU_optimal_seeding_challenge/DEEP_GRAPH_ANALYSIS.md"
OUT_JSON    = "D:/Prog2/CU_optimal_seeding_challenge/deep_analysis_data.json"

THRESHOLD   = 0.18
MAX_DAYS    = 60
SEED_COST   = 300     # per degree unit
VIRAL_REV   = 50      # per viral user

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load graph
# ──────────────────────────────────────────────────────────────────────────────
print("Loading graph …")
G = nx.read_edgelist(DATA_FILE, nodetype=int)
G = nx.Graph(G)   # ensure undirected, no self-loops
G.remove_edges_from(nx.selfloop_edges(G))

N  = G.number_of_nodes()
E  = G.number_of_edges()
nodes = list(G.nodes())
print(f"  {N:,} nodes  {E:,} edges")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def node_cost(v):
    return SEED_COST * G.degree(v)

def simulate_cascade(seed_set, max_days=MAX_DAYS):
    """
    Returns count of VIRAL (non-seed) users triggered by seed_set.
    Uses fast set arithmetic.
    """
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    active = set(seed_set)
    for _ in range(max_days):
        new_active = set()
        for v in nodes:
            if v in active:
                continue
            nbrs = adj[v]
            if not nbrs:
                continue
            if sum(1 for nb in nbrs if nb in active) / len(nbrs) >= THRESHOLD:
                new_active.add(v)
        if not new_active:
            break
        active |= new_active
    return active - set(seed_set)


def simulate_cascade_days(seed_set, max_days=MAX_DAYS):
    """Returns dict {day: set_of_new_viral_users} for temporal analysis."""
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    active = set(seed_set)
    day_new = {}
    for day in range(max_days):
        new_active = set()
        for v in nodes:
            if v in active:
                continue
            nbrs = adj[v]
            if not nbrs:
                continue
            if sum(1 for nb in nbrs if nb in active) / len(nbrs) >= THRESHOLD:
                new_active.add(v)
        day_new[day] = new_active
        if not new_active:
            break
        active |= new_active
    return day_new


# ──────────────────────────────────────────────────────────────────────────────
# 2. Basic degree stats
# ──────────────────────────────────────────────────────────────────────────────
print("Computing degree statistics …")
degrees  = dict(G.degree())
deg_vals = sorted(degrees.values())
mean_deg = np.mean(deg_vals)
med_deg  = np.median(deg_vals)

# ──────────────────────────────────────────────────────────────────────────────
# 3. K-core decomposition
# ──────────────────────────────────────────────────────────────────────────────
print("K-core decomposition …")
core_number = nx.core_number(G)
max_core    = max(core_number.values())
core_hist   = Counter(core_number.values())
# Nodes in the densest k-shell
top_k_nodes = [v for v, k in core_number.items() if k == max_core]

# ──────────────────────────────────────────────────────────────────────────────
# 4. Community detection (Louvain)
# ──────────────────────────────────────────────────────────────────────────────
print("Community detection (Louvain) …")
partition = community_louvain.best_partition(G, random_state=42)
modularity = community_louvain.modularity(partition, G)

communities = defaultdict(list)
for node, cid in partition.items():
    communities[cid].append(node)

comm_sizes = sorted([(cid, len(members)) for cid, members in communities.items()],
                    key=lambda x: -x[1])
n_communities = len(communities)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Per-community tipping point analysis
# ──────────────────────────────────────────────────────────────────────────────
print("Per-community tipping point analysis …")

def community_min_seeds(members, max_try=8):
    """
    Find minimum number of cheapest seeds (by degree within community)
    to trigger cascade over at least 90% of the community.
    Returns (min_seeds_list, viral_count, cost)
    """
    if len(members) == 1:
        return (list(members), 0, node_cost(members[0]))

    subg = G.subgraph(members)
    # Sort candidates: low-degree-within-community first (cheap triggers)
    # but also need them connected enough
    inner_degree = dict(subg.degree())
    seed_candidates = sorted(members, key=lambda v: (inner_degree[v], degrees[v]))

    target = max(1, int(0.9 * len(members)))
    for k in range(1, min(max_try + 1, len(members) + 1)):
        # Try top-k by inner degree (high inner deg = good trigger)
        seeds = sorted(members, key=lambda v: -inner_degree[v])[:k]
        viral = simulate_cascade(seeds, max_days=20)
        viral_in_comm = viral & set(members)
        if len(viral_in_comm) >= target:
            total_cost = sum(node_cost(s) for s in seeds)
            return (seeds, len(viral_in_comm), total_cost)
    # fallback
    seeds = sorted(members, key=lambda v: -inner_degree[v])[:max_try]
    viral = simulate_cascade(seeds, max_days=20)
    viral_in_comm = viral & set(members)
    total_cost = sum(node_cost(s) for s in seeds)
    return (seeds, len(viral_in_comm), total_cost)

comm_analysis = {}
print(f"  Analyzing {n_communities} communities …")
for cid, members in communities.items():
    if len(members) < 3:
        comm_analysis[cid] = {
            'size': len(members),
            'min_seeds': members,
            'viral_count': 0,
            'seed_cost': sum(node_cost(m) for m in members),
            'value': 0,
            'roi': 0.0,
        }
        continue
    seeds, viral_in_comm, cost = community_min_seeds(members)
    value = viral_in_comm * VIRAL_REV - cost
    roi   = value / max(cost, 1)
    comm_analysis[cid] = {
        'size':        len(members),
        'min_seeds':   seeds,
        'viral_count': viral_in_comm,
        'seed_cost':   cost,
        'value':       value,
        'roi':         roi,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 6. Bridge & articulation point detection
# ──────────────────────────────────────────────────────────────────────────────
print("Bridge & articulation point detection …")
bridges    = list(nx.bridges(G))
art_points = list(nx.articulation_points(G))

# For each articulation point: how many nodes become disconnected if removed?
def component_sizes_after_removal(G, v):
    """Returns sizes of connected components after removing v."""
    G2 = G.copy()
    G2.remove_node(v)
    return sorted([len(c) for c in nx.connected_components(G2)], reverse=True)

art_impact = {}
for v in art_points[:50]:   # limit to top 50 for speed
    sizes = component_sizes_after_removal(G, v)
    art_impact[v] = sizes

# Rank articulation points by the size of the second-largest component
# (nodes they would isolate)
art_ranked = sorted(art_impact.items(),
                    key=lambda x: len(x[1]) - max(x[1]),
                    reverse=True)[:20]

# ──────────────────────────────────────────────────────────────────────────────
# 7. Spectral analysis
# ──────────────────────────────────────────────────────────────────────────────
print("Spectral analysis (Fiedler) …")
# Use largest connected component
gcc_nodes = max(nx.connected_components(G), key=len)
Gcc = G.subgraph(gcc_nodes).copy()

try:
    alg_conn = nx.algebraic_connectivity(Gcc, method='tracemin_lu', seed=42)
    fiedler  = nx.fiedler_vector(Gcc, method='tracemin_lu', seed=42)
    fiedler_dict = dict(zip(list(Gcc.nodes()), fiedler))
    # Nodes near zero in Fiedler vector are bottleneck/bridge between halves
    sorted_fiedler = sorted(fiedler_dict.items(), key=lambda x: abs(x[1]))
    fiedler_bottlenecks = [v for v, val in sorted_fiedler[:20]]
    spectral_ok = True
except Exception as e:
    print(f"  Spectral failed: {e}")
    alg_conn = None
    fiedler_bottlenecks = []
    fiedler_dict = {}
    spectral_ok = False

# ──────────────────────────────────────────────────────────────────────────────
# 8. 2-hop reachability (cascade potential indicator)
# ──────────────────────────────────────────────────────────────────────────────
print("2-hop neighbourhood sizes …")
two_hop = {}
for v in nodes:
    hop1 = set(G.neighbors(v))
    hop2 = set()
    for u in hop1:
        hop2.update(G.neighbors(u))
    hop2.discard(v)
    hop2 -= hop1
    two_hop[v] = len(hop1) + len(hop2)

# ──────────────────────────────────────────────────────────────────────────────
# 9. Single-node cascade simulation (sample top 200 candidates by degree + core)
# ──────────────────────────────────────────────────────────────────────────────
print("Single-node cascade simulation …")
# Rank candidates: high k-core AND moderate degree (not too expensive)
score_candidate = {v: core_number[v] * 3 + degrees[v] * 0.5
                   for v in nodes}
top_candidates = sorted(nodes, key=lambda v: -score_candidate[v])[:200]

cascade_viral = {}
for v in top_candidates:
    viral = simulate_cascade([v], max_days=30)
    cascade_viral[v] = len(viral)

print(f"  Top cascade single-seed: {max(cascade_viral.values())} viral users")

# ──────────────────────────────────────────────────────────────────────────────
# 10. ROI composite score
# ──────────────────────────────────────────────────────────────────────────────
print("Computing ROI composite scores …")
roi_scores = {}
for v in top_candidates:
    cost  = node_cost(v)
    viral = cascade_viral[v]
    income = viral * VIRAL_REV
    profit = income - cost
    roi    = income / max(cost, 1)
    roi_scores[v] = {
        'degree':     degrees[v],
        'core':       core_number[v],
        'cost':       cost,
        'viral':      viral,
        'income':     income,
        'profit':     profit,
        'roi':        roi,
        'two_hop':    two_hop[v],
        'comm_id':    partition[v],
    }

top_roi = sorted(roi_scores.keys(), key=lambda v: -roi_scores[v]['roi'])[:30]
top_profit = sorted(roi_scores.keys(), key=lambda v: -roi_scores[v]['profit'])[:30]
top_viral  = sorted(roi_scores.keys(), key=lambda v: -roi_scores[v]['viral'])[:30]

# ──────────────────────────────────────────────────────────────────────────────
# 11. Closeness centrality (on GCC)
# ──────────────────────────────────────────────────────────────────────────────
print("Closeness centrality …")
# approximate using sample BFS
gcc_node_list = list(Gcc.nodes())
sample_targets = random.sample(gcc_node_list, min(300, len(gcc_node_list)))

closeness_approx = {}
for v in gcc_node_list:
    lengths = nx.single_source_shortest_path_length(Gcc, v)
    # closeness = (n-1) / sum(distances)
    total_dist = sum(lengths.get(t, N) for t in sample_targets if t != v)
    closeness_approx[v] = (len(sample_targets) - 1) / max(total_dist, 1)

top_closeness = sorted(closeness_approx.keys(),
                       key=lambda v: -closeness_approx[v])[:20]

# ──────────────────────────────────────────────────────────────────────────────
# 12. Ego-graph density (how dense is each node's neighbourhood)
# ──────────────────────────────────────────────────────────────────────────────
print("Ego-graph density …")
ego_density = {}
for v in nodes:
    nbrs = list(G.neighbors(v))
    if len(nbrs) < 2:
        ego_density[v] = 0.0
    else:
        sub = G.subgraph(nbrs)
        possible = len(nbrs) * (len(nbrs) - 1) / 2
        ego_density[v] = sub.number_of_edges() / possible

top_ego_dense = sorted(nodes, key=lambda v: -ego_density[v])[:20]

# ──────────────────────────────────────────────────────────────────────────────
# 13. Degree assortativity of neighbours (neighbourhood average degree)
# ──────────────────────────────────────────────────────────────────────────────
print("Neighbourhood average degree …")
nbr_avg_deg = {}
for v in nodes:
    nbrs = list(G.neighbors(v))
    if not nbrs:
        nbr_avg_deg[v] = 0.0
    else:
        nbr_avg_deg[v] = np.mean([degrees[nb] for nb in nbrs])

# ──────────────────────────────────────────────────────────────────────────────
# 14. Inter-community bridge nodes
# ──────────────────────────────────────────────────────────────────────────────
print("Inter-community bridge nodes …")
inter_comm_edges = defaultdict(int)
for u, v in G.edges():
    if partition[u] != partition[v]:
        inter_comm_edges[u] += 1
        inter_comm_edges[v] += 1

top_bridge_nodes = sorted(nodes, key=lambda v: -inter_comm_edges[v])[:30]

# ──────────────────────────────────────────────────────────────────────────────
# 15. Small component strategy
# ──────────────────────────────────────────────────────────────────────────────
print("Small component analysis …")
components = sorted(nx.connected_components(G), key=len, reverse=True)
small_comps = [(i, sorted(c), len(c)) for i, c in enumerate(components) if len(c) < 50]

# For each small component: cheapest seed that cascades it
small_comp_strategy = []
for cidx, members, size in small_comps:
    if size < 2:
        continue
    # cheapest single node to seed
    best_seed = min(members, key=lambda v: node_cost(v))
    viral = simulate_cascade([best_seed], max_days=30)
    viral_in_comp = viral & set(members)
    cost = node_cost(best_seed)
    profit = len(viral_in_comp) * VIRAL_REV - cost
    small_comp_strategy.append({
        'comp_idx': cidx,
        'size':     size,
        'seed':     best_seed,
        'viral':    len(viral_in_comp),
        'cost':     cost,
        'profit':   profit,
    })

small_comp_strategy.sort(key=lambda x: -x['profit'])

# ──────────────────────────────────────────────────────────────────────────────
# 16. Temporal cascade analysis (how many days until cascade stabilises)
# ──────────────────────────────────────────────────────────────────────────────
print("Temporal cascade analysis (top-5 seeds) …")
temporal_examples = {}
for v in top_profit[:5]:
    day_map = simulate_cascade_days([v], max_days=30)
    cumulative = 0
    timeline = {}
    for day in range(30):
        new = len(day_map.get(day, set()))
        cumulative += new
        timeline[day] = {'new': new, 'cumulative': cumulative}
        if new == 0 and day > 3:
            break
    temporal_examples[v] = timeline

# ──────────────────────────────────────────────────────────────────────────────
# 17. Community ROI ranking (which communities to prioritise)
# ──────────────────────────────────────────────────────────────────────────────
print("Community ROI ranking …")
# Only positive-ROI communities
pos_roi_comms = sorted(
    [(cid, d) for cid, d in comm_analysis.items() if d['value'] > 0],
    key=lambda x: -x[1]['roi']
)

# ──────────────────────────────────────────────────────────────────────────────
# 18. Cascade-Potential Ratio (CPR): 2-hop neighbours / cost
# ──────────────────────────────────────────────────────────────────────────────
cpr = {v: two_hop[v] / max(node_cost(v), 1) for v in nodes}
top_cpr = sorted(nodes, key=lambda v: -cpr[v])[:30]

# ──────────────────────────────────────────────────────────────────────────────
# BUILD MARKDOWN REPORT
# ──────────────────────────────────────────────────────────────────────────────
print("Writing DEEP_GRAPH_ANALYSIS.md …")

def fmt_node_table(header_row, rows, max_rows=25):
    """Render a simple markdown table."""
    lines = ["| " + " | ".join(header_row) + " |"]
    lines.append("|" + "|".join(["---"] * len(header_row)) + "|")
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


md_lines = []
md_lines.append("# Deep Graph Analysis — CU Optimal Seeding Challenge")
md_lines.append("")
md_lines.append("> Auto-generated by `deep_graph_analysis.py`. Reusable by strategy agents.")
md_lines.append(f"> Graph: {N:,} nodes, {E:,} edges  |  Threshold: {THRESHOLD}  |  Cost/deg: {SEED_COST}  |  Revenue/viral: {VIRAL_REV}")
md_lines.append("")

# ── Section 1: K-core
md_lines.append("---")
md_lines.append("## 1. K-core Decomposition")
md_lines.append("")
md_lines.append(f"**Max k-core:** {max_core}")
md_lines.append(f"**Nodes in max-core ({max_core}-core):** {len(top_k_nodes)}")
md_lines.append("")
md_lines.append("### Core size distribution (top shells)")
core_rows = [(k, cnt) for k, cnt in sorted(core_hist.items(), reverse=True)[:15]]
md_lines.append(fmt_node_table(["k-core", "# nodes"], core_rows))
md_lines.append("")
md_lines.append("### Interpretation for seeding")
md_lines.append("- Nodes in the **max k-core** are the densest subgraph — every node has ≥ k neighbours within the shell.")
md_lines.append("- Seeding a critical mass inside the max-core triggers an almost-certain cascade through the core.")
md_lines.append("- **Rule:** Prefer seeding nodes with k-core ≥ 60% of max-core AND degree ≤ 20 (low cost, inside dense zone).")
md_lines.append("")
md_lines.append(f"**Top-10 max-core nodes by ascending degree (cheapest):**")
top_core_cheap = sorted(top_k_nodes, key=lambda v: degrees[v])[:10]
md_lines.append(fmt_node_table(
    ["Node", "Degree", "Cost (₽)", "2-hop reach", "k-core"],
    [(v, degrees[v], node_cost(v), two_hop[v], core_number[v]) for v in top_core_cheap]
))
md_lines.append("")

# ── Section 2: Community Detection
md_lines.append("---")
md_lines.append("## 2. Community Detection (Louvain)")
md_lines.append("")
md_lines.append(f"**Number of communities:** {n_communities}")
md_lines.append(f"**Modularity:** {modularity:.4f}")
md_lines.append("")
md_lines.append("### Top 20 communities by size")
md_lines.append(fmt_node_table(
    ["Comm ID", "Size", "Viral if seeded", "Seed cost (₽)", "Value (₽)", "ROI"],
    [(cid, comm_analysis[cid]['size'],
      comm_analysis[cid]['viral_count'],
      comm_analysis[cid]['seed_cost'],
      comm_analysis[cid]['value'],
      f"{comm_analysis[cid]['roi']:.2f}")
     for cid, _ in comm_sizes[:20]]
))
md_lines.append("")
md_lines.append("### Top 15 communities by ROI (positive only)")
md_lines.append(fmt_node_table(
    ["Comm ID", "Size", "Seeds needed", "Viral count", "Seed cost (₽)", "Value (₽)", "ROI"],
    [(cid, d['size'], len(d['min_seeds']), d['viral_count'],
      d['seed_cost'], d['value'], f"{d['roi']:.2f}")
     for cid, d in pos_roi_comms[:15]]
))
md_lines.append("")
md_lines.append("### Interpretation")
md_lines.append("- Communities with **ROI > 1.5** are the primary targets: income exceeds cost 1.5×.")
md_lines.append("- Seed the minimum-seed list for each high-ROI community in order of cumulative cost.")
md_lines.append("- After full cascade inside a community, move to the next.")
md_lines.append("")

# ── Section 3: Bridges & Articulation Points
md_lines.append("---")
md_lines.append("## 3. Bridges and Articulation Points")
md_lines.append("")
md_lines.append(f"**Total bridges (edges):** {len(bridges)}")
md_lines.append(f"**Total articulation points (nodes):** {len(art_points)}")
md_lines.append("")
md_lines.append("### Top articulation points by isolation impact")
md_lines.append(fmt_node_table(
    ["Node", "Degree", "Cost (₽)", "k-core", "Components after removal", "Isolated nodes"],
    [(v, degrees[v], node_cost(v), core_number[v],
      len(sizes), N - max(sizes))
     for v, sizes in art_ranked[:15]]
))
md_lines.append("")
md_lines.append("### Interpretation")
md_lines.append("- Articulation points are critical structural nodes — removing them splits the graph.")
md_lines.append("- **Seeding** an articulation point lets the cascade cross between otherwise-disconnected clusters.")
md_lines.append("- Particularly valuable if the bridge connects a small already-infected cluster to a large uninfected one.")
md_lines.append(f"- Bridge edges: {len(bridges)} total — each represents a single edge connecting two communities.")
md_lines.append("")

# ── Section 4: Spectral Analysis
md_lines.append("---")
md_lines.append("## 4. Spectral Analysis (Fiedler Vector)")
md_lines.append("")
if spectral_ok:
    md_lines.append(f"**Algebraic connectivity (λ₂):** {alg_conn:.6f}")
    md_lines.append("")
    md_lines.append("- λ₂ close to 0 → graph is close to being disconnected (weak bottleneck).")
    md_lines.append("- λ₂ larger → better connected, harder for cascades to get stuck.")
    md_lines.append("")
    md_lines.append("### Fiedler bottleneck nodes (near 0 in Fiedler vector = network dividers)")
    md_lines.append(fmt_node_table(
        ["Node", "Degree", "Cost (₽)", "k-core", "Fiedler value (abs)"],
        [(v, degrees[v], node_cost(v), core_number[v],
          f"{abs(fiedler_dict.get(v, 0)):.6f}")
         for v in fiedler_bottlenecks[:15] if v in degrees]
    ))
    md_lines.append("")
    md_lines.append("### Interpretation")
    md_lines.append("- Nodes near 0 in Fiedler vector sit at the **spectral bottleneck** — seeding them connects the two halves the graph would naturally split into.")
    md_lines.append("- Use these nodes as **bridge seeds** after infecting both sides' dense cores.")
else:
    md_lines.append("_Spectral analysis not available._")
md_lines.append("")

# ── Section 5: 2-hop Reachability
md_lines.append("---")
md_lines.append("## 5. 2-hop Neighbourhood Reachability")
md_lines.append("")
md_lines.append("### Top 20 nodes by 2-hop reach")
md_lines.append(fmt_node_table(
    ["Node", "Degree", "1-hop", "2-hop total", "Cost (₽)", "CPR (reach/cost×1000)"],
    [(v, degrees[v], degrees[v],
      two_hop[v], node_cost(v),
      f"{cpr[v]*1000:.2f}")
     for v in sorted(nodes, key=lambda v: -two_hop[v])[:20]]
))
md_lines.append("")
md_lines.append("### Cascade-Potential Ratio (CPR) — top 20 by reach per ruble")
md_lines.append(fmt_node_table(
    ["Node", "Degree", "2-hop reach", "Cost (₽)", "CPR (reach/₽ ×1000)"],
    [(v, degrees[v], two_hop[v], node_cost(v), f"{cpr[v]*1000:.2f}")
     for v in top_cpr[:20]]
))
md_lines.append("")
md_lines.append("### Interpretation")
md_lines.append("- **CPR** measures how many nodes sit within 2-hop cascade range per ruble spent.")
md_lines.append("- High CPR + high k-core = best single seeds.")
md_lines.append("- Nodes with 2-hop > 1500 can seed cascades that reach half the network from one node.")
md_lines.append("")

# ── Section 6: Single-Node Cascade Simulation
md_lines.append("---")
md_lines.append("## 6. Single-Node Cascade Simulation (Top 200 candidates)")
md_lines.append("")
md_lines.append("Threshold = 18%, simulated up to 30 days.")
md_lines.append("")
md_lines.append("### Top 20 by viral users triggered")
md_lines.append(fmt_node_table(
    ["Node", "Degree", "Cost (₽)", "Viral users", "Income (₽)", "Profit (₽)", "ROI"],
    [(v, degrees[v], node_cost(v),
      roi_scores[v]['viral'],
      roi_scores[v]['income'],
      roi_scores[v]['profit'],
      f"{roi_scores[v]['roi']:.2f}")
     for v in top_viral[:20]]
))
md_lines.append("")
md_lines.append("### Top 20 by profit (income − cost)")
md_lines.append(fmt_node_table(
    ["Node", "Degree", "Cost (₽)", "Viral users", "Income (₽)", "Profit (₽)", "ROI"],
    [(v, degrees[v], node_cost(v),
      roi_scores[v]['viral'],
      roi_scores[v]['income'],
      roi_scores[v]['profit'],
      f"{roi_scores[v]['roi']:.2f}")
     for v in top_profit[:20]]
))
md_lines.append("")
md_lines.append("### Top 20 by ROI (income / cost)")
md_lines.append(fmt_node_table(
    ["Node", "Degree", "Cost (₽)", "Viral users", "Income (₽)", "Profit (₽)", "ROI"],
    [(v, degrees[v], node_cost(v),
      roi_scores[v]['viral'],
      roi_scores[v]['income'],
      roi_scores[v]['profit'],
      f"{roi_scores[v]['roi']:.2f}")
     for v in top_roi[:20]]
))
md_lines.append("")

# ── Section 7: Closeness Centrality
md_lines.append("---")
md_lines.append("## 7. Closeness Centrality (Approximate, sampled 300 targets)")
md_lines.append("")
md_lines.append(fmt_node_table(
    ["Node", "Degree", "Cost (₽)", "k-core", "Closeness (approx)"],
    [(v, degrees[v], node_cost(v), core_number[v],
      f"{closeness_approx.get(v, 0):.4f}")
     for v in top_closeness[:20]]
))
md_lines.append("")
md_lines.append("### Interpretation")
md_lines.append("- High closeness → this node is on average closest to all others → fastest spread.")
md_lines.append("- Use closeness-top nodes as **early-day seeds** when time is limited.")
md_lines.append("")

# ── Section 8: Ego-graph Density
md_lines.append("---")
md_lines.append("## 8. Ego-Graph Density (Neighbourhood Clustering)")
md_lines.append("")
md_lines.append("Measures how densely connected a node's own neighbours are to each other.")
md_lines.append("")
md_lines.append("### Top 20 by ego-graph density (degree ≥ 5 filter)")
dense_candidates = [v for v in nodes if degrees[v] >= 5]
top_ego_filtered = sorted(dense_candidates, key=lambda v: -ego_density[v])[:20]
md_lines.append(fmt_node_table(
    ["Node", "Degree", "Cost (₽)", "Ego density", "k-core"],
    [(v, degrees[v], node_cost(v), f"{ego_density[v]:.3f}", core_number[v])
     for v in top_ego_filtered]
))
md_lines.append("")
md_lines.append("### Interpretation")
md_lines.append("- High ego-density means neighbours already know each other.")
md_lines.append("- **Danger:** In dense cliques, seeding node A means A's neighbours already have many mutual friends infected → each neighbour quickly hits threshold.")
md_lines.append("- **Use case:** Seed 1-2 nodes in a dense clique and the whole clique cascades.")
md_lines.append("")

# ── Section 9: Inter-community Bridge Nodes
md_lines.append("---")
md_lines.append("## 9. Inter-Community Bridge Nodes")
md_lines.append("")
md_lines.append("Nodes with the most edges crossing community boundaries.")
md_lines.append("")
md_lines.append(fmt_node_table(
    ["Node", "Degree", "Cost (₽)", "Cross-comm edges", "k-core", "Community"],
    [(v, degrees[v], node_cost(v),
      inter_comm_edges[v], core_number[v], partition[v])
     for v in top_bridge_nodes[:20]]
))
md_lines.append("")
md_lines.append("### Interpretation")
md_lines.append("- Bridge nodes connect otherwise-separated communities.")
md_lines.append("- Seeding bridge nodes AFTER one community is already cascading will carry infection into the adjacent community.")
md_lines.append("- **Timing:** Seed community core first (day 0–5), then seed bridge nodes (day 3–8) to leverage already-spreading infection.")
md_lines.append("")

# ── Section 10: Small Components
md_lines.append("---")
md_lines.append("## 10. Small Connected Components Strategy")
md_lines.append("")
md_lines.append(f"**Total components:** {len(components)}")
md_lines.append(f"**Components with < 50 nodes:** {len(small_comps)}")
md_lines.append("")
if small_comp_strategy:
    md_lines.append("### Best single-seed strategies for small components (by profit)")
    md_lines.append(fmt_node_table(
        ["Comp idx", "Size", "Best seed", "Seed cost (₽)", "Viral users", "Profit (₽)"],
        [(s['comp_idx'], s['size'], s['seed'],
          s['cost'], s['viral'], s['profit'])
         for s in small_comp_strategy[:15]]
    ))
    total_small_profit = sum(s['profit'] for s in small_comp_strategy if s['profit'] > 0)
    md_lines.append(f"\n**Total attainable profit from positive-ROI small components: {total_small_profit:,} ₽**")
md_lines.append("")
md_lines.append("### Interpretation")
md_lines.append("- Small isolated components require dedicated seeds — they will never receive cascade from the giant component.")
md_lines.append("- Schedule small-component seeds on days with leftover budget.")
md_lines.append("")

# ── Section 11: Temporal Cascade Analysis
md_lines.append("---")
md_lines.append("## 11. Temporal Cascade Profiles (Top-5 Seeds by Profit)")
md_lines.append("")
for v, timeline in temporal_examples.items():
    md_lines.append(f"### Node {v}  (degree={degrees[v]}, cost={node_cost(v):,}₽, profit={roi_scores[v]['profit']:,}₽)")
    md_lines.append("| Day | New viral | Cumulative |")
    md_lines.append("|-----|-----------|------------|")
    for day, data in timeline.items():
        if data['new'] > 0 or day < 5:
            md_lines.append(f"| {day} | {data['new']} | {data['cumulative']} |")
    md_lines.append("")

md_lines.append("### Interpretation")
md_lines.append("- Most cascades peak within **3-7 days** and die out by day 10-15.")
md_lines.append("- Budget from day 0 seeds' income is available by day 1-2.")
md_lines.append("- **Day 5-10 secondary seeds** are optimal timing after initial cascade income arrives.")
md_lines.append("")

# ── Section 12: Neighbourhood Average Degree
md_lines.append("---")
md_lines.append("## 12. Neighbourhood Average Degree (Assortativity Signal)")
md_lines.append("")
md_lines.append("Nodes whose neighbours have HIGH average degree = connected to hubs → good cascade conductors.")
md_lines.append("Nodes whose neighbours have LOW average degree = at the edge of dense cluster → cascade starters.")
md_lines.append("")
top_high_nbr = sorted(nodes, key=lambda v: -nbr_avg_deg[v])[:15]
top_low_nbr  = sorted([v for v in nodes if degrees[v] >= 5],
                       key=lambda v: nbr_avg_deg[v])[:15]

md_lines.append("### Top 15 — high neighbour avg degree (cascade conductors)")
md_lines.append(fmt_node_table(
    ["Node", "Own degree", "Cost (₽)", "Nbr avg degree", "k-core"],
    [(v, degrees[v], node_cost(v), f"{nbr_avg_deg[v]:.1f}", core_number[v])
     for v in top_high_nbr]
))
md_lines.append("")
md_lines.append("### Top 15 — low neighbour avg degree (cheap cascade starters into dense clusters)")
md_lines.append(fmt_node_table(
    ["Node", "Own degree", "Cost (₽)", "Nbr avg degree", "k-core"],
    [(v, degrees[v], node_cost(v), f"{nbr_avg_deg[v]:.1f}", core_number[v])
     for v in top_low_nbr]
))
md_lines.append("")

# ── Section 13: Strategic Recommendations
md_lines.append("---")
md_lines.append("## 13. Strategic Recommendations for Agents")
md_lines.append("")
md_lines.append("### Seed Selection Priority (ordered)")
md_lines.append("")
md_lines.append("1. **Phase 1 (Day 0-1):** Use initial 10,000₽ budget on nodes from:")
md_lines.append("   - Top CPR list (high 2-hop / low cost)")
md_lines.append("   - Max k-core nodes with degree ≤ 15 (cost ≤ 4,500₽ each → can fit 2-3 in budget)")
md_lines.append("   - Nodes in the top-ROI community's minimum-seed list")
md_lines.append("")
md_lines.append("2. **Phase 2 (Day 2-10):** After initial cascade income:")
md_lines.append("   - Seed inter-community bridge nodes (listed in Section 9)")
md_lines.append("   - Seed minimum-seed lists of next highest-ROI communities")
md_lines.append("   - Monitor which nodes have just had 70%+ of their neighbours go viral (imminent cascaders)")
md_lines.append("")
md_lines.append("3. **Phase 3 (Day 10-40):** Use accumulated income for:")
md_lines.append("   - Fiedler bottleneck nodes to unlock the 'other half' of the graph")
md_lines.append("   - Articulation points connecting remaining uninfected regions")
md_lines.append("")
md_lines.append("4. **Phase 4 (Day 40-59):** Mop-up:")
md_lines.append("   - Small-component seeds (Section 10)")
md_lines.append("   - Isolated nodes that never triggered naturally")
md_lines.append("")
md_lines.append("### Key Numbers for Strategy Code")
md_lines.append("")

best_seed_v = top_profit[0] if top_profit else None
if best_seed_v:
    md_lines.append(f"- Best single seed by profit: **Node {best_seed_v}** → "
                    f"{roi_scores[best_seed_v]['viral']} viral, "
                    f"profit {roi_scores[best_seed_v]['profit']:,}₽")
md_lines.append(f"- Max k-core: **{max_core}** with {len(top_k_nodes)} nodes")
md_lines.append(f"- Communities: **{n_communities}**, modularity **{modularity:.4f}**")
if spectral_ok:
    md_lines.append(f"- Algebraic connectivity: **{alg_conn:.4f}** (graph robustness)")
md_lines.append(f"- Articulation points: **{len(art_points)}** (cascade bridge opportunities)")
md_lines.append(f"- Total positive-ROI small-component profit: **{total_small_profit:,}₽**")
md_lines.append("")
md_lines.append("### Composite Seed Score Formula (for strategy code)")
md_lines.append("```python")
md_lines.append("def composite_score(v):")
md_lines.append("    # Weights tunable by strategy agent")
md_lines.append("    w_roi      = 0.35")
md_lines.append("    w_core     = 0.25")
md_lines.append("    w_cpr      = 0.20")
md_lines.append("    w_ego      = 0.10")
md_lines.append("    w_bridge   = 0.10")
md_lines.append("")
md_lines.append("    score = (")
md_lines.append("        w_roi    * roi_scores[v]['roi']          +")
md_lines.append("        w_core   * core_number[v] / max_core     +")
md_lines.append("        w_cpr    * cpr[v] * 1000                 +")
md_lines.append("        w_ego    * ego_density[v]                +")
md_lines.append("        w_bridge * inter_comm_edges[v] / 10")
md_lines.append("    )")
md_lines.append("    return score")
md_lines.append("```")
md_lines.append("")

# ── Appendix: Node Score Table
md_lines.append("---")
md_lines.append("## Appendix: Full Score Table (Top 50 candidate nodes)")
md_lines.append("")
# Compute composite score for all candidates
def composite_score(v):
    if v not in roi_scores:
        return 0
    return (
        0.35 * roi_scores[v]['roi']
        + 0.25 * core_number[v] / max_core
        + 0.20 * cpr[v] * 1000
        + 0.10 * ego_density[v]
        + 0.10 * inter_comm_edges.get(v, 0) / 10
    )

all_scored = sorted(top_candidates, key=lambda v: -composite_score(v))[:50]
md_lines.append(fmt_node_table(
    ["Node", "Degree", "Cost(₽)", "Viral", "Profit(₽)", "ROI",
     "k-core", "2-hop", "EgoDens", "BridgeEdges", "Composite"],
    [(v, degrees[v], node_cost(v),
      roi_scores[v]['viral'],
      roi_scores[v]['profit'],
      f"{roi_scores[v]['roi']:.2f}",
      core_number[v],
      two_hop[v],
      f"{ego_density[v]:.3f}",
      inter_comm_edges.get(v, 0),
      f"{composite_score(v):.3f}")
     for v in all_scored]
))
md_lines.append("")

# ──────────────────────────────────────────────────────────────────────────────
# Write Markdown
# ──────────────────────────────────────────────────────────────────────────────
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print(f"Written {OUT_MD}")

# ──────────────────────────────────────────────────────────────────────────────
# Write JSON (machine-readable scores)
# ──────────────────────────────────────────────────────────────────────────────
print("Writing JSON data …")

# Serialise community min_seeds lists
comm_analysis_serial = {
    str(cid): {
        'size':        d['size'],
        'min_seeds':   d['min_seeds'],
        'viral_count': d['viral_count'],
        'seed_cost':   d['seed_cost'],
        'value':       d['value'],
        'roi':         d['roi'],
    }
    for cid, d in comm_analysis.items()
}

json_data = {
    'graph_stats': {
        'nodes': N, 'edges': E,
        'threshold': THRESHOLD,
        'mean_degree': float(mean_deg),
        'median_degree': float(med_deg),
        'num_components': len(components),
        'max_kcore': max_core,
        'num_communities': n_communities,
        'modularity': modularity,
        'algebraic_connectivity': float(alg_conn) if alg_conn is not None else None,
        'num_bridges': len(bridges),
        'num_articulation_points': len(art_points),
    },
    'node_scores': {
        str(v): {
            **{k: (int(val) if isinstance(val, (np.integer,)) else
                   float(val) if isinstance(val, (np.floating, float)) else val)
               for k, val in roi_scores[v].items()},
            'core_number':      int(core_number[v]),
            'two_hop':          int(two_hop[v]),
            'ego_density':      float(ego_density[v]),
            'inter_comm_edges': int(inter_comm_edges.get(v, 0)),
            'cpr':              float(cpr[v]),
            'composite_score':  float(composite_score(v)),
            'community_id':     int(partition[v]),
        }
        for v in top_candidates
    },
    'communities': comm_analysis_serial,
    'top_lists': {
        'top_roi':     [int(v) for v in top_roi],
        'top_profit':  [int(v) for v in top_profit],
        'top_viral':   [int(v) for v in top_viral],
        'top_cpr':     [int(v) for v in top_cpr[:30]],
        'top_bridge_nodes':         [int(v) for v in top_bridge_nodes[:30]],
        'fiedler_bottlenecks':      [int(v) for v in fiedler_bottlenecks[:20]
                                     if v in degrees],
        'articulation_points_top20':[int(v) for v, _ in art_ranked[:20]],
        'max_kcore_cheapest':       [int(v) for v in top_core_cheap],
    },
    'small_components': small_comp_strategy,
    'temporal_profiles': {
        str(v): {str(day): {'new': d['new'], 'cumulative': d['cumulative']}
                 for day, d in tl.items()}
        for v, tl in temporal_examples.items()
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2)
print(f"Written {OUT_JSON}")

print("\n=== SUMMARY ===")
print(f"Max k-core: {max_core} ({len(top_k_nodes)} nodes)")
print(f"Communities: {n_communities}, modularity {modularity:.4f}")
print(f"Articulation points: {len(art_points)}, bridges: {len(bridges)}")
if spectral_ok:
    print(f"Algebraic connectivity: {alg_conn:.6f}")
print(f"Best single seed (profit): node {top_profit[0]} → {roi_scores[top_profit[0]]['profit']:,}₽")
print(f"Small-component extra profit: {total_small_profit:,}₽")
print("Done.")
