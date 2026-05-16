"""
Comprehensive visualization suite for CU Optimal Seeding Challenge.
Generates interactive HTML dashboards and static PNG charts covering:
  1. Network graph colored by community (interactive)
  2. Degree distribution (log-log + histogram)
  3. Centrality comparison scatter matrix
  4. Community analysis (sizes, viral potential, ROI)
  5. Positive-profit seeds analysis
  6. K-core shell structure
  7. Cascade dynamics (threshold table)
  8. Collective Influence vs PageRank vs Degree
  9. Cost-effectiveness frontier
  10. Bridge / articulation-point overlay on network
"""

import json
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from collections import Counter, defaultdict
import os

# ── Load data ──────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
EDGE_FILE = os.path.join(DATA_DIR, "data", "marketing_edges.txt")
JSON_FILE = os.path.join(DATA_DIR, "deep_analysis_data.json")
OUT_DIR   = os.path.join(DATA_DIR, "viz")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading graph...")
G = nx.read_edgelist(EDGE_FILE, nodetype=int)
print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

with open(JSON_FILE, "r", encoding="utf-8") as f:
    analysis = json.load(f)

# Community detection
print("Computing communities...")
partition = community_louvain.best_partition(G, random_state=42)
n_communities = max(partition.values()) + 1
comm_sizes = Counter(partition.values())

# Centralities
print("Computing centralities...")
degree_dict = dict(G.degree())
betweenness = nx.betweenness_centrality(G, k=500, seed=42)
closeness = nx.closeness_centrality(G)
pagerank = nx.pagerank(G, alpha=0.85)
kcore = nx.core_number(G)

nodes = sorted(G.nodes())
node_list = list(nodes)

# ── Helpers ────────────────────────────────────────────────────────────
def save(fig, name, dpi=180):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [ok] saved {path}")


# ========================================================================
# 1. DEGREE DISTRIBUTION  (log-log + histogram)
# ========================================================================
print("\n[1/10] Degree distribution…")
degrees = [degree_dict[n] for n in node_list]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Degree Distribution", fontsize=15, fontweight='bold')

# log-log
deg_counts = Counter(degrees)
xs = sorted(deg_counts.keys())
ys = [deg_counts[x] for x in xs]
axes[0].scatter(xs, ys, s=12, alpha=0.7, c='#2563eb')
axes[0].set_xscale('log'); axes[0].set_yscale('log')
axes[0].set_xlabel('Degree (log)'); axes[0].set_ylabel('Count (log)')
axes[0].set_title('Log-Log Degree Distribution')
axes[0].grid(True, alpha=0.3)

# histogram
axes[1].hist(degrees, bins=60, color='#2563eb', edgecolor='white', alpha=0.8)
axes[1].axvline(np.median(degrees), color='red', ls='--', label=f'Median={np.median(degrees):.0f}')
axes[1].axvline(np.mean(degrees), color='orange', ls='--', label=f'Mean={np.mean(degrees):.1f}')
axes[1].set_xlabel('Degree'); axes[1].set_ylabel('Count')
axes[1].set_title('Degree Histogram')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

save(fig, "01_degree_distribution.png")


# ========================================================================
# 2. CENTRALITY COMPARISON SCATTER MATRIX
# ========================================================================
print("[2/10] Centrality comparison…")
centralities = {
    'Degree': degree_dict,
    'Betweenness': betweenness,
    'Closeness': closeness,
    'PageRank': pagerank,
    'K-core': kcore,
}
names = list(centralities.keys())
n_c = len(names)

fig, axes = plt.subplots(n_c, n_c, figsize=(18, 18))
fig.suptitle("Centrality Pairwise Scatter Matrix", fontsize=16, fontweight='bold', y=0.92)

for i in range(n_c):
    for j in range(n_c):
        ax = axes[i][j]
        xi = [centralities[names[j]][n] for n in node_list]
        yi = [centralities[names[i]][n] for n in node_list]
        if i == j:
            ax.hist(xi, bins=40, color='#6366f1', alpha=0.7, edgecolor='white')
            ax.set_ylabel('Count')
        else:
            ax.scatter(xi, yi, s=2, alpha=0.3, c='#6366f1')
        if i == n_c - 1:
            ax.set_xlabel(names[j], fontsize=9)
        if j == 0:
            ax.set_ylabel(names[i], fontsize=9)
        ax.tick_params(labelsize=6)

fig.tight_layout(rect=[0, 0, 1, 0.9])
save(fig, "02_centrality_scatter_matrix.png")


# ========================================================================
# 3. COMMUNITY ANALYSIS DASHBOARD
# ========================================================================
print("[3/10] Community analysis…")
comm_data = analysis['communities']
comm_ids_sorted = sorted(comm_data.keys(), key=lambda c: comm_data[c]['size'], reverse=True)
# top 20 communities by size
top_comms = comm_ids_sorted[:20]

sizes_arr = [comm_data[c]['size'] for c in top_comms]
viral_arr = [comm_data[c]['viral_count'] for c in top_comms]
roi_arr   = [comm_data[c]['roi'] for c in top_comms]
seeds_arr = [len(comm_data[c]['min_seeds']) for c in top_comms]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Community Analysis (Top 20 by Size)", fontsize=15, fontweight='bold')

# sizes vs viral
ax = axes[0, 0]
x_pos = np.arange(len(top_comms))
w = 0.35
ax.bar(x_pos - w/2, sizes_arr, w, label='Total Size', color='#3b82f6', alpha=0.8)
ax.bar(x_pos + w/2, viral_arr, w, label='Viral Users', color='#f97316', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'C{c}' for c in top_comms], rotation=45, fontsize=7)
ax.set_ylabel('Node Count')
ax.set_title('Community Size vs Viral Reach')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# viral efficiency
ax = axes[0, 1]
efficiency = [v/s if s > 0 else 0 for v, s in zip(viral_arr, sizes_arr)]
colors = ['#22c55e' if e > 0.8 else '#f97316' if e > 0.3 else '#ef4444' for e in efficiency]
ax.bar(x_pos, efficiency, color=colors, edgecolor='white')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'C{c}' for c in top_comms], rotation=45, fontsize=7)
ax.set_ylabel('Viral/Size Ratio')
ax.set_title('Community Cascade Efficiency')
ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# ROI
ax = axes[1, 0]
bar_colors = ['#22c55e' if r > -0.5 else '#f97316' if r > -0.9 else '#ef4444' for r in roi_arr]
ax.bar(x_pos, roi_arr, color=bar_colors, edgecolor='white')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'C{c}' for c in top_comms], rotation=45, fontsize=7)
ax.set_ylabel('ROI')
ax.set_title('Community ROI (seed cost vs viral income)')
ax.axhline(0, ls='-', color='black', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')

# seeds needed
ax = axes[1, 1]
ax.bar(x_pos, seeds_arr, color='#8b5cf6', edgecolor='white')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'C{c}' for c in top_comms], rotation=45, fontsize=7)
ax.set_ylabel('Seeds Needed (for 90% cascade)')
ax.set_title('Min Seeds to Cascade 90% of Community')
ax.grid(True, alpha=0.3, axis='y')

fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "03_community_analysis.png")


# ========================================================================
# 4. POSITIVE-PROFIT SEEDS
# ========================================================================
print("[4/10] Positive-profit seeds…")
pos_seeds = analysis['optimal_seeds']['positive_profit_seeds']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Positive-Profit Single-Seed Analysis (33 nodes)", fontsize=15, fontweight='bold')

seed_ids = list(pos_seeds.keys())
profits = [pos_seeds[s]['profit'] for s in seed_ids]
costs   = [pos_seeds[s]['cost'] for s in seed_ids]
virals  = [pos_seeds[s]['viral'] for s in seed_ids]
degs    = [pos_seeds[s]['deg'] for s in seed_ids]
comms   = [pos_seeds[s]['community'] for s in seed_ids]

# cost vs profit scatter
ax = axes[0]
sc = ax.scatter(costs, profits, c=virals, s=[d*3+20 for d in degs], cmap='YlOrRd', alpha=0.8, edgecolors='black', linewidth=0.5)
for i, sid in enumerate(seed_ids):
    if profits[i] > 15000 or costs[i] > 10000:
        ax.annotate(sid, (costs[i], profits[i]), fontsize=7, ha='left', va='bottom')
ax.set_xlabel('Cost (rubles)')
ax.set_ylabel('Profit (rubles)')
ax.set_title('Cost vs Profit (size=degree, color=viral)')
plt.colorbar(sc, ax=ax, label='Viral users')
ax.grid(True, alpha=0.3)

# profit by community
ax = axes[1]
comm_profits = defaultdict(list)
for i, s in enumerate(seed_ids):
    comm_profits[comms[i]].append(profits[i])
comm_keys = sorted(comm_profits.keys())
comm_max = [max(comm_profits[c]) for c in comm_keys]
comm_avg = [np.mean(comm_profits[c]) for c in comm_keys]
x2 = np.arange(len(comm_keys))
ax.bar(x2 - 0.2, comm_max, 0.4, label='Max Profit', color='#22c55e', alpha=0.8)
ax.bar(x2 + 0.2, comm_avg, 0.4, label='Avg Profit', color='#3b82f6', alpha=0.8)
ax.set_xticks(x2)
ax.set_xticklabels([f'C{c}' for c in comm_keys], rotation=45)
ax.set_ylabel('Profit (rubles)')
ax.set_title('Profit by Community')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# degree vs viral with ROI annotation
ax = axes[2]
roi_vals = [p/c if c > 0 else 0 for p, c in zip(profits, costs)]
sc2 = ax.scatter(degs, virals, c=roi_vals, s=80, cmap='RdYlGn', alpha=0.8, edgecolors='black', linewidth=0.5)
for i, sid in enumerate(seed_ids):
    if virals[i] > 500 or roi_vals[i] > 5:
        ax.annotate(sid, (degs[i], virals[i]), fontsize=7, ha='left')
ax.set_xlabel('Degree')
ax.set_ylabel('Viral Users')
ax.set_title('Degree vs Viral (color=ROI)')
plt.colorbar(sc2, ax=ax, label='ROI (profit/cost)')
ax.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "04_positive_profit_seeds.png")


# ========================================================================
# 5. K-CORE DECOMPOSITION
# ========================================================================
print("[5/10] K-core decomposition…")
core_counts = Counter(kcore.values())
core_vals = sorted(core_counts.keys())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("K-Core Decomposition", fontsize=15, fontweight='bold')

# shell sizes
ax = axes[0]
ax.bar(core_vals, [core_counts[c] for c in core_vals], color='#6366f1', alpha=0.8, edgecolor='white')
ax.set_xlabel('K-core Number')
ax.set_ylabel('Nodes in Shell')
ax.set_title('K-Core Shell Sizes')
ax.grid(True, alpha=0.3, axis='y')

# cumulative
ax = axes[1]
cum_sizes = np.cumsum([core_counts[c] for c in core_vals])
ax.plot(core_vals, cum_sizes, 'o-', color='#2563eb', markersize=3)
ax.fill_between(core_vals, cum_sizes, alpha=0.15, color='#2563eb')
ax.set_xlabel('K-core Number')
ax.set_ylabel('Cumulative Nodes')
ax.set_title('Cumulative Nodes by Core')
ax.grid(True, alpha=0.3)

# core vs degree
ax = axes[2]
ax.scatter([degree_dict[n] for n in node_list], [kcore[n] for n in node_list],
           s=3, alpha=0.3, c='#6366f1')
ax.set_xlabel('Degree')
ax.set_ylabel('K-core Number')
ax.set_title('Degree vs K-core (correlation)')
ax.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "05_kcore_decomposition.png")


# ========================================================================
# 6. CASCADE DYNAMICS
# ========================================================================
print("[6/10] Cascade dynamics…")
threshold_table = analysis['cascade_dynamics']['threshold_table']
easy_targets = analysis['cascade_dynamics']['easy_target_counts']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Cascade Dynamics (Threshold = 18%)", fontsize=15, fontweight='bold')

# threshold: degree → neighbors needed
ax = axes[0]
degs_t = sorted(int(d) for d in threshold_table.keys())
nbrs_needed = [threshold_table[str(d)] for d in degs_t]
ax.step(degs_t[:50], nbrs_needed[:50], where='post', color='#ef4444', linewidth=2)
ax.fill_between(degs_t[:50], nbrs_needed[:50], step='post', alpha=0.15, color='#ef4444')
ax.set_xlabel('Node Degree')
ax.set_ylabel('Infected Neighbors Needed')
ax.set_title('Activation Threshold by Degree')
ax.grid(True, alpha=0.3)

# easy targets pie
ax = axes[1]
labels = ['Need 1', 'Need 2', 'Need 3', 'Need 8+']
vals = [easy_targets.get(k, 0) for k in ['need_1_nbr', 'need_2_nbr', 'need_3_nbr', 'need_8plus']]
colors_pie = ['#22c55e', '#84cc16', '#f97316', '#ef4444']
wedges, texts, autotexts = ax.pie(vals, labels=labels, autopct='%1.0f%%', colors=colors_pie, startangle=90)
ax.set_title('Cascade Difficulty Distribution')

# degree histogram colored by ease
ax = axes[2]
easy_nodes = {1: [], 2: [], 3: [], 8: []}
for n in node_list:
    d = degree_dict[n]
    needed = int(np.ceil(d * 0.18))
    if needed <= 1: easy_nodes[1].append(d)
    elif needed <= 2: easy_nodes[2].append(d)
    elif needed <= 3: easy_nodes[3].append(d)
    else: easy_nodes[8].append(d)
ax.hist([easy_nodes[1], easy_nodes[2], easy_nodes[3], easy_nodes[8]],
        bins=40, stacked=True, color=colors_pie, label=labels, edgecolor='white', alpha=0.85)
ax.set_xlabel('Degree')
ax.set_ylabel('Count')
ax.set_title('Degree Distribution by Cascade Difficulty')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "06_cascade_dynamics.png")


# ========================================================================
# 7. COLLECTIVE INFLUENCE vs PAGERANK vs DEGREE
# ========================================================================
print("[7/10] CI vs PageRank vs Degree…")
ci_data = analysis['collective_influence']
pr_data = analysis['pagerank']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Centrality Metrics Comparison (Top Nodes)", fontsize=15, fontweight='bold')

# CI vs Degree for top CI nodes
ax = axes[0]
ci_nodes = list(ci_data.keys())
ci_vals_plot = [ci_data[n]['ci'] for n in ci_nodes]
ci_degs = [degree_dict[int(n)] for n in ci_nodes]
ax.scatter(ci_degs, ci_vals_plot, s=60, c='#8b5cf6', edgecolors='black', linewidth=0.5, alpha=0.8)
for n in ci_nodes[:5]:
    ax.annotate(n, (degree_dict[int(n)], ci_data[n]['ci']), fontsize=7)
ax.set_xlabel('Degree')
ax.set_ylabel('Collective Influence (CI₁)')
ax.set_title('Degree vs Collective Influence')
ax.grid(True, alpha=0.3)

# CI-ROI vs PR-ROI
ax = axes[1]
common_nodes = [n for n in ci_nodes if n in pr_data]
if common_nodes:
    ci_roi_vals = [ci_data[n]['ci_roi'] for n in common_nodes]
    pr_roi_vals = [pr_data[n]['pr_roi'] for n in common_nodes]
    ax.scatter(ci_roi_vals, pr_roi_vals, s=60, c='#f97316', edgecolors='black', linewidth=0.5, alpha=0.8)
    for n in common_nodes[:5]:
        ax.annotate(n, (ci_data[n]['ci_roi'], pr_data[n]['pr_roi']), fontsize=7)
ax.set_xlabel('CI-ROI')
ax.set_ylabel('PageRank-ROI')
ax.set_title('CI-ROI vs PageRank-ROI')
ax.grid(True, alpha=0.3)

# PageRank distribution all nodes
ax = axes[2]
pr_all = [pagerank[n] for n in node_list]
ax.hist(pr_all, bins=80, color='#2563eb', edgecolor='white', alpha=0.8)
ax.set_xlabel('PageRank')
ax.set_ylabel('Count')
ax.set_title('PageRank Distribution (all nodes)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "07_ci_pagerank_degree.png")


# ========================================================================
# 8. COST-EFFECTIVENESS FRONTIER
# ========================================================================
print("[8/10] Cost-effectiveness frontier…")
node_scores = analysis['node_scores']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Cost-Effectiveness Analysis", fontsize=15, fontweight='bold')

# all scored nodes: cost vs viral
ax = axes[0]
ns_nodes = list(node_scores.keys())
ns_costs = [node_scores[n]['cost'] for n in ns_nodes]
ns_viral = [node_scores[n]['viral'] for n in ns_nodes]
ns_comm  = [partition.get(int(n), 0) for n in ns_nodes]
sc = ax.scatter(ns_costs, ns_viral, c=ns_comm, s=15, cmap='tab20', alpha=0.6, edgecolors='none')
# highlight positive profit
for sid, sdata in pos_seeds.items():
    ax.scatter(sdata['cost'], sdata['viral'], s=80, c='red', marker='*', zorder=5)
ax.set_xlabel('Seeding Cost (rubles)')
ax.set_ylabel('Viral Users Reached')
ax.set_title('Cost vs Viral (★ = positive profit, color = community)')
ax.grid(True, alpha=0.3)

# pareto frontier
ax = axes[1]
# sort by cost, track pareto front
items = [(node_scores[n]['cost'], node_scores[n]['viral'], n) for n in ns_nodes]
items.sort(key=lambda x: x[0])
pareto = []
best_viral = -1
for cost, viral, nid in items:
    if viral > best_viral:
        best_viral = viral
        pareto.append((cost, viral, nid))
pareto_costs = [p[0] for p in pareto]
pareto_viral = [p[1] for p in pareto]
ax.scatter(ns_costs, ns_viral, s=10, alpha=0.3, c='gray', label='All nodes')
ax.plot(pareto_costs, pareto_viral, 'r-o', markersize=5, linewidth=2, label='Pareto frontier', zorder=5)
for p in pareto:
    ax.annotate(p[2], (p[0], p[1]), fontsize=6, color='red')
ax.set_xlabel('Seeding Cost (rubles)')
ax.set_ylabel('Viral Users Reached')
ax.set_title('Pareto Frontier: Cost vs Viral')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "08_cost_effectiveness.png")


# ========================================================================
# 9. NETWORK VISUALIZATION (communities + bridges)
# ========================================================================
print("[9/10] Network layout (this may take a minute)…")

# Use only giant component for clarity
giant = max(nx.connected_components(G), key=len)
Gg = G.subgraph(giant).copy()

# layout — spring is expensive for 3700+ nodes, use a faster approach
# kamada_kawai on a coarsened graph, then refine
print("  computing layout...")
pos = nx.spring_layout(Gg, k=0.3, iterations=50, seed=42)

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.set_title("Network Graph — Giant Component (colored by community, size ∝ degree)",
             fontsize=14, fontweight='bold')

# draw edges (thin, gray)
edge_x, edge_y = [], []
for u, v in Gg.edges():
    x0, y0 = pos[u]; x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
ax.plot(edge_x, edge_y, '-', color='#cccccc', linewidth=0.05, alpha=0.3)

# draw nodes
node_xs = [pos[n][0] for n in Gg.nodes()]
node_ys = [pos[n][1] for n in Gg.nodes()]
node_colors = [partition.get(n, 0) for n in Gg.nodes()]
node_sizes = [np.sqrt(degree_dict[n]) * 2 + 1 for n in Gg.nodes()]

sc = ax.scatter(node_xs, node_ys, c=node_colors, s=node_sizes, cmap='tab20',
                alpha=0.7, edgecolors='none', zorder=2)

# highlight articulation points
art_points = list(nx.articulation_points(Gg))
art_x = [pos[n][0] for n in art_points]
art_y = [pos[n][1] for n in art_points]
ax.scatter(art_x, art_y, s=30, facecolors='none', edgecolors='red', linewidth=0.8,
           zorder=3, label=f'Articulation points ({len(art_points)})')

# highlight top profit seeds
for sid in list(pos_seeds.keys())[:10]:
    n = int(sid)
    if n in pos:
        ax.scatter(pos[n][0], pos[n][1], s=120, marker='*', c='yellow',
                   edgecolors='black', linewidth=1, zorder=4)
        ax.annotate(sid, (pos[n][0], pos[n][1]), fontsize=6, fontweight='bold',
                    color='black', zorder=5)

ax.legend(loc='upper left', fontsize=10)
ax.axis('off')

save(fig, "09_network_communities.png", dpi=150)


# ========================================================================
# 10. SUMMARY DASHBOARD
# ========================================================================
print("[10/10] Summary dashboard…")
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.3)

stats = analysis['graph_stats']

# text summary panel
ax = fig.add_subplot(gs[0, 0])
ax.axis('off')
info_text = (
    f"GRAPH STATS\n"
    f"-------------\n"
    f"Nodes: {stats['nodes']:,}\n"
    f"Edges: {stats['edges']:,}\n"
    f"Mean degree: {stats['mean_degree']:.1f}\n"
    f"Median degree: {stats['median_degree']:.0f}\n"
    f"Components: {stats['num_components']}\n"
    f"Max k-core: {stats['max_kcore']}\n"
    f"Communities: {stats['num_communities']}\n"
    f"Modularity: {stats['modularity']:.4f}\n"
    f"Fiedler: {stats['algebraic_connectivity']:.4f}\n"
    f"Bridges: {stats['num_bridges']}\n"
    f"Art. points: {stats['num_articulation_points']}\n"
    f"Positive seeds: {len(pos_seeds)}"
)
ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f9ff', edgecolor='#3b82f6', alpha=0.8))

# degree dist mini
ax = fig.add_subplot(gs[0, 1])
ax.scatter(xs, ys, s=8, alpha=0.7, c='#2563eb')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title('Degree Dist.', fontsize=10)
ax.grid(True, alpha=0.3)

# community sizes
ax = fig.add_subplot(gs[0, 2])
top10_comms = comm_ids_sorted[:10]
ax.barh(range(len(top10_comms)),
        [comm_data[c]['size'] for c in top10_comms],
        color='#3b82f6', alpha=0.8)
ax.set_yticks(range(len(top10_comms)))
ax.set_yticklabels([f'C{c}' for c in top10_comms], fontsize=8)
ax.set_title('Top 10 Communities', fontsize=10)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# k-core dist mini
ax = fig.add_subplot(gs[0, 3])
ax.bar(core_vals, [core_counts[c] for c in core_vals], color='#6366f1', alpha=0.7, width=1)
ax.set_title('K-core Shells', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# centrality pairs row
pairs = [('Degree', 'Betweenness'), ('Degree', 'PageRank'),
         ('Betweenness', 'Closeness'), ('K-core', 'Degree')]
for idx, (n1, n2) in enumerate(pairs):
    ax = fig.add_subplot(gs[1, idx])
    x_vals = [centralities[n1][n] for n in node_list]
    y_vals = [centralities[n2][n] for n in node_list]
    ax.scatter(x_vals, y_vals, s=2, alpha=0.2, c='#6366f1')
    ax.set_xlabel(n1, fontsize=8)
    ax.set_ylabel(n2, fontsize=8)
    ax.set_title(f'{n1} vs {n2}', fontsize=9)
    ax.grid(True, alpha=0.3)

# bottom row: profit seeds, cascade, top seeds table, viral histogram
ax = fig.add_subplot(gs[2, 0:2])
ax.scatter(costs, profits, c=virals, s=[d*3+20 for d in degs], cmap='YlOrRd',
           alpha=0.8, edgecolors='black', linewidth=0.5)
for i, sid in enumerate(seed_ids):
    if profits[i] > 10000:
        ax.annotate(sid, (costs[i], profits[i]), fontsize=7)
ax.set_xlabel('Cost')
ax.set_ylabel('Profit')
ax.set_title('Positive-Profit Seeds (size=degree, color=viral)', fontsize=10)
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[2, 2])
ax.step(degs_t[:40], nbrs_needed[:40], where='post', color='#ef4444', linewidth=2)
ax.fill_between(degs_t[:40], nbrs_needed[:40], step='post', alpha=0.15, color='#ef4444')
ax.set_xlabel('Degree')
ax.set_ylabel('Neighbors needed')
ax.set_title('Cascade Threshold', fontsize=10)
ax.grid(True, alpha=0.3)

# viral distribution
ax = fig.add_subplot(gs[2, 3])
all_viral = [node_scores[n]['viral'] for n in ns_nodes]
ax.hist(all_viral, bins=30, color='#f97316', edgecolor='white', alpha=0.8)
ax.set_xlabel('Viral Users')
ax.set_ylabel('Count')
ax.set_title('Viral Reach Distribution', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle("CU Optimal Seeding Challenge — Analysis Dashboard", fontsize=18, fontweight='bold', y=0.98)
save(fig, "10_summary_dashboard.png")


# ========================================================================
# INTERACTIVE PLOTLY NETWORK
# ========================================================================
print("\n[Bonus] Interactive Plotly network…")
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # reuse pos from matplotlib layout
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.15, color='#aaa'),
        hoverinfo='none', mode='lines', opacity=0.3
    )

    node_x = [pos[n][0] for n in Gg.nodes()]
    node_y = [pos[n][1] for n in Gg.nodes()]
    node_text = [
        f"Node {n}<br>Degree: {degree_dict[n]}<br>K-core: {kcore[n]}"
        f"<br>Community: {partition.get(n,0)}<br>Betweenness: {betweenness[n]:.4f}"
        f"<br>PageRank: {pagerank[n]:.5f}"
        for n in Gg.nodes()
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers', hoverinfo='text', text=node_text,
        marker=dict(
            size=[np.sqrt(degree_dict[n])*1.5 + 2 for n in Gg.nodes()],
            color=[partition.get(n, 0) for n in Gg.nodes()],
            colorscale='Portland',
            line_width=0.3
        )
    )

    # highlight seeds
    seed_trace = go.Scatter(
        x=[pos[int(s)][0] for s in list(pos_seeds.keys())[:15] if int(s) in pos],
        y=[pos[int(s)][1] for s in list(pos_seeds.keys())[:15] if int(s) in pos],
        mode='markers+text',
        text=[s for s in list(pos_seeds.keys())[:15] if int(s) in pos],
        textposition='top center',
        marker=dict(size=15, color='yellow', symbol='star', line=dict(width=1, color='black')),
        name='Profit seeds'
    )

    fig_plotly = go.Figure(data=[edge_trace, node_trace, seed_trace],
                           layout=go.Layout(
                               title='Interactive Network — Communities & Centralities',
                               showlegend=False,
                               hovermode='closest',
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               width=1400, height=1000,
                               template='plotly_white'
                           ))
    plotly_path = os.path.join(OUT_DIR, "network_interactive.html")
    fig_plotly.write_html(plotly_path)
    print(f"  [ok] saved {plotly_path}")
except Exception as e:
    print(f"  [FAIL] Plotly interactive failed: {e}")


print("\n" + "="*60)
print("All visualizations saved to:", OUT_DIR)
print("="*60)
