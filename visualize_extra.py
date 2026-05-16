"""
Extra visualizations:
  1. Centrality correlation heatmap
  2. Fixed Pareto frontier (only nodes with viral > 0)
  3. Community-level meta-graph
  4. Degree vs Betweenness colored by community (zoomed)
  5. Top-30 nodes ranked by different centralities (parallel coordinates style)
"""

import json
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from collections import Counter, defaultdict
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
EDGE_FILE = os.path.join(DATA_DIR, "data", "marketing_edges.txt")
JSON_FILE = os.path.join(DATA_DIR, "deep_analysis_data.json")
OUT_DIR   = os.path.join(DATA_DIR, "viz")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading graph...")
G = nx.read_edgelist(EDGE_FILE, nodetype=int)

with open(JSON_FILE, "r", encoding="utf-8") as f:
    analysis = json.load(f)

partition = community_louvain.best_partition(G, random_state=42)
degree_dict = dict(G.degree())
node_list = sorted(G.nodes())

print("Computing centralities...")
betweenness = nx.betweenness_centrality(G, k=500, seed=42)
closeness = nx.closeness_centrality(G)
pagerank = nx.pagerank(G, alpha=0.85)
kcore = nx.core_number(G)
eigenvector = nx.eigenvector_centrality(G, max_iter=500)

def save(fig, name, dpi=180):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [ok] saved {path}")

# ========================================================================
# 1. CENTRALITY CORRELATION HEATMAP
# ========================================================================
print("\n[1/5] Centrality correlation heatmap...")
centralities = {
    'Degree': degree_dict,
    'Betweenness': betweenness,
    'Closeness': closeness,
    'PageRank': pagerank,
    'K-core': kcore,
    'Eigenvector': eigenvector,
}
names = list(centralities.keys())
n_c = len(names)

# build matrix
matrix = np.zeros((len(node_list), n_c))
for j, name in enumerate(names):
    for i, n in enumerate(node_list):
        matrix[i, j] = centralities[name][n]

corr = np.corrcoef(matrix.T)

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(n_c))
ax.set_yticks(range(n_c))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(names, fontsize=11)

# annotate values
for i in range(n_c):
    for j in range(n_c):
        color = 'white' if abs(corr[i, j]) > 0.7 else 'black'
        ax.text(j, i, f'{corr[i,j]:.3f}', ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)

plt.colorbar(im, ax=ax, label='Pearson Correlation', shrink=0.8)
ax.set_title('Centrality Correlation Heatmap', fontsize=15, fontweight='bold', pad=15)
save(fig, "11_centrality_correlation_heatmap.png")


# ========================================================================
# 2. FIXED COST-EFFECTIVENESS (only viral > 0 nodes + proper Pareto)
# ========================================================================
print("[2/5] Fixed cost-effectiveness...")
node_scores = analysis['node_scores']
pos_seeds = analysis['optimal_seeds']['positive_profit_seeds']

# filter to nodes with viral > 0
active_nodes = {n: v for n, v in node_scores.items() if v['viral'] > 0}
print(f"  {len(active_nodes)} nodes with viral > 0 out of {len(node_scores)}")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Cost-Effectiveness (nodes with viral > 0 only)", fontsize=15, fontweight='bold')

ax = axes[0]
ac_costs = [active_nodes[n]['cost'] for n in active_nodes]
ac_viral = [active_nodes[n]['viral'] for n in active_nodes]
ac_comm  = [partition.get(int(n), 0) for n in active_nodes]
ac_profit = [active_nodes[n]['profit'] for n in active_nodes]
sc = ax.scatter(ac_costs, ac_viral, c=ac_profit, s=40, cmap='RdYlGn', alpha=0.8,
                edgecolors='black', linewidth=0.3)
# mark positive profit
for sid, sdata in pos_seeds.items():
    ax.scatter(sdata['cost'], sdata['viral'], s=120, c='none', marker='o',
               edgecolors='red', linewidth=2, zorder=5)
    ax.annotate(sid, (sdata['cost'], sdata['viral']), fontsize=7, color='red',
                fontweight='bold', ha='left', va='bottom')
ax.set_xlabel('Seeding Cost (rubles)', fontsize=12)
ax.set_ylabel('Viral Users', fontsize=12)
ax.set_title('Cost vs Viral (color = profit)')
plt.colorbar(sc, ax=ax, label='Profit (rubles)')
ax.grid(True, alpha=0.3)

# Pareto frontier (proper)
ax = axes[1]
items = sorted([(active_nodes[n]['cost'], active_nodes[n]['viral'], n) for n in active_nodes],
               key=lambda x: x[0])
pareto = []
best_viral = -1
for cost, viral, nid in items:
    if viral > best_viral:
        best_viral = viral
        pareto.append((cost, viral, nid))

ax.scatter(ac_costs, ac_viral, s=20, alpha=0.4, c='#94a3b8', label='Active nodes')
pareto_c = [p[0] for p in pareto]
pareto_v = [p[1] for p in pareto]
ax.plot(pareto_c, pareto_v, 'r-o', markersize=6, linewidth=2, label='Pareto frontier', zorder=5)
for p in pareto:
    ax.annotate(p[2], (p[0], p[1]), fontsize=7, color='red', ha='right')
# fill area under pareto
ax.fill_between(pareto_c, pareto_v, alpha=0.08, color='red')
ax.set_xlabel('Seeding Cost (rubles)', fontsize=12)
ax.set_ylabel('Viral Users', fontsize=12)
ax.set_title('Pareto Frontier (cost vs viral reach)')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "12_cost_effectiveness_fixed.png")


# ========================================================================
# 3. COMMUNITY META-GRAPH
# ========================================================================
print("[3/5] Community meta-graph...")
comm_data = analysis['communities']

# build meta-graph: communities as nodes, inter-community edges as weighted edges
meta_edges = defaultdict(int)
for u, v in G.edges():
    cu = partition.get(u, -1)
    cv = partition.get(v, -1)
    if cu != cv and cu >= 0 and cv >= 0:
        key = (min(cu, cv), max(cu, cv))
        meta_edges[key] += 1

# only communities with size >= 10
big_comms = {c for c, data in comm_data.items() if data['size'] >= 10}
comm_sizes = {c: comm_data[c]['size'] for c in big_comms}
comm_viral_ratio = {c: comm_data[c]['viral_count'] / comm_data[c]['size']
                    for c in big_comms if comm_data[c]['size'] > 0}

MG = nx.Graph()
for c in big_comms:
    MG.add_node(int(c), size=comm_data[c]['size'],
                viral_ratio=comm_viral_ratio.get(c, 0))

for (c1, c2), w in meta_edges.items():
    if str(c1) in big_comms and str(c2) in big_comms and w >= 5:
        MG.add_edge(int(c1), int(c2), weight=w)

fig, ax = plt.subplots(figsize=(14, 14))
ax.set_title("Community Meta-Graph\n(node size = community size, color = viral efficiency, edge width = cross-edges)",
             fontsize=13, fontweight='bold')

pos_meta = nx.spring_layout(MG, k=2.5, seed=42, weight='weight')

# edge widths
edge_weights = [MG[u][v]['weight'] for u, v in MG.edges()]
max_w = max(edge_weights) if edge_weights else 1
edge_widths = [w / max_w * 8 + 0.5 for w in edge_weights]

# node sizes and colors
node_sizes_meta = [comm_sizes.get(str(n), 10) * 3 for n in MG.nodes()]
node_colors_meta = [comm_viral_ratio.get(str(n), 0) for n in MG.nodes()]

nx.draw_networkx_edges(MG, pos_meta, ax=ax, width=edge_widths, alpha=0.3, edge_color='#94a3b8')
sc = nx.draw_networkx_nodes(MG, pos_meta, ax=ax, node_size=node_sizes_meta,
                            node_color=node_colors_meta, cmap='RdYlGn', vmin=0, vmax=1,
                            edgecolors='black', linewidths=1)

labels = {n: f"C{n}\n({comm_sizes.get(str(n), '?')})" for n in MG.nodes()}
nx.draw_networkx_labels(MG, pos_meta, labels, ax=ax, font_size=8, font_weight='bold')

# edge labels for top edges
top_edges = sorted(MG.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:15]
edge_labels_dict = {(u, v): str(d['weight']) for u, v, d in top_edges}
nx.draw_networkx_edge_labels(MG, pos_meta, edge_labels_dict, ax=ax, font_size=7, font_color='#666')

plt.colorbar(sc, ax=ax, label='Viral Efficiency (viral/size)', shrink=0.6)
ax.axis('off')
save(fig, "13_community_metagraph.png")


# ========================================================================
# 4. DEGREE vs BETWEENNESS (zoomed, colored by community)
# ========================================================================
print("[4/5] Degree vs Betweenness detailed...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Degree vs Betweenness Centrality", fontsize=15, fontweight='bold')

degs_all = [degree_dict[n] for n in node_list]
bet_all = [betweenness[n] for n in node_list]
comms_all = [partition.get(n, 0) for n in node_list]

# full view
ax = axes[0]
sc = ax.scatter(degs_all, bet_all, c=comms_all, s=8, cmap='tab20', alpha=0.6)
ax.set_xlabel('Degree', fontsize=12)
ax.set_ylabel('Betweenness Centrality', fontsize=12)
ax.set_title('Full view (color = community)')
ax.grid(True, alpha=0.3)

# annotate top-10 betweenness
top_bet = sorted(range(len(node_list)), key=lambda i: bet_all[i], reverse=True)[:10]
for idx in top_bet:
    ax.annotate(str(node_list[idx]),
                (degs_all[idx], bet_all[idx]),
                fontsize=7, fontweight='bold', color='red')

# zoomed to low-degree high-betweenness (interesting bridge nodes)
ax = axes[1]
# filter: degree < 100, betweenness > 0.005
mask = [(d, b, c, n) for d, b, c, n in zip(degs_all, bet_all, comms_all, node_list)
        if d < 100 and b > 0.002]
if mask:
    md, mb, mc, mn = zip(*mask)
    sc2 = ax.scatter(md, mb, c=mc, s=30, cmap='tab20', alpha=0.7, edgecolors='black', linewidth=0.3)
    for d, b, c, n in mask:
        if b > 0.01:
            ax.annotate(str(n), (d, b), fontsize=7, fontweight='bold')
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Betweenness Centrality', fontsize=12)
    ax.set_title('Bridge nodes: low degree + high betweenness')
    ax.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "14_degree_vs_betweenness.png")


# ========================================================================
# 5. TOP-30 NODES RANKING COMPARISON (bump chart)
# ========================================================================
print("[5/5] Top-30 multi-centrality ranking...")

# rank nodes by each centrality and show how rankings differ
rankings = {}
for name, cent_dict in centralities.items():
    sorted_nodes = sorted(node_list, key=lambda n: cent_dict[n], reverse=True)
    rankings[name] = {n: rank for rank, n in enumerate(sorted_nodes[:50])}

# find nodes that appear in top-30 of at least 2 centralities
top30_sets = {name: set(list(rankings[name].keys())[:30]) for name in names}
common_nodes = set()
for n1 in names:
    for n2 in names:
        if n1 != n2:
            common_nodes |= (top30_sets[n1] & top30_sets[n2])

# take top 20 most frequently ranked nodes
node_freq = Counter()
for name in names:
    for n in list(rankings[name].keys())[:30]:
        node_freq[n] += 1
top_nodes = [n for n, _ in node_freq.most_common(20)]

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_title("Top Node Rankings Across Centrality Metrics\n(lines connect same node across metrics)",
             fontsize=14, fontweight='bold')

x_positions = np.arange(len(names))
cmap = plt.cm.tab20

for idx, node in enumerate(top_nodes):
    y_vals = []
    x_vals = []
    for j, name in enumerate(names):
        if node in rankings[name]:
            y_vals.append(rankings[name][node] + 1)  # 1-indexed rank
            x_vals.append(j)
    if len(x_vals) >= 2:
        color = cmap(idx / len(top_nodes))
        ax.plot(x_vals, y_vals, '-o', color=color, linewidth=2, markersize=8,
                alpha=0.7, label=f'Node {node}')
        # label at rightmost position
        ax.annotate(str(node), (x_vals[-1] + 0.1, y_vals[-1]),
                    fontsize=8, color=color, fontweight='bold', va='center')

ax.set_xticks(x_positions)
ax.set_xticklabels(names, fontsize=12)
ax.set_ylabel('Rank (lower = better)', fontsize=12)
ax.set_ylim(52, 0)  # invert: rank 1 at top
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=7, ncol=4, framealpha=0.9)

fig.tight_layout()
save(fig, "15_multi_centrality_rankings.png")


print("\n" + "="*60)
print("Extra visualizations saved to:", OUT_DIR)
print("="*60)
