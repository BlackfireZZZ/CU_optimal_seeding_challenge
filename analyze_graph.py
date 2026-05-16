import sys
sys.stdout.reconfigure(encoding='utf-8')
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import numpy as np
from collections import Counter

# ── 1. Load graph ──────────────────────────────────────────────────────────────
print("Loading graph …")
G = nx.DiGraph()
with open("D:/Prog2/CU_optimal_seeding_challenge/data/marketing_edges.txt") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            G.add_edge(parts[0], parts[1])

print(f"Nodes : {G.number_of_nodes():,}")
print(f"Edges : {G.number_of_edges():,}")

# ── 2. Connected components ────────────────────────────────────────────────────
print("\n── Weakly Connected Components (WCC) ──")
wccs = list(nx.weakly_connected_components(G))
wccs_sorted = sorted(wccs, key=len, reverse=True)
print(f"Number of WCC : {len(wccs):,}")
print(f"Largest WCC size : {len(wccs_sorted[0]):,}  ({100*len(wccs_sorted[0])/G.number_of_nodes():.1f}% of nodes)")
if len(wccs) > 1:
    print(f"2nd largest WCC  : {len(wccs_sorted[1]):,}")
print(f"Singleton WCCs   : {sum(1 for c in wccs if len(c)==1):,}")

print("\n── Strongly Connected Components (SCC) ──")
sccs = list(nx.strongly_connected_components(G))
sccs_sorted = sorted(sccs, key=len, reverse=True)
print(f"Number of SCC : {len(sccs):,}")
print(f"Largest SCC size : {len(sccs_sorted[0]):,}  ({100*len(sccs_sorted[0])/G.number_of_nodes():.1f}% of nodes)")
if len(sccs) > 1:
    print(f"2nd largest SCC  : {len(sccs_sorted[1]):,}")
print(f"Singleton SCCs   : {sum(1 for c in sccs if len(c)==1):,}")

# SCC size distribution
scc_size_counts = Counter(len(c) for c in sccs)
print("\nSCC size distribution (top 10 sizes):")
for size, count in sorted(scc_size_counts.items(), reverse=True)[:10]:
    print(f"  size {size:>6,} : {count} SCC(s)")

# ── 3. Diameter estimate (on largest WCC subgraph) ─────────────────────────────
print("\n── Diameter estimate ──")
Gu = G.to_undirected()
largest_wcc_nodes = wccs_sorted[0]
Gsub_undi = Gu.subgraph(largest_wcc_nodes)

# BFS from several random sources to estimate diameter
random.seed(42)
sample_sources = random.sample(list(largest_wcc_nodes), min(30, len(largest_wcc_nodes)))
max_eccentricity = 0
for src in sample_sources:
    lengths = nx.single_source_shortest_path_length(Gsub_undi, src)
    eccentricity = max(lengths.values())
    if eccentricity > max_eccentricity:
        max_eccentricity = eccentricity
print(f"Diameter estimate (lower bound, 30 BFS samples on largest WCC): {max_eccentricity}")

# ── 4. Centrality metrics ──────────────────────────────────────────────────────
print("\n── Degree Centrality (top 20 nodes) ──")
in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())
tot_deg = {n: in_deg[n] + out_deg[n] for n in G.nodes()}

top20_total = sorted(tot_deg, key=tot_deg.get, reverse=True)[:20]
print(f"{'Node':<15} {'In-deg':>8} {'Out-deg':>9} {'Total':>8}")
print("-" * 42)
for n in top20_total:
    print(f"{str(n):<15} {in_deg[n]:>8,} {out_deg[n]:>9,} {tot_deg[n]:>8,}")

# Betweenness on sampled subgraph (top 500 nodes by degree to keep it fast)
print("\n── Betweenness Centrality (approx, k=500 samples) ──")
bc = nx.betweenness_centrality(G, k=500, normalized=True, seed=42)
top10_bc = sorted(bc, key=bc.get, reverse=True)[:10]
print(f"{'Node':<15} {'Betweenness':>14}")
print("-" * 30)
for n in top10_bc:
    print(f"{str(n):<15} {bc[n]:>14.6f}")

# ── 5. Visualization ───────────────────────────────────────────────────────────
print("\n── Building visualization (top 100 nodes by total degree) ──")
top100_nodes = sorted(tot_deg, key=tot_deg.get, reverse=True)[:100]
Gvis = G.subgraph(top100_nodes).copy()

# Node sizes proportional to total degree (scaled)
degrees_vis = np.array([tot_deg[n] for n in Gvis.nodes()])
node_sizes  = 50 + 3000 * (degrees_vis - degrees_vis.min()) / max(degrees_vis.max() - degrees_vis.min(), 1)

# Node colour by in-degree (warm palette)
in_degs_vis = np.array([in_deg[n] for n in Gvis.nodes()])
norm = plt.Normalize(vmin=in_degs_vis.min(), vmax=in_degs_vis.max())
node_colors = cm.plasma(norm(in_degs_vis))

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# --- subplot 1: full top-100 subgraph ---
ax1 = axes[0]
random.seed(42)
pos = nx.spring_layout(Gvis, seed=42, k=0.6, iterations=80)
nx.draw_networkx_edges(
    Gvis, pos, ax=ax1,
    alpha=0.25, width=0.5, edge_color='#555555',
    arrows=True, arrowsize=8,
    connectionstyle='arc3,rad=0.05'
)
sc = nx.draw_networkx_nodes(
    Gvis, pos, ax=ax1,
    node_size=node_sizes,
    node_color=node_colors,
    alpha=0.92
)
# label only top-20
top20_set = set(top20_total)
labels = {n: str(n) for n in Gvis.nodes() if n in top20_set}
nx.draw_networkx_labels(Gvis, pos, labels=labels, ax=ax1, font_size=6, font_color='white')
sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax1, fraction=0.03, pad=0.02, label='In-degree')
ax1.set_title("Top 100 nodes by total degree\n(node size ∝ total degree, colour ∝ in-degree)", fontsize=11)
ax1.axis('off')

# --- subplot 2: degree distribution (log-log) ---
ax2 = axes[1]
all_total_degrees = list(tot_deg.values())
degree_counts = Counter(all_total_degrees)
xs = sorted(degree_counts.keys())
ys = [degree_counts[x] for x in xs]
ax2.loglog(xs, ys, 'o', markersize=3, alpha=0.6, color='steelblue', label='Total degree')

in_degree_counts = Counter(in_deg.values())
xs2 = sorted(in_degree_counts.keys())
ys2 = [in_degree_counts[x] for x in xs2]
ax2.loglog(xs2, ys2, 's', markersize=3, alpha=0.6, color='crimson', label='In-degree')

ax2.set_xlabel("Degree (log scale)")
ax2.set_ylabel("Count (log scale)")
ax2.set_title("Degree distribution (log-log)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(
    f"Marketing Graph — {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges\n"
    f"WCC: {len(wccs):,} | SCC: {len(sccs):,} | Largest WCC: {len(wccs_sorted[0]):,} nodes | Largest SCC: {len(sccs_sorted[0]):,} nodes",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
out_path = "D:/Prog2/CU_optimal_seeding_challenge/graph_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to {out_path}")

print("\nDone.")
