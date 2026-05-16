"""
cascade_deep_analysis.py
========================
Supplementary analysis focused on:
  1.  Cascade dynamics: why single seeds fail + what multi-seed threshold is
  2.  Degree-1/2/3 node catalogue (free cascade targets)
  3.  Collective Influence (CI) score — best IM metric for LT model
  4.  Minimum-cost affordable seed sets (budget ≤ 10000)
  5.  Tipping-zone detection: nodes "almost ready" to cascade
  6.  Clique / dense subgraph analysis for cascade amplification
  7.  Profitability analysis: what it takes to break even
  8.  Top seed combinations for day-0 (budget 10000)
  9.  PageRank as cascade predictor
 10.  Community structure within GCC — internal cascade paths

Appends new sections to DEEP_GRAPH_ANALYSIS.md.
"""

import sys, json, math, random
from collections import defaultdict, Counter
import numpy as np
import networkx as nx
import community as community_louvain

sys.stdout.reconfigure(encoding='utf-8')

DATA_FILE = "D:/Prog2/CU_optimal_seeding_challenge/data/marketing_edges.txt"
OUT_MD    = "D:/Prog2/CU_optimal_seeding_challenge/DEEP_GRAPH_ANALYSIS.md"
OUT_JSON  = "D:/Prog2/CU_optimal_seeding_challenge/deep_analysis_data.json"

THRESHOLD  = 0.18
MAX_DAYS   = 60
SEED_COST  = 300
VIRAL_REV  = 50
INIT_BUD   = 10000

random.seed(42)
np.random.seed(42)

print("Loading graph …")
G = nx.read_edgelist(DATA_FILE, nodetype=int)
G = nx.Graph(G)
G.remove_edges_from(nx.selfloop_edges(G))
nodes = list(G.nodes())
N, E = G.number_of_nodes(), G.number_of_edges()
degrees = dict(G.degree())
adj = {v: set(G.neighbors(v)) for v in nodes}

# Load existing JSON
with open(OUT_JSON) as f:
    existing = json.load(f)

core_number = nx.core_number(G)
partition   = community_louvain.best_partition(G, random_state=42)
communities = defaultdict(list)
for node, cid in partition.items():
    communities[cid].append(node)

def node_cost(v):    return SEED_COST * degrees[v]
def simulate_fast(seed_set, max_days=MAX_DAYS):
    active = set(seed_set)
    for _ in range(max_days):
        new_active = set()
        for v in nodes:
            if v in active: continue
            nbrs = adj[v]
            if not nbrs: continue
            if sum(1 for nb in nbrs if nb in active) / len(nbrs) >= THRESHOLD:
                new_active.add(v)
        if not new_active: break
        active |= new_active
    return active - set(seed_set)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Cascade dynamics: threshold analysis
# ──────────────────────────────────────────────────────────────────────────────
print("Cascade dynamics analysis …")

# For each degree, min infected neighbors to cascade
thresh_table = {}
for d in range(1, 300):
    needed = math.ceil(THRESHOLD * d)
    thresh_table[d] = needed

# Distribution of "min neighbors needed"
deg_vals = [degrees[v] for v in nodes]
min_needed = [math.ceil(THRESHOLD * d) for d in deg_vals]
nn_counter = Counter(min_needed)

# How many nodes can be cascaded with exactly 1 seed (because they have deg 1-5)
easy_targets = {
    1: [v for v in nodes if math.ceil(THRESHOLD * degrees[v]) == 1],  # need 1 infected nb
    2: [v for v in nodes if math.ceil(THRESHOLD * degrees[v]) == 2],
    3: [v for v in nodes if math.ceil(THRESHOLD * degrees[v]) == 3],
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Degree-1/2/3 nodes (cheap cascade targets)
# ──────────────────────────────────────────────────────────────────────────────
print("Degree-1/2/3 catalogue …")
low_degree_nodes = {k: sorted(v, key=lambda x: degrees[x]) for k, v in [
    (1, [v for v in nodes if degrees[v] == 1]),
    (2, [v for v in nodes if degrees[v] == 2]),
    (3, [v for v in nodes if degrees[v] == 3]),
    (4, [v for v in nodes if degrees[v] == 4]),
    (5, [v for v in nodes if degrees[v] == 5]),
]}

# For degree-1 nodes: their single neighbor — seeding that neighbor triggers them for free
deg1_parents = {}
for v in low_degree_nodes[1]:
    parent = list(G.neighbors(v))[0]
    deg1_parents[v] = parent

# Group degree-1 leaves by parent
leaves_per_parent = defaultdict(list)
for leaf, parent in deg1_parents.items():
    leaves_per_parent[parent].append(leaf)

top_leaf_parents = sorted(leaves_per_parent.keys(),
                          key=lambda v: len(leaves_per_parent[v]), reverse=True)[:20]

# ──────────────────────────────────────────────────────────────────────────────
# 3. Collective Influence (CI_1) — best IM predictor
# ──────────────────────────────────────────────────────────────────────────────
print("Collective Influence scores …")
# CI_1(v) = (deg(v) - 1) * sum over neighbors u: (deg(u) - 1)
ci_scores = {}
for v in nodes:
    nbrs = list(G.neighbors(v))
    ci = (degrees[v] - 1) * sum(degrees[u] - 1 for u in nbrs)
    ci_scores[v] = ci

top_ci = sorted(nodes, key=lambda v: -ci_scores[v])[:30]

# CI normalized by cost (CI-ROI)
ci_roi = {v: ci_scores[v] / max(node_cost(v), 1) for v in nodes}
top_ci_roi = sorted(nodes, key=lambda v: -ci_roi[v])[:30]

# ──────────────────────────────────────────────────────────────────────────────
# 4. Affordable seed sets (budget ≤ 10000, Day 0)
# ──────────────────────────────────────────────────────────────────────────────
print("Affordable seed sets for Day 0 …")

# All nodes affordable as single seed within budget
affordable = [v for v in nodes if node_cost(v) <= INIT_BUD]
print(f"  Nodes affordable within 10000: {len(affordable)}")

# Nodes affordable with budget leaving room for more seeds
very_cheap = [v for v in nodes if node_cost(v) <= 1000]   # degree ≤ 3
cheap       = [v for v in nodes if 1000 < node_cost(v) <= 3000]   # degree 4-10

# Top combinations: pick best CI-ROI nodes within budget
# Greedy: pick best CI-ROI until budget exhausted
def greedy_budget_seeds(budget, candidates, top_k=10):
    """Pick up to top_k nodes from candidates by CI-ROI within budget."""
    candidates_sorted = sorted(candidates, key=lambda v: -ci_roi[v])
    selected = []
    remaining = budget
    for v in candidates_sorted:
        cost = node_cost(v)
        if cost <= remaining and len(selected) < top_k:
            selected.append(v)
            remaining -= cost
    return selected, budget - remaining

day0_seeds_coi, day0_cost = greedy_budget_seeds(INIT_BUD, affordable, top_k=10)
day0_viral = simulate_fast(day0_seeds_coi, max_days=10)
day0_income = len(day0_viral) * VIRAL_REV
day0_profit = day0_income - day0_cost

print(f"  Day-0 greedy (CI-ROI): {len(day0_seeds_coi)} seeds, cost={day0_cost}, viral={len(day0_viral)}, profit={day0_profit}")

# Alternative: seed from max-k-core cheapest nodes
max_core = max(core_number.values())
kcore_threshold = int(0.7 * max_core)
kcore_cheap = sorted(
    [v for v in nodes if core_number[v] >= kcore_threshold],
    key=lambda v: node_cost(v)
)
day0_seeds_kcore, day0_cost_k = greedy_budget_seeds(INIT_BUD, kcore_cheap, top_k=10)
day0_viral_k = simulate_fast(day0_seeds_kcore, max_days=10)
day0_income_k = len(day0_viral_k) * VIRAL_REV
day0_profit_k = day0_income_k - day0_cost_k

print(f"  Day-0 k-core cheap: {len(day0_seeds_kcore)} seeds, cost={day0_cost_k}, viral={len(day0_viral_k)}, profit={day0_profit_k}")

# Alternative: cheapest nodes in densest community
comm_sizes = sorted(communities.items(), key=lambda x: -len(x[1]))
largest_comm_id, largest_comm = comm_sizes[0]
comm_cheap = sorted(largest_comm, key=lambda v: node_cost(v))
day0_seeds_comm, day0_cost_c = greedy_budget_seeds(INIT_BUD, comm_cheap, top_k=10)
day0_viral_c = simulate_fast(day0_seeds_comm, max_days=10)
day0_income_c = len(day0_viral_c) * VIRAL_REV
day0_profit_c = day0_income_c - day0_cost_c

print(f"  Day-0 largest-community cheap: {len(day0_seeds_comm)} seeds, cost={day0_cost_c}, viral={len(day0_viral_c)}, profit={day0_profit_c}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Tipping zone: nodes "almost ready" to cascade
# ──────────────────────────────────────────────────────────────────────────────
print("Tipping zone detection …")
# Simulate day-0 seeds spreading for 5 days, then find "almost tipping" nodes
def find_tipping_zone(active_set, steps=5):
    """After spreading `steps` days from active_set, find nodes at ≥50% of threshold."""
    current = set(active_set)
    for _ in range(steps):
        new = set()
        for v in nodes:
            if v in current: continue
            nbrs = adj[v]
            if not nbrs: continue
            if sum(1 for nb in nbrs if nb in current) / len(nbrs) >= THRESHOLD:
                new.add(v)
        current |= new

    # Nodes NOT yet active but close to threshold
    tipping = []
    for v in nodes:
        if v in current: continue
        nbrs = adj[v]
        if not nbrs: continue
        frac = sum(1 for nb in nbrs if nb in current) / len(nbrs)
        if frac >= 0.5 * THRESHOLD:  # at 50%+ of threshold
            needed = math.ceil(THRESHOLD * len(nbrs)) - sum(1 for nb in nbrs if nb in current)
            tipping.append({
                'node': v,
                'degree': degrees[v],
                'cost': node_cost(v),
                'current_frac': frac,
                'needed_more': needed,
                'core': core_number[v],
            })
    return current, sorted(tipping, key=lambda x: x['needed_more'])

# Find tipping zone after 3 days with best day-0 seeds
best_day0 = day0_seeds_coi if day0_profit >= day0_profit_k else day0_seeds_kcore
final_active, tipping_nodes = find_tipping_zone(best_day0, steps=3)
print(f"  Active after 3 days: {len(final_active)}, tipping-zone nodes: {len(tipping_nodes)}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Clique analysis: where cascades self-amplify
# ──────────────────────────────────────────────────────────────────────────────
print("Clique / quasi-clique analysis …")
# Find cliques of size ≥ 5
# For large graphs this can be slow, use sampling
gcc_nodes = max(nx.connected_components(G), key=len)
Gcc = G.subgraph(gcc_nodes).copy()

# Use k-core subgraph as proxy for dense subgraphs
top_shell = [v for v in gcc_nodes if core_number[v] >= int(0.8 * max_core)]
Gcore = G.subgraph(top_shell).copy()

# Find triangles (3-cliques) — building block of cascades
triangles = nx.triangles(G)
top_triangle_nodes = sorted(nodes, key=lambda v: -triangles[v])[:20]

# Clique communities within high k-core subgraph
try:
    cliques = list(nx.find_cliques(Gcore))
    cliques_sorted = sorted(cliques, key=len, reverse=True)[:20]
    largest_clique = cliques_sorted[0] if cliques_sorted else []
except Exception:
    cliques_sorted = []
    largest_clique = []

# ──────────────────────────────────────────────────────────────────────────────
# 7. PageRank as cascade predictor
# ──────────────────────────────────────────────────────────────────────────────
print("PageRank analysis …")
pr = nx.pagerank(G, alpha=0.85, max_iter=200)
top_pr = sorted(nodes, key=lambda v: -pr[v])[:30]

# PR per cost (best value PageRank nodes)
pr_roi = {v: pr[v] / max(node_cost(v), 1) for v in nodes}
top_pr_roi = sorted(nodes, key=lambda v: -pr_roi[v])[:30]

# ──────────────────────────────────────────────────────────────────────────────
# 8. Profitability constraint analysis
# ──────────────────────────────────────────────────────────────────────────────
print("Profitability analysis …")

# Break-even: seed with degree d needs 6d viral attributable
# How many cascadeable nodes must a seed "own" to profit?
breakeven_per_degree = {}
for d in range(1, 301):
    cost = SEED_COST * d
    needed_viral = math.ceil(cost / VIRAL_REV)  # = 6d
    breakeven_per_degree[d] = needed_viral

# For each node: estimated cascade ratio needed
# In practice: realistic cascade per single seed in top 200 candidates
cascade_data = existing.get('node_scores', {})
profitable_nodes = [
    (int(v), d) for v, d in cascade_data.items()
    if d['profit'] > 0
]
profitable_nodes.sort(key=lambda x: -x[1]['profit'])

# ──────────────────────────────────────────────────────────────────────────────
# 9. Multi-seed simulation: top seed pairs (greedy pairwise)
# ──────────────────────────────────────────────────────────────────────────────
print("Multi-seed pair simulation (day-0 budget constraint) …")

# Top single seeds we can afford
affordable_ci = sorted(
    [v for v in nodes if node_cost(v) <= INIT_BUD],
    key=lambda v: -ci_scores[v]
)[:50]

# Try pairs with total cost ≤ 10000
best_pair = None
best_pair_viral = 0
best_pair_profit = -999999

checked = 0
for i, v1 in enumerate(affordable_ci[:30]):
    for v2 in affordable_ci[i+1:30]:
        if node_cost(v1) + node_cost(v2) > INIT_BUD:
            continue
        viral = simulate_fast([v1, v2], max_days=15)
        profit = len(viral) * VIRAL_REV - node_cost(v1) - node_cost(v2)
        if profit > best_pair_profit:
            best_pair_profit = profit
            best_pair_viral  = len(viral)
            best_pair = (v1, v2)
        checked += 1

print(f"  Checked {checked} pairs. Best: {best_pair}, viral={best_pair_viral}, profit={best_pair_profit}")

# ──────────────────────────────────────────────────────────────────────────────
# 10. Full greedy campaign simulation (best estimated strategy)
# ──────────────────────────────────────────────────────────────────────────────
print("Full greedy campaign simulation …")

def simulate_full_campaign(strategy_fn, max_days=60):
    """
    Runs a full 60-day campaign using a strategy function.
    strategy_fn(day, budget, active_set) -> list of nodes to seed today
    Returns (total_income, total_cost, profit, day_log)
    """
    active = set()
    budget = INIT_BUD
    total_income = 0
    total_cost   = 0
    day_log = []
    adj_local = adj  # shared adjacency

    for day in range(max_days):
        # Get seeds from strategy
        seeds_today = strategy_fn(day, budget, active)
        seeds_today = [v for v in seeds_today if node_cost(v) <= budget
                       and v not in active][:10]

        # Pay for seeds
        cost_today = sum(node_cost(v) for v in seeds_today)
        budget -= cost_today
        total_cost += cost_today
        active |= set(seeds_today)

        # One viral spread step
        new_viral = set()
        for v in nodes:
            if v in active: continue
            nbrs = adj_local[v]
            if not nbrs: continue
            if sum(1 for nb in nbrs if nb in active) / len(nbrs) >= THRESHOLD:
                new_viral.add(v)

        income_today = len(new_viral) * VIRAL_REV
        total_income += income_today
        budget += income_today
        active |= new_viral

        day_log.append({
            'day': day,
            'seeds': seeds_today,
            'cost': cost_today,
            'viral_new': len(new_viral),
            'income': income_today,
            'budget_after': budget,
            'total_active': len(active),
        })

    profit = total_income - total_cost
    return total_income, total_cost, profit, day_log

# Strategy A: CI-ROI greedy (refresh every 5 days)
def strategy_ci_roi(day, budget, active):
    if day % 5 != 0 and day != 0:
        return []
    candidates = [v for v in nodes if v not in active and node_cost(v) <= budget]
    if not candidates:
        return []
    # Score: CI-ROI but discount already-active neighbors
    def adjusted_ci(v):
        nbrs = adj[v]
        # Bonus: how many of v's neighbors are near-tipping?
        active_nbrs = sum(1 for nb in nbrs if nb in active)
        return ci_scores[v] * (1 + active_nbrs * 0.1) / max(node_cost(v), 1)
    return sorted(candidates, key=adjusted_ci, reverse=True)[:10]

income_A, cost_A, profit_A, log_A = simulate_full_campaign(strategy_ci_roi)
print(f"  Strategy CI-ROI (5-day): income={income_A:,}, cost={cost_A:,}, profit={profit_A:,}")

# Strategy B: Seed every day from cheapest affordable CI-ROI
def strategy_daily_cheap(day, budget, active):
    if budget < 300:
        return []
    candidates = [v for v in nodes if v not in active and node_cost(v) <= min(budget, 3000)]
    if not candidates:
        candidates = [v for v in nodes if v not in active and node_cost(v) <= budget]
    return sorted(candidates, key=lambda v: -ci_roi[v])[:10]

income_B, cost_B, profit_B, log_B = simulate_full_campaign(strategy_daily_cheap)
print(f"  Strategy Daily-Cheap-CI: income={income_B:,}, cost={cost_B:,}, profit={profit_B:,}")

# Strategy C: Aggressive core seeding — day 0 seed max k-core cheap, then bridges
def strategy_core_then_bridges(day, budget, active):
    if day == 0:
        seeds = sorted(
            [v for v in nodes if core_number[v] >= int(0.6 * max_core)
             and node_cost(v) <= INIT_BUD],
            key=lambda v: node_cost(v)
        )[:10]
        affordable_seeds = []
        rem = budget
        for s in seeds:
            if node_cost(s) <= rem:
                affordable_seeds.append(s)
                rem -= node_cost(s)
        return affordable_seeds
    elif day <= 10:
        # Seed bridges / articulation points
        from_data = existing.get('top_lists', {}).get('top_bridge_nodes', [])
        candidates = [v for v in from_data if v not in active and node_cost(v) <= budget]
        return candidates[:10]
    else:
        # Fill with cheap CI-ROI nodes
        candidates = [v for v in nodes if v not in active and node_cost(v) <= min(budget, 5000)]
        return sorted(candidates, key=lambda v: -ci_roi[v])[:10]

income_C, cost_C, profit_C, log_C = simulate_full_campaign(strategy_core_then_bridges)
print(f"  Strategy Core+Bridges: income={income_C:,}, cost={cost_C:,}, profit={profit_C:,}")

# Strategy D: Daily seeding every day from tipping-zone aware nodes
def make_tipping_strategy():
    """Closure that maintains state across days."""
    state = {'tipping': []}

    def strategy(day, budget, active):
        if not active:
            # First call: no seeds yet, pick by CI-ROI
            candidates = [v for v in nodes if v not in active and node_cost(v) <= budget]
            return sorted(candidates, key=lambda v: -ci_roi[v])[:10]

        # Find nodes near threshold
        near_tip = []
        for v in nodes:
            if v in active: continue
            nbrs = adj[v]
            if not nbrs: continue
            frac = sum(1 for nb in nbrs if nb in active) / len(nbrs)
            if frac >= 0.5 * THRESHOLD and node_cost(v) <= budget:
                # How many more we need
                needed = math.ceil(THRESHOLD * len(nbrs)) - sum(1 for nb in nbrs if nb in active)
                near_tip.append((v, needed, node_cost(v), frac))

        # Sort by: smallest "needed" first, then by cost
        near_tip.sort(key=lambda x: (x[1], x[2]))

        selected = []
        rem = budget
        for v, needed, cost, frac in near_tip:
            if cost <= rem and v not in active:
                selected.append(v)
                rem -= cost
            if len(selected) >= 10:
                break

        # Fill remaining slots with CI-ROI nodes
        if len(selected) < 10:
            remaining_slots = 10 - len(selected)
            seed_set = set(selected)
            fillers = [v for v in nodes
                       if v not in active and v not in seed_set
                       and node_cost(v) <= rem]
            fillers.sort(key=lambda v: -ci_roi[v])
            for v in fillers[:remaining_slots]:
                selected.append(v)

        return selected

    return strategy

income_D, cost_D, profit_D, log_D = simulate_full_campaign(make_tipping_strategy())
print(f"  Strategy Tipping-Zone: income={income_D:,}, cost={cost_D:,}, profit={profit_D:,}")

# Best strategy summary
strategies = [
    ('CI-ROI 5-day batch', income_A, cost_A, profit_A, log_A),
    ('Daily Cheap CI-ROI', income_B, cost_B, profit_B, log_B),
    ('Core + Bridges', income_C, cost_C, profit_C, log_C),
    ('Tipping-Zone daily', income_D, cost_D, profit_D, log_D),
]
best_strat = max(strategies, key=lambda x: x[3])
print(f"\n  BEST: {best_strat[0]}, profit={best_strat[3]:,}")

# ──────────────────────────────────────────────────────────────────────────────
# Write supplementary sections to DEEP_GRAPH_ANALYSIS.md
# ──────────────────────────────────────────────────────────────────────────────
print("\nWriting supplementary sections to DEEP_GRAPH_ANALYSIS.md …")

def fmt_table(header_row, rows, max_rows=20):
    lines = ["| " + " | ".join(header_row) + " |"]
    lines.append("|" + "|".join(["---"] * len(header_row)) + "|")
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)

supp = []
supp.append("\n---")
supp.append("## 14. Cascade Dynamics — Why Single Seeds Fail")
supp.append("")
supp.append("**Root cause:** Mean degree = 42.5. Threshold = 18%. Average node needs `ceil(0.18 * 42.5) = 8` infected neighbors.")
supp.append("A single seed can only be 1 infected neighbor for any other node.")
supp.append("")
supp.append("### Min infected neighbors needed to cascade, by degree")
supp.append(fmt_table(
    ["Node degree", "Infected nbrs needed", "# nodes at this degree"],
    [(d, math.ceil(THRESHOLD*d), len([v for v in nodes if degrees[v]==d]))
     for d in [1,2,3,4,5,6,8,10,15,20,30,50,75,100,150,200] if any(degrees[v]==d for v in nodes)]
))
supp.append("")
supp.append("### Distribution of 'neighbors needed'")
supp.append(fmt_table(
    ["Min nbrs needed", "# nodes"],
    sorted(nn_counter.items())[:20]
))
supp.append("")
supp.append("### Key insight")
supp.append("- **Nodes needing just 1 infected neighbor:** `" + str(sum(1 for v in nodes if math.ceil(THRESHOLD*degrees[v])==1)) + "` nodes (degree 1-5)")
supp.append("- **Nodes needing 2:** `" + str(sum(1 for v in nodes if math.ceil(THRESHOLD*degrees[v])==2)) + "` nodes")
supp.append("- **Nodes needing 8+:** `" + str(sum(1 for v in nodes if math.ceil(THRESHOLD*degrees[v])>=8)) + "` nodes — require coordinated multi-seed attack")
supp.append("")

supp.append("---")
supp.append("## 15. Degree-1/2/3 Node Catalogue (Free/Cheap Cascade Targets)")
supp.append("")
for k in [1, 2, 3, 4, 5]:
    supp.append(f"**Degree-{k} nodes:** {len(low_degree_nodes[k])} total, cost = {SEED_COST*k}₽ each, need {math.ceil(THRESHOLD*k)} infected neighbor(s) to cascade")
supp.append("")
supp.append("### Degree-1 leaf nodes and their parents (seeding parent = free viral user)")
supp.append(f"Total degree-1 nodes: {len(low_degree_nodes[1])}")
supp.append("")
supp.append("**Parents with most degree-1 leaves (seed parent to get multiple free virals):**")
supp.append(fmt_table(
    ["Parent node", "Parent degree", "Parent cost (₽)", "# leaf children", "Free virals"],
    [(v, degrees[v], node_cost(v), len(leaves_per_parent[v]), len(leaves_per_parent[v]))
     for v in top_leaf_parents[:15]]
))
supp.append("")
supp.append("> **Strategy:** Seeding a parent with 3+ degree-1 leaves gives free viral users for each leaf.")
supp.append("")

supp.append("---")
supp.append("## 16. Collective Influence (CI_1) Score")
supp.append("")
supp.append("CI_1(v) = (deg(v) - 1) * sum_{u in N(v)} (deg(u) - 1)")
supp.append("Higher CI = node sits at hub of hubs = best cascade initiator for LT model.")
supp.append("")
supp.append("### Top 20 by CI score")
supp.append(fmt_table(
    ["Node", "Degree", "Cost (₽)", "CI score", "k-core"],
    [(v, degrees[v], node_cost(v), ci_scores[v], core_number[v])
     for v in top_ci[:20]]
))
supp.append("")
supp.append("### Top 20 by CI-ROI (CI per ruble spent)")
supp.append(fmt_table(
    ["Node", "Degree", "Cost (₽)", "CI score", "CI-ROI (x1e6)"],
    [(v, degrees[v], node_cost(v), ci_scores[v], f"{ci_roi[v]*1e6:.1f}")
     for v in top_ci_roi[:20]]
))
supp.append("")
supp.append("> **Use CI-ROI ranking** for budget-constrained day-0 seed selection.")
supp.append("")

supp.append("---")
supp.append("## 17. Affordable Day-0 Seed Sets (Budget = 10,000₽)")
supp.append("")
supp.append(f"Affordable nodes (cost ≤ 10000): **{len(affordable)}** nodes")
supp.append(f"Very cheap (cost ≤ 1000, degree ≤ 3): **{len(very_cheap)}** nodes")
supp.append(f"Cheap (cost 1000-3000, degree 4-10): **{len(cheap)}** nodes")
supp.append("")
supp.append("### Greedy CI-ROI budget packing (Day 0)")
supp.append(f"Seeds: {day0_seeds_coi}")
supp.append(f"Total cost: {day0_cost}₽ | Viral (10 days): {len(day0_viral)} | Income: {day0_income}₽ | Profit: {day0_profit}₽")
supp.append("")
supp.append("### k-Core cheap seeds (Day 0)")
supp.append(f"Seeds: {day0_seeds_kcore}")
supp.append(f"Total cost: {day0_cost_k}₽ | Viral (10 days): {len(day0_viral_k)} | Income: {day0_income_k}₽ | Profit: {day0_profit_k}₽")
supp.append("")
supp.append("### Largest-community cheap seeds (Day 0)")
supp.append(f"Seeds: {day0_seeds_comm}")
supp.append(f"Total cost: {day0_cost_c}₽ | Viral (10 days): {len(day0_viral_c)} | Income: {day0_income_c}₽ | Profit: {day0_profit_c}₽")
supp.append("")

if best_pair:
    pair_v1, pair_v2 = best_pair
    supp.append("### Best 2-node combination (within budget)")
    supp.append(f"Nodes: {pair_v1} (deg={degrees[pair_v1]}, cost={node_cost(pair_v1)}₽) + "
                f"{pair_v2} (deg={degrees[pair_v2]}, cost={node_cost(pair_v2)}₽)")
    supp.append(f"Total cost: {node_cost(pair_v1)+node_cost(pair_v2)}₽ | Viral: {best_pair_viral} | Profit: {best_pair_profit}₽")
supp.append("")

supp.append("---")
supp.append("## 18. Tipping-Zone Detection")
supp.append("")
supp.append(f"After 3 days from Day-0 seeds, **{len(tipping_nodes)}** nodes are in the tipping zone (50%+ of threshold infected).")
supp.append("")
if tipping_nodes:
    supp.append("### Tipping-zone nodes needing fewest extra seeds (best ROI to push over the edge)")
    supp.append(fmt_table(
        ["Node", "Degree", "Cost (₽)", "Current frac", "More infected needed", "k-core"],
        [(t['node'], t['degree'], t['cost'],
          f"{t['current_frac']:.3f}", t['needed_more'], t['core'])
         for t in tipping_nodes[:20]]
    ))
    supp.append("")
    supp.append("> **Day 3-5 strategy:** Seed these nodes to push them over the threshold — they're already halfway there, so the cascade propagates at minimal cost.")
supp.append("")

supp.append("---")
supp.append("## 19. PageRank Analysis")
supp.append("")
supp.append("### Top 20 by PageRank")
supp.append(fmt_table(
    ["Node", "Degree", "Cost (₽)", "PageRank", "k-core"],
    [(v, degrees[v], node_cost(v), f"{pr[v]:.5f}", core_number[v])
     for v in top_pr[:20]]
))
supp.append("")
supp.append("### Top 20 by PageRank-ROI (PR per ruble)")
supp.append(fmt_table(
    ["Node", "Degree", "Cost (₽)", "PageRank", "PR-ROI (x1e6)"],
    [(v, degrees[v], node_cost(v), f"{pr[v]:.5f}", f"{pr_roi[v]*1e6:.2f}")
     for v in top_pr_roi[:20]]
))
supp.append("")

supp.append("---")
supp.append("## 20. Triangle Density (Cascade Amplification Zones)")
supp.append("")
supp.append("High triangle count means dense local clustering — cascades self-amplify here.")
supp.append("")
supp.append("### Top 20 nodes by triangle count")
supp.append(fmt_table(
    ["Node", "Degree", "Cost (₽)", "Triangles", "k-core"],
    [(v, degrees[v], node_cost(v), triangles[v], core_number[v])
     for v in top_triangle_nodes[:20]]
))
supp.append("")
if largest_clique:
    supp.append(f"### Largest clique in max-core subgraph: size {len(largest_clique)}")
    cheap_in_clique = sorted(largest_clique, key=lambda v: node_cost(v))[:5]
    supp.append(f"Cheapest 5 nodes to seed from this clique: {cheap_in_clique}")
    clique_cost = sum(node_cost(v) for v in cheap_in_clique)
    supp.append(f"Cost for 5 seeds: {clique_cost}₽")
supp.append("")

supp.append("---")
supp.append("## 21. Full Campaign Strategy Simulation Results")
supp.append("")
supp.append("Four complete 60-day campaign strategies simulated:")
supp.append("")
supp.append(fmt_table(
    ["Strategy", "Income (₽)", "Cost (₽)", "Profit (₽)"],
    [(name, f"{income:,}", f"{cost:,}", f"{profit:,}")
     for name, income, cost, profit, _ in strategies]
))
supp.append("")
best_name, best_inc, best_cst, best_prf, best_log = best_strat
supp.append(f"**Best strategy: {best_name} with profit = {best_prf:,}₽**")
supp.append("")
supp.append(f"### Day-by-day log for best strategy ({best_name})")
supp.append(fmt_table(
    ["Day", "Seeds", "Seed cost (₽)", "Viral new", "Income (₽)", "Budget after (₽)"],
    [(row['day'],
      len(row['seeds']),
      row['cost'],
      row['viral_new'],
      row['income'],
      f"{row['budget_after']:,}")
     for row in best_log if row['cost'] > 0 or row['viral_new'] > 0 or row['day'] < 5]
))
supp.append("")

supp.append("---")
supp.append("## 22. Profitability Analysis")
supp.append("")
supp.append("Break-even equation per seed: `viral_from_seed * 50 > 300 * degree(seed)`")
supp.append("=> `viral_from_seed > 6 * degree(seed)`")
supp.append("")
supp.append("### Minimum viral users needed to break even")
supp.append(fmt_table(
    ["Seed degree", "Cost (₽)", "Min viral needed", "Probability (qualitative)"],
    [(d, SEED_COST*d, breakeven_per_degree[d],
      "HIGH" if d <= 5 else "MEDIUM" if d <= 15 else "LOW" if d <= 30 else "VERY LOW")
     for d in [1,2,3,5,8,10,15,20,30,50,100,150,200]]
))
supp.append("")
supp.append("### Conclusion: Viable seeding targets")
supp.append("- **Degree 1-5 nodes:** Need only 6-30 viral users to break even — very achievable if well-placed")
supp.append("- **Degree 6-15 nodes:** Need 36-90 viral users — possible if at the center of a dense cluster")
supp.append("- **Degree 16-50 nodes:** Need 96-300 viral users — hard to achieve with single seed; use only as group catalyst")
supp.append("- **Degree 50+ nodes:** Break-even requires 300+ viral users — only worth it as a cascade amplifier")
supp.append("")
if profitable_nodes:
    supp.append("### Profitably-simulated seeds (from single-node cascade simulations)")
    supp.append(fmt_table(
        ["Node", "Degree", "Cost (₽)", "Viral", "Income (₽)", "Profit (₽)"],
        [(v, cascade_data[str(v)]['degree'], cascade_data[str(v)]['cost'],
          cascade_data[str(v)]['viral'], cascade_data[str(v)]['income'],
          cascade_data[str(v)]['profit'])
         for v, _ in profitable_nodes[:20]]
    ))
supp.append("")

supp.append("---")
supp.append("## 23. Agent Implementation Guide")
supp.append("")
supp.append("### Recommended Day-0 Algorithm (pseudocode)")
supp.append("```python")
supp.append("# 1. Score every affordable node")
supp.append("for v in nodes:")
supp.append("    if node_cost(v) > budget: continue")
supp.append("    score[v] = ci_roi[v] * 0.5 + pr_roi[v] * 0.3 + (core_number[v]/max_core) * 0.2")
supp.append("")
supp.append("# 2. Greedy pack (knapsack-style)")
supp.append("seeds = []")
supp.append("for v in sorted(nodes, key=lambda v: -score[v]):")
supp.append("    if budget >= node_cost(v) and len(seeds) < 10:")
supp.append("        seeds.append(v); budget -= node_cost(v)")
supp.append("")
supp.append("# 3. After each day: find tipping-zone nodes")
supp.append("for v in non_active_nodes:")
supp.append("    frac = active_neighbors(v) / degree(v)")
supp.append("    if frac >= 0.5 * THRESHOLD:")
supp.append("        add_to_priority_queue(v, priority=frac)")
supp.append("")
supp.append("# 4. Next day: seed tipping-zone first, then fill with CI-ROI")
supp.append("```")
supp.append("")
supp.append("### Data files for agents")
supp.append("- `deep_analysis_data.json` — node scores, community analysis, top lists")
supp.append("- `DEEP_GRAPH_ANALYSIS.md` — this file (human + agent readable)")
supp.append("- `PLAN.md` — strategy plan with budget timing")
supp.append("- `RESEARCH.md` — academic background on IM algorithms")
supp.append("")

with open(OUT_MD, "a", encoding="utf-8") as f:
    f.write("\n".join(supp))

# Update JSON with new data
existing['cascade_dynamics'] = {
    'threshold_table': {str(d): math.ceil(THRESHOLD*d) for d in range(1, 50)},
    'easy_target_counts': {
        'need_1_nbr': sum(1 for v in nodes if math.ceil(THRESHOLD*degrees[v])==1),
        'need_2_nbr': sum(1 for v in nodes if math.ceil(THRESHOLD*degrees[v])==2),
        'need_3_nbr': sum(1 for v in nodes if math.ceil(THRESHOLD*degrees[v])==3),
        'need_8plus': sum(1 for v in nodes if math.ceil(THRESHOLD*degrees[v])>=8),
    },
}
existing['collective_influence'] = {
    str(v): {'ci': ci_scores[v], 'ci_roi': ci_roi[v]} for v in top_ci[:50]
}
existing['pagerank'] = {
    str(v): {'pr': pr[v], 'pr_roi': pr_roi[v]} for v in top_pr[:50]
}
existing['day0_strategies'] = {
    'ci_roi_greedy': {'seeds': day0_seeds_coi, 'cost': day0_cost, 'viral': len(day0_viral), 'profit': day0_profit},
    'kcore_cheap':   {'seeds': day0_seeds_kcore, 'cost': day0_cost_k, 'viral': len(day0_viral_k), 'profit': day0_profit_k},
    'comm_cheap':    {'seeds': day0_seeds_comm, 'cost': day0_cost_c, 'viral': len(day0_viral_c), 'profit': day0_profit_c},
    'best_pair':     {'seeds': list(best_pair) if best_pair else [], 'viral': best_pair_viral, 'profit': best_pair_profit},
}
existing['campaign_simulations'] = {
    s[0]: {'income': s[1], 'cost': s[2], 'profit': s[3]} for s in strategies
}
existing['tipping_zone'] = tipping_nodes[:30]
existing['degree_1_leaves'] = {
    'total': len(low_degree_nodes[1]),
    'top_parents': [(v, degrees[v], node_cost(v), len(leaves_per_parent[v]))
                    for v in top_leaf_parents[:20]],
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(existing, f, indent=2)

print(f"Updated {OUT_MD}")
print(f"Updated {OUT_JSON}")
print("\n=== SUPPLEMENTARY SUMMARY ===")
print(f"Nodes needing 1 infected neighbor: {sum(1 for v in nodes if math.ceil(THRESHOLD*degrees[v])==1)}")
print(f"Degree-1 leaf nodes: {len(low_degree_nodes[1])}")
print(f"Best CI-ROI day0: profit={day0_profit}")
print(f"Best full campaign: {best_strat[0]} profit={best_strat[3]:,}")
print("Done.")
