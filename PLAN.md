# Work Plan: Winning Strategy

## Competition Summary
- Undirected graph: 3,953 nodes, 84,070 edges (Facebook SNAP)
- Budget: 10,000 rubles start + viral income reinvestment
- Win condition: maximize `50 × viral_users − 300 × Σ degree(seeds)`
- Max 10 seeds/day, 60 days

---

## Target Score Analysis

**Upper bound estimate:**
- Total nodes: 3,953
- If ALL nodes go viral (impossible for seeds themselves): 3,953 × 50 = 197,650 rubles income
- Minus seed costs: depends on strategy
- Realistic top score: 100,000–150,000 rubles

**Baseline:** -8,900 rubles (2 seeds on day 0, 1 on day 1)

**Good score target:** 50,000+ rubles

---

## Phase 1: Foundation (Implement Simulator) ✅ done in baseline

The baseline already provides:
- `G = nx.read_edgelist(...)` — undirected graph
- `update_affected(G)` — one-day viral spread step
- `balances(G, steps)` — simulate full campaign and return daily balance
- `balance_graph(G, steps)` — visualize + print profit

**Action:** Use baseline simulator directly. Do NOT reimplement.

---

## Phase 2: Greedy Seeding Algorithm

### Step 2.1: Compute Node Features
```python
# For each node:
features[v] = {
    'degree': G.degree(v),
    'cost': 300 * G.degree(v),
    'core_number': nx.core_number(G)[v],
    'pagerank': nx.pagerank(G)[v],
    'betweenness': nx.betweenness_centrality(G, k=500)[v],  # approximate
    'clustering': nx.clustering(G)[v],
}
```

### Step 2.2: Single-Node Cascade Simulation
For each candidate node, simulate how many viral users it generates if seeded alone:
```python
def simulate_cascade(G, seeds):
    G_copy = G.copy()
    nx.set_node_attributes(G_copy, False, 'status')
    for s in seeds: G_copy.nodes[s]['status'] = True
    total_viral = 0
    for day in range(60):
        new = update_affected(G_copy)
        total_viral += len(new)
        if not new: break
    return total_viral

single_node_cascade = {v: simulate_cascade(G, [v]) for v in G.nodes()}
```

### Step 2.3: ROI Ranking
```python
roi = {v: single_node_cascade[v] * 50 / max(300 * G.degree(v), 1) 
       for v in G.nodes()}
top_roi_nodes = sorted(roi, key=roi.get, reverse=True)
```

---

## Phase 3: Strategy Design

### Strategy A: Pure Greedy (Quick Baseline+)

Day 0: Seed the top-10 affordable nodes (cost ≤ budget/10 each, highest ROI)
Day 1-59: Greedily pick nodes with highest marginal gain from current infected set

**Expected gain:** ~10,000–30,000 profit

### Strategy B: Cluster Cascade (Recommended Primary)

**Key insight:** Facebook network has dense clusters. Seeding a critical mass within a cluster triggers cascade through the whole cluster.

```
1. Run community detection (Louvain algorithm)
2. For each community: 
   a. Find minimum seed set that triggers full community cascade
   b. Compute: community_value = 50 × community_size - seed_costs
3. Rank communities by community_value
4. Schedule: seed cheapest communities first, reinvest income for more
```

**Steps:**
```python
import community as community_louvain  # python-louvain

partition = community_louvain.best_partition(G)
communities = defaultdict(list)
for node, comm_id in partition.items():
    communities[comm_id].append(node)

# For each community, find minimum seed set to cascade
for comm_id, members in communities.items():
    subgraph = G.subgraph(members)
    # Sort by degree within community - seed high-degree members
    seed_order = sorted(members, key=lambda v: G.degree(v), reverse=True)
    # Find tipping point: how many seeds needed to cascade all?
```

### Strategy C: CELF Budget-Aware

Full CELF with budget constraints:
```python
# Lazy forward greedy with marginal gain tracking
heap = [(-simulate_cascade(G, [v]), v) for v in affordable_nodes]
selected = []
budget = 10000

while budget > 0 and heap:
    gain, v = heapq.heappop(heap)
    # Recompute marginal gain given current selected set
    marginal = simulate_cascade(G, selected + [v]) - simulate_cascade(G, selected)
    if marginal cost-effective:
        selected.append(v)
        budget -= cost(v)
```

### Strategy D: Temporal Optimization (Advanced)

The time dimension matters! Nodes activated on day 0 spread for 60 days; nodes activated on day 50 only spread for 10 days.

**Optimal timing:**
- Day 0-2: Seed trigger nodes that start cascades into large clusters (spend full budget)
- Day 3-10: Let viral income accumulate, seed bridge nodes when affordable
- Day 10-30: Seed secondary communities with accumulated income
- Day 30-59: Seed isolated nodes or remaining clusters

---

## Phase 4: Parameter Optimization

### Sensitivity Analysis
- Test different initial seed counts (1 vs 3 vs 10 on day 0)
- Test different ordering (degree-first vs betweenness-first vs k-core-first)
- Tune: which days to seed vs. let cascade run

### Simulation-Based Optimization
```python
from scipy.optimize import differential_evolution

def objective(schedule_params):
    # Decode params to step dict
    profit = simulate_campaign(G, step_dict)
    return -profit  # minimize negative profit

result = differential_evolution(objective, bounds, maxiter=100)
```

---

## Phase 5: Implementation Checklist

- [ ] Load graph correctly as undirected (✅ baseline does this)
- [ ] Implement `simulate_cascade(G, seeds)` for single-node/set ROI
- [ ] Compute all node centralities (degree, k-core, betweenness, pagerank)
- [ ] Community detection (Louvain)
- [ ] Greedy seeding with budget constraints
- [ ] Temporal scheduling optimization
- [ ] Validate submission format (day, node_ids CSV)
- [ ] Test on multiple random seeds to verify simulation
- [ ] Submit and iterate

---

## Key Code Patterns

### Convert strategy dict to submission CSV
```python
import pandas as pd

def strategy_to_csv(steps, filename='submission.csv'):
    rows = []
    for day in range(60):
        if day in steps and steps[day]:
            node_ids = ' '.join(map(str, steps[day]))
        else:
            node_ids = '-1'
        rows.append({'day': day, 'node_ids': node_ids})
    pd.DataFrame(rows).to_csv(filename, index=False)
```

### Fast cascade simulation
```python
def fast_cascade(adj, seeds, threshold=0.18, max_days=60):
    """Uses numpy/sets for fast simulation without full nx copy."""
    n = max(max(adj.keys()), max(seeds)) + 1
    active = set(seeds)
    for _ in range(max_days):
        new_active = set()
        for node, neighbors in adj.items():
            if node in active: continue
            if len(neighbors) == 0: continue
            frac = sum(1 for nb in neighbors if nb in active) / len(neighbors)
            if frac >= threshold:
                new_active.add(node)
        if not new_active: break
        active |= new_active
    return active - set(seeds)  # only viral, not seeds
```

---

## Risk Factors & Mitigations

| Risk | Mitigation |
|------|-----------|
| Overfitting to one seeding strategy | Test multiple strategies, compare profits |
| Cascade doesn't trigger at all | Ensure seeds are in same connected component |
| Budget runs out too fast | Schedule seeds across days, reinvest income |
| Small components missed | Handle each component separately |
| Simulation is slow (3953 nodes × many candidates) | Use fast numpy-based simulation |

---

## Quick Wins

1. **Seed low-degree nodes in dense clusters** — cost 300×3=900 per node, can trigger cascade of dozens
2. **Use all 10 slots per day from day 1 onward** — baseline only uses 2-3 seeds total
3. **Focus on giant component first** — 94.4% of all nodes reachable
4. **Nodes 1577 (bridge), 606, 1077** — betweenness leaders, unlock new regions cheaply

---

## Expected Performance Tiers

| Strategy | Expected Profit |
|----------|----------------|
| Baseline (2 seeds) | ~-8,900 |
| Random 10 seeds/day | ~0–5,000 |
| Degree-based greedy | ~15,000–25,000 |
| k-Core + Community | ~40,000–70,000 |
| CELF + Temporal optimization | ~70,000–120,000 |
| Near-optimal | ~120,000–150,000 |
