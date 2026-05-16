# Solution Report — Network Influence Profit Challenge

> **This is the authoritative current state.** Other analysis files contain historical data
> from earlier iterations (90K strategy). This file reflects the actual 134K submission.

**Current Score: 134,150 rubles**  
**Leaderboard #1: 140,750 rubles**  
**Gap: ~6,600 rubles**

---

## What Worked

### 1. Graph Analysis (pre-solution)
- Identified 15 connected components (main WCC: 3,732 nodes, 94.4%)
- Community detection (Louvain): 39 communities, modularity 0.85
- Found 33 **gateway nodes** with positive single-seed profit
- Key discovery: **node 1304** (deg=37, cost=11,100) triggers 862 viral users → 32,000₽ profit alone

### 2. Greedy Marginal Gain Selection
- Simulated LT cascade (θ=0.18) for every node from empty graph
- Greedy selection by maximum marginal profit per step
- 7 core seeds: 1304, 3057, 3775, 2263, 167, 2788, 154
- These cover communities 8, 10, 32, 28, 3, 14

### 3. Multi-Seed Community Search
- For uncovered communities, tried combos up to 6 nodes
- Found profitable entries for comm 19 (nodes 3143+3428) and comm 27 (node 1992)

### 4. Near-Threshold Exploitation
- Full sweep of all nodes for marginal profit given current cascade
- Found nodes 1084 and 27 — these trigger cascades ONLY in context of existing active nodes

### 5. Optimal Day-0 Scheduling
- Enumerated all 2^12 subsets to find best day-0 combo within 10k budget
- Best day-0: [3057, 2263, 167, 2788, 154, 3143] cost=9,900₽
- Impact-based scheduling for remaining seeds (by cascade size, not cost)
- ALL cascades complete within 60 days → theory profit = actual profit

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Total seeds | 12 |
| Total seed cost | 35,100₽ |
| Total active nodes | 3,397 / 3,953 (85.9%) |
| Viral users (non-seeded) | 3,385 |
| Income | 169,250₽ |
| **Net Profit** | **134,150₽** |

---

## What Doesn't Work (Dead Ends)

### Uncovered Communities (556 inactive nodes)
- **Comm 25** (237 nodes): Mean degree 142, median 157. Nodes need 26+ active neighbors to trigger. Seeding any node costs 40,000-70,000₽ for only 20 viral. **Impossible to profit.**
- **Comm 16** (180 nodes): Completely isolated (0 external edges). Best 5-node combo yields only 2 viral at cost 1,500₽. Cascades die after 1-2 hops. **Structurally unseedable.**
- **Comm 26** (73 nodes): Mean degree 43, 0 external edges. Same problem as comm 25 at smaller scale.

### Why These Fail
At θ=0.18, a node with degree 142 needs 26 active neighbors. A single seed is 1 active neighbor. You'd need to seed 26 mutually-connected nodes to start a cascade — costing 26 × 300 × 150 = 1.17M₽ for maybe 200 × 50 = 10,000₽ income.

---

## Seed List (for other agents)

```python
# Phase 1: Day 0 (cost ≤ 10,000₽)
DAY0_SEEDS = [3057, 2263, 167, 2788, 154, 3143]  # cost=9,900₽

# Phase 2: Days 1-16 (as budget allows, highest impact first)
LATER_SEEDS = [1304, 3775, 3428, 1992, 1084, 27]  # cost=25,200₽

# Full schedule:
# Day 0: 3057 2263 167 2788 154 3143
# Day 3: 27 (cost 1200)
# Day 6: 3775 (cost 2400)
# Day 8: 3428 (cost 1500)
# Day 11: 1992 (cost 4200)
# Day 12: 1084 (cost 4200)
# Day 16: 1304 (cost 11100) — needs income accumulation!
```

---

## Parameters

```python
BUDGET = 10_000
COST_K = 300        # contract cost = 300 × degree
INCOME = 50         # income per viral user
MAX_PER_DAY = 10
CAMPAIGN_DURATION = 60
THRESHOLD = 0.18    # LT model threshold
```

---

## Ideas to Reach 140k+ (untested)

1. **Unified greedy over ALL nodes** (not just positive-single-profit): may find nodes profitable only through synergy with existing cascade
2. **Alternate greedy metrics**: try ROI (profit/cost) instead of absolute profit — might select cheaper seeds leaving budget for extras
3. **Exact simulation of cascade timing**: some nodes might trigger faster cascades if seeded at specific days
4. **Multi-seed coordinated attack**: seed 3+ nodes simultaneously in dense clusters of uncovered communities
5. **Alternative community detection**: different Louvain seeds or algorithms might reveal different gateway nodes
6. **Genetic algorithm / simulated annealing**: random search over seed sets
