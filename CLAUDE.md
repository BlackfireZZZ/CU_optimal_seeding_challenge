# CU Optimal Seeding Challenge

## Quick Context

Network influence profit maximization. Seed nodes in a social graph, cascade spreads via Linear Threshold model, maximize `income - costs`.

**Current score: 134,150₽** (leaderboard #1: 140,750₽, gap ~6,600₽)

## Parameters

```python
GRAPH       = "data/marketing_edges.txt"  # undirected, 3953 nodes, 84070 edges
THRESHOLD   = 0.18       # node activates if ≥18% neighbors active
BUDGET      = 10_000     # initial rubles
COST        = 300 * degree(node)  # per seed
INCOME      = 50         # per viral user (NOT seeds)
MAX_PER_DAY = 10
DAYS        = 60
PROFIT      = sum(viral) * 50 - sum(seed_costs)
```

## Current Best Solution (134,150₽)

12 seeds, 3,397/3,953 nodes active (85.9%), 3,385 viral users.

```python
# Day 0 (budget=10,000₽, cost=9,900₽):
DAY0 = [3057, 2263, 167, 2788, 154, 3143]

# Days 1-16 (funded by cascade income):
LATER = {
    3:  [27],      # cost 1,200₽
    6:  [3775],    # cost 2,400₽
    8:  [3428],    # cost 1,500₽
    11: [1992],    # cost 4,200₽
    12: [1084],    # cost 4,200₽
    16: [1304],    # cost 11,100₽ — main cascade amplifier
}
```

## Dead Ends (proven unprofitable)

- **Comm 25** (237 nodes): mean degree 142, need 26 active neighbors per node. Impossible.
- **Comm 16** (180 nodes): completely isolated, no external edges. Max 2 viral from 5 seeds.
- **Comm 26** (73 nodes): same as 25 at smaller scale.
- **High-degree hubs**: all lose money (cost >> cascade value at this budget).
- **Nodes 93, 337**: 100% overlap with node 167 (same community gateway). Never seed together.

## Key Insights

1. **Only 33 nodes have positive single-seed profit** — rest are unprofitable alone
2. **Gateway nodes** (cheap entry into dense community) beat hub nodes every time
3. **Node 1304** is the linchpin: deg=37, triggers 862 viral users (community 17, 323 nodes)
4. **θ=0.18 is critical**: at θ=0.19 the entire strategy produces -1,700₽
5. **LT model is deterministic**: same seeds → same result, no randomness
6. **556 unreachable nodes** remain in dense isolated clusters (see dead ends above)

## Ideas to Close the 6,600₽ Gap

1. Unified greedy over ALL nodes (not just known positives) — synergy-only seeds
2. Alternative community detection (different Louvain seeds)
3. Exact cascade timing optimization (which day matters for some nodes)
4. Multi-seed coordinated attacks (3-5 nodes) in remaining communities
5. Genetic algorithm / simulated annealing over seed sets

## File Index

| File | Purpose | When to read |
|------|---------|--------------|
| **CLAUDE.md** | This file — entry point | Always (loaded by default) |
| **SOLUTION_REPORT.md** | Current 134K solution details, seed list, dead ends | When building on current solution |
| **COMPETITION.md** | Official rules, submission format, baseline code | When checking constraints |
| **DEEP_GRAPH_ANALYSIS.md** | 20-section reference: communities, cascades, gateways, all 33 profitable seeds | When searching for new seeds or understanding graph structure |
| **SENSITIVITY_ANALYSIS.md** | Threshold sensitivity, seed interactions, robustness | When evaluating strategy changes |
| **deep_analysis_data.json** | Machine-readable: node scores, communities, optimal seeds | When writing code that needs node data |
| **solution.py** | Current solver (v5): full pipeline from graph → submission.csv | When modifying the solver |
| **GRAPH_ANALYSIS.md** | Basic graph stats (superseded by DEEP_GRAPH_ANALYSIS) | Rarely — quick reference only |
| **RESEARCH.md** | Academic IM background (Kempe, CELF, CI) | When exploring new algorithmic approaches |
| **PLAN.md** | Historical work plan | Low priority — mostly done |

## Submission Format

```csv
day,node_ids
0,3057 2263 167 2788 154 3143
1,-1
2,-1
3,27
...
```

File: `submission.csv`, 60 rows (days 0-59), `-1` = no seeds that day.

## Running the Solver

```bash
python solution.py  # loads graph, finds seeds, validates, writes submission.csv
```

Requires: networkx, numpy, scipy, pandas. Graph file at `data/marketing_edges.txt`.
