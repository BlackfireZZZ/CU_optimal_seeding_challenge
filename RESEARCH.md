# Research: Influence Maximization Techniques

## Problem Formulation

This is an **Influence Maximization (IM)** problem with budget constraints and a time dimension.

Given:
- Undirected graph G(V, E)
- Threshold model: node activates if ≥18% neighbors are active
- Budget: 10,000 initial + accumulated viral income
- Max 10 seeds per day, 60 days total
- Profit = 50 × |viral_users| − 300 × Σ degree(seed_i)

Classic IM (Kempe et al. 2003) is NP-hard; greedy with (1-1/e) approximation guarantee under IC/LT models.

---

## Cascade Model

This competition uses the **Linear Threshold (LT) model** with a fixed threshold θ = 0.18.

### LT Model Properties:
- Deterministic (given seeds, spread is fully determined)
- Monotone: more seeds → at least as much spread
- Submodular: marginal gain decreasing with set size

### Cascade Dynamics:
- Day 0: seeds become active
- Day 1: neighbors with ≥18% active neighbors become active
- Day 2: further propagation
- Continues until no new activations (fixpoint)

### Key Insight for θ=0.18:
- A node v with degree d needs ⌈0.18d⌉ = max(1, ⌈0.18d⌉) active neighbors
- **Degree-1 nodes go viral immediately** when their single neighbor is seeded!
- Dense cliques/clusters amplify cascades internally

---

## Node Centrality Metrics Relevant to This Problem

### 1. Degree Centrality
- **Formula:** deg(v) / (n-1)
- **Relevance:** Direct measure of spread potential; also directly determines cost
- **Issue:** High degree = high cost; not always best ROI

### 2. Weighted Degree / Cost-effectiveness
- **Formula:** expected_cascade(v) / cost(v)
- **Relevance:** ROI metric; maximize cascade per ruble spent
- Best metric for initial seed selection

### 3. Betweenness Centrality
- **Formula:** Σ σ(s,t|v)/σ(s,t) over all pairs
- **Relevance:** Identifies bridge nodes that connect communities
- Seeding bridges can unlock entire new clusters
- Node 1577 has highest betweenness (0.0267) — critical bridge

### 4. Closeness Centrality
- **Formula:** (n-1) / Σ d(v,u)
- **Relevance:** Nodes close to all others spread influence faster
- Useful for time-constrained campaigns (60 days)

### 5. Eigenvector / PageRank Centrality
- **Relevance:** Nodes connected to other high-degree nodes
- Captures second-order influence (friends of hubs)

### 6. K-Core Decomposition
- **Definition:** Maximal subgraph where every node has degree ≥ k
- **Relevance:** High k-core nodes are deeply embedded in the network core
- Seeding k-core maximizes internal cascades
- **Strong predictor** of influence spread in threshold models

### 7. Collective Influence (CI)
- **Formula:** CI_l(i) = (deg(i)-1) × Σ_{j∈∂Ball(i,l)} (deg(j)-1)
- **Relevance:** Considers cascade potential beyond immediate neighbors
- Optimal for minimizing immunization / maximizing seeding

### 8. Discount Heuristic (Chen et al. 2009)
- Avoid seeding already-influenced neighbors
- Select node v greedily, discount neighbors' "remaining threshold"

---

## Seeding Strategy Techniques

### Baseline Approaches

#### 1. Random Seeding
- Profit: negligible
- Upper bound on cascade: low

#### 2. High-Degree Greedy
- Pick top-degree nodes first
- Problem: most high-degree nodes cost 87,000+ rubles; initial budget only 10,000

#### 3. Low-Cost + High-Cascade Potential
- Pick nodes with high degree relative to cost
- Since cost = 300 × degree, all nodes have the same cost-per-degree ratio
- BUT: marginal cascade is nonlinear → cheaper nodes may still be better ROI

### Advanced Approaches

#### 4. Simulation-Based Greedy (Best known approach)
```python
# CELF algorithm (Cost-Effective Lazy Forward)
# Kempe 2003 + Leskovec 2007
for round in range(max_seeds):
    best_node = argmax[marginal_gain(v, current_seeds) for v in V]
    seeds.add(best_node)
```
- Runs full cascade simulation for each candidate
- Uses lazy evaluation to avoid re-computing all candidates each round
- ~700x speedup vs naive greedy

#### 5. Community-Aware Seeding
```
1. Detect communities (Louvain, Infomap, Label Propagation)
2. For each community: find internal hub (highest internal degree)
3. Seed community hubs first → internal cascade fills community
4. Then seed inter-community bridges
```

#### 6. k-Core Seeding
```python
core_numbers = nx.core_number(G)
# Sort by core number descending
# For same core: sort by degree / cost
```

#### 7. Two-Phase Strategy
- **Phase 1 (Days 0-5):** Seed low-cost nodes near densely connected clusters to build initial cascade momentum
- **Phase 2 (Days 6-60):** Use accumulated viral revenue to seed more expensive bridge nodes and unlock new regions

#### 8. Budget-Aware Dynamic Programming
- State: (day, budget, set_of_infected_nodes)
- Too large for exact DP, but approx with Monte Carlo
- Key: reinvest viral income immediately

---

## Key Formula Insights

### Break-even Analysis
A seeded node with degree d directly costs 300d rubles.
To break even via viral spread from that seed:
- Need at least 300d/50 = 6d viral users attributable to that seed
- High-degree node (d=50): needs 300 viral users triggered
- Low-degree node (d=5): needs 30 viral users triggered

### Cascade Amplification
If we seed a node with degree d, and each of its neighbors subsequently goes viral and has degree d_n:
- First wave: d potential viral users (neighbors)
- Second wave: Σ (d_n - 1) potential users from neighbors' neighbors
- Exponential growth if neighbors have high degree

### Threshold Exploitation
To trigger a node v with degree d_v, we need 0.18 × d_v neighbors infected.
- For a clique of size k: seeding 0.18k members triggers all others
- Cost of seeding 0.18k clique members ≈ 300 × 0.18k × d_avg_in_clique
- Gain: (0.82k) × 50 viral users

---

## Critical Implementation Notes

### Simulation Correctness (from baseline)
```python
# Status updates are BATCHED (all nodes update simultaneously per day)
# NOT sequential (which would give different results)
for node in non_affected:
    if affected_neighbors / len(neighbors) >= threshold:
        new_affected.append(node)  # collect first
# Then apply all updates
```

### Budget Management
- Income from day t is available starting day t+1
- Initial 10,000 may only allow 1-3 seeds on day 0
- Days 1-5: cascade income feeds next seeds
- By day 10+: accumulated income can fund 10 seeds/day

### The "No Direct Income" Rule
- **Seeds do NOT generate 50 rubles** — only viral users do
- This means pure cost analysis: minimize seed costs, maximize viral cascade

---

## Promising Approaches for Top Score

### Approach A: Greedy Simulation (Recommended)
1. Run full LT cascade simulation
2. Greedily add seed with best marginal gain / cost ratio
3. Respect daily budget and max-10 constraint
4. Use CELF for efficiency

### Approach B: k-Core + Community Decomposition
1. Find k-core decomposition (high-core nodes spread best)
2. Find communities (Louvain)
3. Identify community "seeds" (bridge nodes between communities)
4. Schedule: core seeds first, then bridges on later days with more budget

### Approach C: Threshold-Aware "Tipping Point" Search
1. For each node v, compute how many current seeds cover its neighborhood
2. Find nodes that are "one step from tipping" (need just 1-2 more seeds nearby)
3. Efficiently cascade entire clusters

### Approach D: Genetic Algorithm / Simulated Annealing
1. Encode solution as: for each day, which ≤10 nodes to seed
2. Fitness = simulate_profit(solution)
3. Mutate: swap nodes between days, add/remove seeds
4. Run for many iterations

---

## Literature References

- Kempe, Kleinberg, Tardos (2003) - "Maximizing the Spread of Influence through a Social Network"
- Leskovec et al. (2007) - CELF algorithm, cost-effective influence maximization
- Chen, Wang, Yang (2009) - "Efficient Influence Maximization in Social Networks"
- Morone & Makse (2015) - Collective Influence algorithm
- SNAP Facebook dataset: ego-networks from Facebook
