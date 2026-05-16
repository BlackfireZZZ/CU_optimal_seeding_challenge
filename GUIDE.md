# CU Optimal Seeding Challenge: technical guide

Latest score note: the fresh `fenix` branch reached `140500` with a cheaper constructor-bundle schedule. See `FENIX_LATEST_ANALYSIS.md` for the current best result and comparison against the earlier `130900` candidate.

This guide documents the repository, the competition model, and the optimization
techniques used in the extended local research pipeline. It is written as a
practical handoff: what each idea means, why it helps on this graph, how to
validate it, and where the current best strategy came from.

## 1. Problem in one paragraph

We have an undirected social graph from the SNAP Facebook dataset. Each day we
may sign advertising contracts with up to 10 people. A contracted person becomes
active immediately. Other people become active when at least 18% of their
neighbors are already active. Contract cost is proportional to degree:

```text
cost(node) = 300 * degree(node)
```

The task is to maximize campaign profit over 60 days under the cash constraint.
The starting cash is 10,000 rubles, and viral revenue can be reinvested into
later contracts.

## 2. Repository map

The original repository contains the baseline analysis and problem framing:

- `COMPETITION.md` describes the official rules, budget, scoring, and submission format.
- `RESEARCH.md` surveys influence maximization methods that are relevant here.
- `PLAN.md` gives an implementation roadmap for a competitive solver.
- `GRAPH_ANALYSIS.md` and `DEEP_GRAPH_ANALYSIS.md` summarize graph structure.
- `Marketing_Campaign.ipynb` contains the baseline notebook.
- `analyze_graph.py`, `deep_graph_analysis.py`, and `cascade_deep_analysis.py` perform exploratory graph analysis.
- `data/marketing_edges.txt` is the graph edge list.
- `data/sample_submission.csv` is the required output format.

The extended local pipeline that produced the strongest result was organized as
a small Python package plus scripts:

- `network_influence_profit/graph.py`: graph loading, degree lookup, components, clustering, structural-hole score.
- `network_influence_profit/simulation.py`: campaign rules, schedule validation, threshold simulation, submission parsing/writing.
- `network_influence_profit/heuristics.py`: node feature ranking, marginal gain, greedy construction.
- `network_influence_profit/centrality.py`: degree, closeness, betweenness, eigenvector candidate signals.
- `network_influence_profit/deficit.py`: threshold-deficit features for near-activation targeting.
- `network_influence_profit/pair_scoring.py`: single-node and pair-unit scoring, synergy search, local refinement.
- `network_influence_profit/optimize.py`: beam search and simulated annealing.
- `scripts/run_optuna.py`: hyperparameter tuning over candidate generation and optimizer settings.
- `scripts/score_pairs_and_build_strategy.py`: pair/single unit experiments.
- `scripts/sweep_central_edge_strategies.py`: centrality and edge-pair sweeps.
- `scripts/reinvest_greedy_solver.py`: cash-flow-aware greedy solver.
- `scripts/reinvest_deficit_solver.py`: reinvestment solver driven by threshold deficits.
- `scripts/reinvest_anneal.py`: simulated annealing over reinvestment schedules.

## 3. Rule interpretation and the important scoring caveat

There are two slightly different scoring interpretations in the material:

1. `COMPETITION.md` says revenue is paid for virally acquired users, not directly
   contracted users.
2. The extended local simulator used in the final experiments computes:

```text
profit = 50 * active_count - total_contract_cost
```

where `active_count` includes all active nodes. This was based on the local team
note wording "revenue per involved person".

Before final leaderboard submission, this should be verified against the
authenticated competition statement or public score. If the judge excludes
contracted seeds from revenue, the simulator should subtract the unique
contracted nodes from the revenue count:

```text
profit = 50 * (active_count - contracted_count) - total_contract_cost
```

The optimization techniques remain useful either way, but exact local profit
numbers change.

## 4. Graph facts that drive the strategy

The graph has:

- 3,953 observed nodes.
- 84,070 undirected edges.
- 15 connected components.
- One giant component of 3,732 nodes.
- Degree range from 1 to 293.
- Average local clustering around 0.544.

These facts matter:

- Influence cannot cross connected components, so components can be scored independently.
- High-degree nodes are often too expensive at the start because cost grows as `300 * degree`.
- Dense neighborhoods create threshold cascades once enough local neighbors are active.
- Degree-1 and low-degree nodes are cheap, but their value depends on whether they unlock larger neighborhoods.
- Bridge-like nodes with low local clustering can expose several different local pockets.

## 5. Core simulator

The simulator is the source of truth for every strategy. A schedule is a mapping:

```text
day -> [contracted_node_ids]
```

Validation checks:

- day is in `[0, 59]`;
- no more than 10 contracts per day;
- every node exists in the graph;
- a node is not contracted twice;
- spending is feasible under the chosen funding model.

The threshold is:

```text
threshold(node) = max(1, ceil(0.18 * degree(node)))
```

Two propagation modes were supported:

- `one_step`: after daily contracts, every inactive node gets one activation check.
- `cascade`: after daily contracts, activation repeats to a fixed point on the same day.

The strongest local results came from `one_step` propagation plus reinvestment.

## 6. Fixed-budget baseline

The first solver assumed a hard total budget of 10,000. This is useful as a
baseline because it finds a compact initial seed set.

Best fixed-budget local schedule:

```text
day 0: 443 3057 3775 3991 2263
```

Local result under the active-count scoring interpretation:

```text
profit:       64250
active_count: 1483
revenue:      74150
cost:         9900
```

Why it works:

- Node `3057` is a strong trigger.
- Node `3775` is not outstanding alone, but has large synergy with `3057`.
- Cheap helper nodes such as `443`, `3991`, and `2263` tip nearby thresholds without consuming much budget.

## 7. Cash-flow reinvestment

The official budget is not just a static cap. You start with 10,000 cash, then
revenue from activations can fund later contracts. The reinvestment model tracks:

```text
cash_start = 10000
cash_after_contracts(day) >= 0
cash += 50 * new_activations_that_day
```

This changes the search problem. A contract can be bad in isolation but good if
it accelerates cash generation or unlocks a later expensive seed.

Best reinvestment schedule found locally:

```text
day 0:  3057 2263
day 2:  93
day 4:  34
day 15: 1398 3791
day 20: 154
day 21: 2163
day 25: 2745
day 30: 3939
day 40: 27
```

Local result under the active-count scoring interpretation:

```text
profit:       120650
active_count: 3397
revenue:      169850
cost:         49200
cash_end:     130650
contracts:    10
```

Contract costs:

```text
3057: degree 16, cost  4800
2263: degree  6, cost  1800
93:   degree  7, cost  2100
34:   degree  1, cost   300
1398: degree 43, cost 12900
3791: degree  1, cost   300
154:  degree  1, cost   300
2163: degree 30, cost  9000
2745: degree 32, cost  9600
3939: degree  3, cost   900
27:   degree 24, cost  7200
```

## 8. Technique: structural-hole scoring

Local clustering tells us how connected a node's neighbors are to one another.
If a node has low clustering, its neighbors are less redundant. It may bridge
separate pockets.

The structural-hole signal is:

```text
structural_hole(node) = 1 - local_clustering(node)
```

Why this helps:

- high-degree clique nodes can be expensive and redundant;
- bridge nodes can touch different neighborhoods;
- low-clustering cheap nodes are often good exploration candidates.

Structural-hole score was never used alone. It was mixed with simulated profit,
degree, cost, and threshold leverage.

## 9. Technique: threshold deficits

The key threshold question is not "how central is this node?", but:

```text
How many inactive nodes become active, or almost active, if we contract this node?
```

For each inactive target:

```text
remaining_need(target) =
    threshold(target) - active_neighbor_count(target)
```

Candidate nodes were rewarded for:

- immediate targets with `remaining_need <= 0` after the contract;
- one-short targets with `remaining_need == 1`;
- two-short targets with `remaining_need == 2`;
- large inactive frontiers;
- useful centrality signals;
- low contract cost.

This is more competition-specific than generic centrality because it directly
models the 18% activation rule.

## 10. Technique: pair scoring and synergy

Single-node ranking misses combinations. Some nodes are mediocre alone but
excellent together because they jointly satisfy thresholds in the same area.

The pair scorer evaluated same-component pairs and computed:

```text
synergy_vs_sum  = pair_profit - left_single_profit - right_single_profit
synergy_vs_best = pair_profit - max(left_single_profit, right_single_profit)
```

The important discovered pair:

```text
3057 + 3775
pair profit:     52500
pair active:     1194
synergy_vs_sum:  21900
synergy_vs_best: 22250
```

That result explains why `3775` enters the fixed-budget strategy even though it
does not dominate as a single seed.

## 11. Technique: candidate pools

Full search over all nodes and all days is too large. The pipeline built
candidate pools from several complementary rankings:

- best single-node simulated profit;
- best single-node ROI;
- high degree;
- centrality scores;
- structural-hole score;
- deficit-based near-threshold score;
- nodes appearing in strong pairs;
- unresolved nodes near current active frontiers.

This keeps search wide enough to find non-obvious seeds while avoiding a slow
all-nodes brute force loop at every step.

## 12. Technique: greedy construction

The greedy solvers repeatedly choose the next contract with the best estimated
value under the current active set and current cash.

The fast score has four main parts:

```text
score =
    immediate_activation_value
  + almost_ready_value
  + structural_bonus
  - cost_penalty
```

The reinvestment greedy solver adds cash-flow awareness:

- do not buy a node if cash would go negative;
- allow neutral or mildly negative local moves when they improve liquidity or unlock later growth;
- schedule expensive nodes only after enough activation revenue exists.

## 13. Technique: beam search

Greedy keeps only one path. Beam search keeps the best `B` partial schedules.
This helps when a locally second-best seed is part of a much better later
combination.

Practical configuration:

```text
candidate_pool_size: 200-800
beam_width:          20-100
max_contracts:       limited by budget or experiment setting
```

Beam search is useful for early fixed-budget experiments, but it gets expensive
when reinvestment and day placement both matter.

## 14. Technique: simulated annealing

Simulated annealing was used as a finisher on top of greedy/manual schedules.
It mutates a schedule and sometimes accepts worse moves early so it can escape
local optima.

Mutation types:

- add a candidate node;
- remove a node;
- swap one contracted node for another;
- move a node to another day;
- shift a block of days earlier or later.

Acceptance rule:

```text
accept if delta_profit >= 0
otherwise accept with probability exp(delta_profit / temperature)
```

The temperature cools geometrically from a high value to a low value. High
temperature explores; low temperature polishes.

The best reinvestment branch came from greedy/manual setup followed by
annealing and a late-add refinement sweep.

## 15. Technique: Optuna hyperparameter search

Optuna was used to tune:

- optimizer type;
- propagation mode;
- candidate source;
- candidate pool size;
- max contract count;
- annealing iterations;
- annealing temperatures;
- beam width;
- heuristic weights.

Best fixed-budget trial:

```json
{
  "method": "anneal",
  "propagation": "one_step",
  "candidate_source": "mixed",
  "candidate_pool_size": 110,
  "max_contracts": 8,
  "iterations": 600,
  "seed": 245861,
  "start_contracts": 3,
  "start_temperature": 26.288344423897332,
  "end_temperature": 0.16414901438658727,
  "stop_when_no_positive_gain": false,
  "beam_width": 6
}
```

Optuna did not replace the simulator; it only tuned the strategy generator that
the simulator evaluated.

## 16. Validation checklist

Before trusting any submission:

1. Parse the submission CSV.
2. Validate day range and max 10 contracts per day.
3. Validate no duplicate contracted nodes.
4. Validate every contracted node exists in the graph.
5. Re-simulate the campaign with the intended propagation mode.
6. Verify cash never goes negative in reinvestment mode.
7. Recompute profit, active count, revenue, cost, and final cash.
8. Compare local assumptions with the public leaderboard behavior.

Expected validation for the best local reinvestment schedule:

```text
CampaignRules(propagation="one_step", funding_mode="reinvest")
profit=120650
active_count=3397
revenue=169850
cost=49200
cash_end=130650
```

## 17. Reproduction commands from the extended pipeline

Fixed-budget baseline:

```bash
python3 scripts/make_submission.py \
  --method greedy \
  --output submissions/greedy.csv
```

Optuna tuning:

```bash
python3 scripts/run_optuna.py
```

Pair-unit scoring:

```bash
python3 scripts/score_pairs_and_build_strategy.py \
  --propagation one_step
```

Reinvestment greedy:

```bash
python3 scripts/reinvest_greedy_solver.py \
  --propagation one_step \
  --pool-limit 1800 \
  --eval-limit 260 \
  --max-steps 140 \
  --liquidity-weight 0.25 \
  --active-weight 6.0 \
  --min-delta-profit 0 \
  --output-prefix reinvest_greedy
```

Reinvestment annealing:

```bash
python3 scripts/reinvest_anneal.py \
  --iterations 3500 \
  --candidate-limit 1600 \
  --seed 303 \
  --start-temperature 9000 \
  --end-temperature 5 \
  --output submissions/reinvest_anneal_seed303.csv
```

## 18. Practical recommendations

- Treat the simulator as the judge. Centrality is only a candidate generator.
- Use reinvestment, because the best strategy spends far more than 10,000 total
  but remains cash-feasible.
- Search for pairs and bundles, not just individual seeds.
- Keep day placement in the optimization; the same node can be good or bad
  depending on whether enough cash exists and whether the cascade has matured.
- Preserve experiment logs with parameters, random seed, profit, active count,
  cost, and output file.
- Re-run the final schedule under the exact official scoring rule before
  submitting.

## 19. Current best known local schedule

Use this as the canonical local candidate until the scoring caveat is resolved:

```csv
day,node_ids
0,3057 2263
1,-1
2,93
3,-1
4,34
5,-1
6,-1
7,-1
8,-1
9,-1
10,-1
11,-1
12,-1
13,-1
14,-1
15,1398 3791
16,-1
17,-1
18,-1
19,-1
20,154
21,2163
22,-1
23,-1
24,-1
25,2745
26,-1
27,-1
28,-1
29,-1
30,3939
31,-1
32,-1
33,-1
34,-1
35,-1
36,-1
37,-1
38,-1
39,-1
40,27
41,-1
42,-1
43,-1
44,-1
45,-1
46,-1
47,-1
48,-1
49,-1
50,-1
51,-1
52,-1
53,-1
54,-1
55,-1
56,-1
57,-1
58,-1
59,-1
```
