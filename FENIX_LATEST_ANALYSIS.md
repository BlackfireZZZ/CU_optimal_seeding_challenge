# Latest Fenix Analysis: 140500 Strategy

This note documents the fresh `fenix` update at commit `6ebd433` and compares it with the previous local bests. The important change is not broader reach; it is a much cheaper construction that preserves almost the same cascade.

## Current Best

Canonical file:

- `best_sub.csv`
- `best_sub.report.json`
- mirrored in `secret_ivent/best_sub.csv`

Official-style result:

```text
profit:  140500
income:  169000
costs:   28500
balance: 150500
active:  3396
viral:   3380
seeds:   16
```

Schedule:

```text
day 0:  2263 2678 3432
day 2:  262 205 138
day 4:  34
day 15: 1398 3953 3808
day 20: 154
day 21: 1924 2595
day 25: 3428 3143
day 40: 27
```

## Comparison With Previous Best

Previous `fenix` result:

```text
profit: 130900
income: 169300
costs:  38400
viral:  3386
active: 3396
seeds:  10
```

New `fenix` result:

```text
profit: 140500
income: 169000
costs:  28500
viral:  3380
active: 3396
seeds:  16
```

Delta:

```text
profit: +9600
viral:  -6
cost:   -9900
active: same
```

The new solution accepts six fewer paid viral users but saves 9900 rubles in contract costs, so the net score improves by 9600.

## What Changed

Removed from the 130900 solution:

```text
167, 2568, 2745, 3057, 3775
```

Added:

```text
138, 205, 262, 1924, 2595, 2678, 3143, 3428, 3432, 3808, 3953
```

The old strategy relied on a smaller number of stronger, more expensive triggers:

```text
old cost: 38400
old seeds: 10
```

The new strategy uses more low-cost constructors to reproduce the same large cascade:

```text
new cost: 28500
new seeds: 16
```

## Contract Cost Breakdown

```text
node 2263: degree 6,  cost 1800
node 2678: degree 8,  cost 2400
node 3432: degree 5,  cost 1500
node 262:  degree 3,  cost 900
node 205:  degree 1,  cost 300
node 138:  degree 1,  cost 300
node 34:   degree 2,  cost 600
node 1398: degree 43, cost 12900
node 3953: degree 3,  cost 900
node 3808: degree 1,  cost 300
node 154:  degree 1,  cost 300
node 1924: degree 5,  cost 1500
node 2595: degree 2,  cost 600
node 3428: degree 7,  cost 2100
node 3143: degree 3,  cost 900
node 27:   degree 4,  cost 1200
total: 28500
```

## Why The New Approach Works

The new `constructor_anneal.py` is the key technique. Instead of only swapping individual nodes, it mutates small packs of cheap nodes inside label-propagation communities and inactive components.

Important ideas:

- Build many community partitions using different label-propagation seeds.
- Rank nodes inside each community by internal connectivity, weak-neighbor leverage, and low degree.
- Mutate whole packs, not only individual nodes.
- Remove expensive nodes and replace them with cheap constructor bundles.
- Repair every candidate schedule to keep cash feasibility and avoid already-active buys.
- Anneal over add/remove/replace/move-day operations.

This finds cheaper threshold constructors that single-node replacement misses.

## Local Verification

I validated the fresh schedule locally with the official-style simulator:

```text
submissions/fenix_best_140500.csv: official_profit=140500, viral=3380, cost=28500
submissions/fenix_best_for_now.csv: official_profit=130900, viral=3386, cost=38400
submissions/reinvest_best_final_one_step.csv: official_profit=120050, viral=3385, cost=49200
```

I also checked a full single replacement pass against all 3953 graph nodes:

```text
start profit=140500 viral=3380 cost=28500 balance=150500 candidates=3953
round=0 no improvement
final profit=140500 viral=3380 cost=28500 balance=150500
```

So the current solution is locally stable against one-for-one replacement.

## Frontier Check

After the 140500 schedule:

```text
inactive nodes: 557
inactive components: 23
largest inactive components: 228, 180, 25, 20, 19, 18, 9, 9, 8, 8, 6, 4, 3, 3, 2
```

Quick exact frontier check over cheap/peripheral candidates:

```text
positive single additions: 0
best single-addition delta: -200
```

Best negative additions:

| node | delta | viral_delta | cost |
|---:|---:|---:|---:|
| 90 | -200 | 2 | 300 |
| 145 | -200 | 2 | 300 |
| 749 | -200 | 2 | 300 |
| 775 | -200 | 2 | 300 |

The frontier still has unactivated pockets, but single cheap peripheral buys do not pay for themselves. Further improvement probably needs another constructor-bundle search, not isolated additions.

## Best Next Work

1. Extend `constructor_anneal.py` with a local cost-minimization phase that freezes active spread and tries to remove/replace packs.
2. Add a bundle-level frontier search for the inactive components of sizes 228 and 180.
3. Penalize viral loss less than cost loss when active count stays fixed, because the 140500 jump came from accepting `-6` viral for `-9900` cost.
4. Keep `139450` and `130900` variants as diversity starts for more annealing, not as final submissions.

