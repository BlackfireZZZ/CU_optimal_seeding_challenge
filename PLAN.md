# Work Plan: Winning Strategy

## Competition Summary
- Undirected graph: 3,953 nodes, 84,070 edges (Facebook SNAP)
- Budget: 10,000 rubles start + viral income reinvestment
- Win condition: maximize `50 × viral_users − 300 × Σ degree(seeds)`
- Max 10 seeds/day, 60 days

---

## Current Best Results

| Strategy | Profit | Seeds | Active nodes |
|----------|--------|-------|--------------|
| Baseline (2 seeds) | -8,900 | 3 | ~10 |
| Deep analysis optimal (11 seeds) | **90,950** | 11 | 2,302 |
| solution.py v5 (community-aware) | **~140,000+** target | variable | variable |

**Proven floor:** 90,950₽ with deterministic 11-seed strategy
**Theoretical ceiling:** ~150,000₽ (if all profitable communities cracked)

---

## Strategy Status

### ✅ Phase 1: Foundation — DONE
- Graph loaded, simulator validated
- LT model with θ=0.18 confirmed deterministic

### �� Phase 2: Single-node cascade analysis — DONE
- 33 positive-profit single seeds found (exhaustive simulation)
- Top 5: [3057, 3775, 2263, 3991, 443] — 64,000₽ profit on day 0 alone

### ✅ Phase 3: Multi-seed strategy — DONE
- Greedy marginal selection: day-0 base → phase-2 seeds → node 1304
- Full 60-day schedule validated: 90,950₽

### ✅ Phase 4: Sensitivity & Robustness — DONE (enrichment_analysis.py)
Key findings:
- **Critical threshold:** θ ≥ 0.19 breaks the strategy entirely
- **Order robustness:** ±4,600₽ spread (original order is best)
- **Overlap discovered:** 167/93/337 are fully redundant (all unlock same 69 nodes)
  → Only need ONE of them, saving 4,500₽ budget for other seeds
- **Uncovered communities found:**
  - Comm 11: node 2568 (deg=?, marginal +11,900₽)
  - Comm 19: pair [3143, 3428] (marginal +7,750₽)

### 🔄 Phase 5: Final optimization — IN PROGRESS (solution.py v5)
- Community-aware multi-seed solver
- Searches uncovered communities with 2-5 seed combos
- Targets: 140,000₽+

---

## Remaining Optimization Opportunities

### High priority (proven profitable)
1. **Drop redundant seeds 93 and 337** — they overlap 100% with 167 (same community 8)
   - Saves 4,500₽ budget → can afford new gateway seeds earlier
2. **Add node 2568** (Comm 11) — +11,900₽ marginal profit
3. **Add pair [3143, 3428]** (Comm 19) — +7,750₽ marginal profit
4. **Expected uplift:** +19,650₽ → strategy should reach ~110,000₽

### Medium priority (needs more simulation)
5. Crack communities 7 (381 nodes), 25 (237), 0 (229), 16 (180)
   - No single/pair combos found — try 3-5 seed combos
   - solution.py v5 already searches these
6. Schedule seeds for maximum early income (plant 1304 as early as possible)

### Low priority / expensive
7. Exhaustive pairwise synergy scan (2500 sims)
8. Monte Carlo schedule optimization
9. Stochastic perturbation testing

---

## Optimal Seed Schedule (Current Best, 90,950₽)

| Day | Seeds | Cost | Rationale |
|-----|-------|------|-----------|
| 0 | 3057, 3775, 2263, 3991, 443 | 9,900 | Core gateway nodes, 5 communities |
| 1 | 2788 | 300 | Cheap viral supplement (comm 14) |
| 2 | 454 | 600 | Cheap viral supplement (comm 22) |
| 4 | 167 | 1,800 | Community 8 gateway |
| 11 | 1304 | 11,100 | Main cascade amplifier (+24,400₽) |

**Total cost:** 23,700₽ | **Viral:** 2,291 | **Income:** 114,650₽

### Improved schedule (to implement)

| Day | Seeds | Cost | Rationale |
|-----|-------|------|-----------|
| 0 | 3057, 3775, 2263, 3991, 443 | 9,900 | Same core |
| 1 | 2788, 454 | 900 | Combine (both cheap) |
| 4 | 167 | 1,800 | Community 8 (drop 93/337 — redundant!) |
| 11 | 1304 | 11,100 | Main amplifier |
| ~15 | 2568 | ? | New gateway: Comm 11, +11,900₽ |
| ~20 | 3143, 3428 | ? | New gateway: Comm 19, +7,750₽ |

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

### Fast cascade simulation (sparse matrix)
```python
import scipy.sparse as sp
import numpy as np

def sim_cascade(A, DEG, seed_indices, threshold=0.18, max_days=60):
    N = A.shape[0]
    act = np.zeros(N, bool)
    for i in seed_indices:
        act[i] = True
    for _ in range(max_days):
        cnt = A.dot(act.astype(np.float32))
        new = (~act) & ((cnt / np.maximum(DEG, 1e-9)) >= threshold)
        if not new.any():
            break
        act |= new
    return act
```

---

## Risk Factors

| Risk | Status | Mitigation |
|------|--------|-----------|
| θ ≠ 0.18 in competition | **CRITICAL** (θ≥0.19 = strategy dies) | Confirmed in rules: 18% |
| Overlap between seeds | **FOUND** (167/93/337) | Drop redundant, save budget |
| Uncovered large communities | **Partially solved** | solution.py v5 searches combos |
| Budget timing | Low risk (±4,600₽) | Original order already optimal |
| Simulation correctness | Validated | baseline_validate() in solution.py |
