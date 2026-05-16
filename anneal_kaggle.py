#!/usr/bin/env python3
"""
Simulated Annealing for Network Influence Profit Maximization.
Designed to run on Kaggle (CPU/GPU, ~9h budget).

Key insights baked in:
- theta=0.18 LT model, deterministic cascade
- 33 known profitable single-seeds (gateway nodes)
- Best known: 139,450rub (13 seeds, cost=29,700rub)
- Dead communities: 25 (237 nodes), 16 (180), 26 (73) -- skip these
- Nodes 167/93/337 are 100% redundant -- only one needed
- Node 1304 (deg=37) triggers 862 viral -- linchpin but expensive
- Income ceiling: ~169,250rub (3,385 viral x 50)
- Budget timing is critical: can't afford expensive seeds early

Strategy: massive parallel restarts with simulated annealing,
aggressive candidate pruning, and multi-resolution search.
"""

import math
import random
import time
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass

# --- Constants ----------------------------------------------------------------
INITIAL_BUDGET = 10_000
COST_PER_NEIGHBOR = 300
INCOME_PER_VIRAL = 50
MAX_CONTRACTS_PER_DAY = 10
DAYS = 60
THRESHOLD = 0.18

# Annealing parameters (tune for Kaggle time budget)
TOTAL_TIME_BUDGET = 8 * 3600  # 8 hours
NUM_RESTARTS = 30
ITERATIONS_PER_RESTART = 500_000  # ~5min per restart at ~1500 iter/s
MAX_BUY_DAY = 52  # don't buy after day 52 (cascade needs time)

# Known best seeds from analysis (starting points for restarts)
KNOWN_BEST_SEEDS = {
    # Fenix best_sub: 139,450rub
    "fenix_best": {0: [2263, 2678, 3432], 2: [167], 4: [34], 15: [1398, 3953, 3808],
                   20: [154], 21: [2568], 25: [3428, 3143], 40: [27]},
    # Variant: day-0 heavy, let cascade fund later seeds
    "day0_heavy": {0: [2263, 2678, 3432, 167, 154], 4: [34], 15: [1398, 3953, 3808],
                   21: [2568], 25: [3428, 3143], 40: [27]},
    # Variant: minimal cost approach
    "minimal": {0: [2263, 2678, 3432], 2: [167], 4: [34],
                15: [3953, 3808], 21: [2568], 25: [3428, 3143]},
    # Original 134K key nodes as starting point (reordered for budget)
    "gateway_core": {0: [3057, 2263, 167, 2788, 154], 3: [27], 6: [3775],
                     8: [3428], 15: [2568], 20: [1304]},
}


@dataclass
class Graph:
    ids: list  # internal_idx -> original_node_id
    index: dict  # original_node_id -> internal_idx
    adj: list   # adjacency lists (internal indices)
    degree: list
    threshold: list  # ceil(0.18 * degree)


def read_graph(path: str) -> Graph:
    raw_edges = []
    nodes = set()
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            a, b = int(parts[0]), int(parts[1])
            if a == b:
                continue
            raw_edges.append((a, b))
            nodes.add(a)
            nodes.add(b)

    ids = sorted(nodes)
    index = {node: i for i, node in enumerate(ids)}
    n = len(ids)
    adj_sets = [set() for _ in range(n)]
    for a, b in raw_edges:
        ia, ib = index[a], index[b]
        adj_sets[ia].add(ib)
        adj_sets[ib].add(ia)

    adj = [sorted(s) for s in adj_sets]
    degree = [len(v) for v in adj]
    threshold = [max(1, math.ceil(THRESHOLD * d)) for d in degree]
    return Graph(ids=ids, index=index, adj=adj, degree=degree, threshold=threshold)


# --- Fast Simulation (optimized for speed) -----------------------------------

def fast_simulate(graph: Graph, strategy: dict, return_active=False):
    """Simulate LT cascade. Returns profit dict. ~0.8ms per call."""
    n = len(graph.ids)
    active = bytearray(n)
    infected_counts = [0] * n
    ready = []
    queued_next = bytearray(n)

    balance = INITIAL_BUDGET
    income = 0
    costs = 0

    def activate(node, next_ready):
        if active[node]:
            return False
        active[node] = 1
        for nb in graph.adj[node]:
            if active[nb]:
                continue
            infected_counts[nb] += 1
            if infected_counts[nb] >= graph.threshold[nb] and not queued_next[nb]:
                queued_next[nb] = 1
                next_ready.append(nb)
        return True

    for day in range(DAYS):
        next_ready = []
        viral = []
        for node in ready:
            if not active[node] and infected_counts[node] >= graph.threshold[node]:
                viral.append(node)

        for node in viral:
            activate(node, next_ready)
        income += INCOME_PER_VIRAL * len(viral)
        balance += INCOME_PER_VIRAL * len(viral)

        for node in strategy.get(day, []):
            cost = COST_PER_NEIGHBOR * graph.degree[node]
            balance -= cost
            costs += cost
            if balance < -1e-9:
                raise ValueError(f"Negative balance day {day}: {balance}")
            activate(node, next_ready)

        for node in ready:
            queued_next[node] = 0
        ready = next_ready

    result = {
        "profit": income - costs,
        "income": income,
        "costs": costs,
        "balance": balance,
        "active": sum(active),
        "viral": income // INCOME_PER_VIRAL,
        "seeds": sum(len(v) for v in strategy.values()),
    }
    if return_active:
        return result, active
    return result


def fast_repair(graph: Graph, strategy: dict) -> dict:
    """Drop invalid seeds (already active, over budget, duplicates)."""
    n = len(graph.ids)
    active = bytearray(n)
    infected_counts = [0] * n
    ready = []
    queued_next = bytearray(n)
    balance = INITIAL_BUDGET
    clean = {}
    used = set()

    def activate(node, next_ready):
        if active[node]:
            return False
        active[node] = 1
        for nb in graph.adj[node]:
            if active[nb]:
                continue
            infected_counts[nb] += 1
            if infected_counts[nb] >= graph.threshold[nb] and not queued_next[nb]:
                queued_next[nb] = 1
                next_ready.append(nb)
        return True

    for day in range(DAYS):
        next_ready = []
        viral = [node for node in ready
                 if not active[node] and infected_counts[node] >= graph.threshold[node]]
        for node in viral:
            activate(node, next_ready)
        balance += INCOME_PER_VIRAL * len(viral)

        day_seeds = []
        for node in strategy.get(day, []):
            cost = COST_PER_NEIGHBOR * graph.degree[node]
            if node in used or active[node] or len(day_seeds) >= MAX_CONTRACTS_PER_DAY:
                continue
            if cost > balance:
                continue
            day_seeds.append(node)
            used.add(node)
            balance -= cost
            activate(node, next_ready)

        if day_seeds:
            clean[day] = day_seeds

        for node in ready:
            queued_next[node] = 0
        ready = next_ready

    return clean


# --- Candidate Generation ----------------------------------------------------

def label_propagation(graph: Graph, iterations=24, seed=7):
    """Fast community detection."""
    rng = random.Random(seed)
    n = len(graph.ids)
    labels = list(range(n))
    order = list(range(n))

    for _ in range(iterations):
        rng.shuffle(order)
        changed = 0
        for node in order:
            counts = Counter(labels[nb] for nb in graph.adj[node])
            if not counts:
                continue
            best_count = max(counts.values())
            best = min(label for label, count in counts.items() if count == best_count)
            if labels[node] != best:
                labels[node] = best
                changed += 1
        if not changed:
            break

    # Remap to contiguous IDs ordered by community size
    groups = defaultdict(list)
    for node, label in enumerate(labels):
        groups[label].append(node)
    ordered = [label for label, _ in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0]))]
    remap = {label: i for i, label in enumerate(ordered)}
    return [remap[label] for label in labels]


def build_candidates(graph: Graph, labels: list, strategy: dict = None) -> list:
    """Build ranked candidate pool for annealing. Returns internal indices."""
    n = len(graph.ids)

    # Find inactive components from current best strategy
    inactive_nodes = set()
    if strategy:
        result, active = fast_simulate(graph, strategy, return_active=True)
        inactive_nodes = {i for i in range(n) if not active[i]}

    # Community info
    community_size = Counter(labels)
    label_members = defaultdict(list)
    for node, label in enumerate(labels):
        label_members[label].append(node)

    # Score all nodes
    scores = []
    for node in range(n):
        deg = graph.degree[node]
        if deg == 0:
            continue
        cost = COST_PER_NEIGHBOR * deg

        # Key features for cascade initiation
        same_comm = sum(1 for nb in graph.adj[node] if labels[nb] == labels[node])
        low_thresh_nbs = sum(1 for nb in graph.adj[node] if graph.threshold[nb] <= 2)
        medium_thresh_nbs = sum(1 for nb in graph.adj[node] if graph.threshold[nb] <= 4)

        # Inactive neighbor bonus
        inactive_nbs = sum(1 for nb in graph.adj[node] if nb in inactive_nodes) if inactive_nodes else 0

        # Composite score
        score = (
            low_thresh_nbs * 6.0
            + medium_thresh_nbs * 2.0
            + same_comm * 1.5
            + inactive_nbs * 3.0
            + community_size[labels[node]] * 0.03
            - deg * 2.0
            - graph.threshold[node] * 3.0
        )
        scores.append((score, node))

    scores.sort(reverse=True)

    # Take top candidates + all cheap nodes
    candidates = set()
    for _, node in scores[:600]:
        candidates.add(node)
    # All nodes with degree <= 16 (affordable)
    for node in range(n):
        if graph.degree[node] <= 16 and graph.degree[node] > 0:
            candidates.add(node)
    # Neighbors of top candidates with reasonable degree
    top_set = set(node for _, node in scores[:100])
    for node in top_set:
        for nb in graph.adj[node]:
            if graph.degree[nb] <= 50:
                candidates.add(nb)

    # Sort by score
    candidate_list = sorted(candidates, key=lambda n: next(
        (s for s, nd in scores if nd == n), -1000), reverse=True)

    # If too many, trim
    return candidate_list[:1200]


def build_packs(graph: Graph, labels: list, rng: random.Random, n_packs=500) -> list:
    """Generate seed-set packs (coordinated multi-seed attacks)."""
    community_size = Counter(labels)
    label_members = defaultdict(list)
    for node, label in enumerate(labels):
        label_members[label].append(node)

    # Rank communities by size
    big_communities = [label for label, _ in community_size.most_common(50)]
    packs = []

    for label in big_communities:
        members = label_members[label]
        # Cheap members only
        cheap = [n for n in members if graph.degree[n] <= 20]
        if len(cheap) < 2:
            continue

        # Try deterministic best-scoring subsets
        def node_weight(node):
            same = sum(1 for nb in graph.adj[node] if labels[nb] == label)
            weak = sum(1 for nb in graph.adj[node] if graph.threshold[nb] <= 3)
            return same * 2.0 + weak * 4.0 - graph.degree[node] * 1.2

        cheap.sort(key=node_weight, reverse=True)
        # Take prefix subsets
        for size in range(2, min(8, len(cheap) + 1)):
            pack = set(cheap[:size])
            cost = sum(COST_PER_NEIGHBOR * graph.degree[n] for n in pack)
            if cost <= 60000:
                packs.append(pack)

    # Random packs
    for _ in range(n_packs):
        label = rng.choice(big_communities)
        members = [n for n in label_members[label] if graph.degree[n] <= 40]
        if len(members) < 2:
            continue
        size = rng.randint(2, min(8, len(members)))
        pack = set(rng.sample(members, size))
        cost = sum(COST_PER_NEIGHBOR * graph.degree[n] for n in pack)
        if cost <= 60000:
            packs.append(pack)

    return packs


# --- Simulated Annealing Core ------------------------------------------------

def anneal(
    graph: Graph,
    start_strategy: dict,
    candidates: list,
    packs: list,
    labels: list,
    iterations: int,
    seed: int,
    max_day: int = MAX_BUY_DAY,
    t_start: float = 5000.0,
    t_end: float = 30.0,
    cooling_power: float = 1.6,
    verbose: bool = True,
) -> tuple:
    """
    Simulated annealing with multiple move types:
    1. Add single node (weighted by candidate score)
    2. Remove single node
    3. Replace node with another
    4. Shift node to different day
    5. Insert pack (coordinated multi-seed)
    6. Swap two nodes between days
    7. Add 2-3 nodes simultaneously (cluster attack)
    """
    rng = random.Random(seed)
    label_members = defaultdict(list)
    for node, label in enumerate(labels):
        label_members[label].append(node)
    labels_by_size = [label for label, _ in Counter(labels).most_common(60)]

    # Prepare weighted candidate sampling
    n_cands = len(candidates)
    # Weight: 1/sqrt(rank+1) -- heavily favors top candidates
    cand_weights = [1.0 / math.sqrt(i + 1) for i in range(n_cands)]

    current = fast_repair(graph, start_strategy)
    try:
        current_result = fast_simulate(graph, current)
    except ValueError:
        current = fast_repair(graph, {})
        current_result = fast_simulate(graph, current)

    best = {day: nodes[:] for day, nodes in current.items()}
    best_result = dict(current_result)

    accept_count = 0
    improve_count = 0
    last_improve_step = 0
    t0 = time.time()

    for step in range(iterations):
        # Temperature schedule
        progress = step / max(1, iterations)
        temperature = max(t_end, t_start * (1.0 - progress) ** cooling_power)

        trial = {day: nodes[:] for day, nodes in current.items()}
        used = {node for nodes in trial.values() for node in nodes}
        op = rng.random()

        if op < 0.20:
            # ADD: insert a random candidate on a random day
            day = rng.randint(0, max_day)
            if len(trial.get(day, [])) < MAX_CONTRACTS_PER_DAY:
                for _ in range(50):
                    node = rng.choices(candidates, weights=cand_weights, k=1)[0]
                    if node not in used:
                        trial.setdefault(day, []).append(node)
                        break

        elif op < 0.35 and used:
            # REMOVE: remove a random seed
            days_with_seeds = [d for d, nodes in trial.items() if nodes]
            if days_with_seeds:
                day = rng.choice(days_with_seeds)
                trial[day].pop(rng.randrange(len(trial[day])))
                if not trial[day]:
                    del trial[day]

        elif op < 0.55 and used:
            # REPLACE: swap one seed for another candidate
            days_with_seeds = [d for d, nodes in trial.items() if nodes]
            if days_with_seeds:
                day = rng.choice(days_with_seeds)
                pos = rng.randrange(len(trial[day]))
                old_node = trial[day][pos]
                used_without = used - {old_node}
                for _ in range(50):
                    node = rng.choices(candidates, weights=cand_weights, k=1)[0]
                    if node not in used_without:
                        trial[day][pos] = node
                        break

        elif op < 0.72 and used:
            # SHIFT: move a seed to a different day
            days_with_seeds = [d for d, nodes in trial.items() if nodes]
            if days_with_seeds:
                old_day = rng.choice(days_with_seeds)
                pos = rng.randrange(len(trial[old_day]))
                node = trial[old_day].pop(pos)
                if not trial[old_day]:
                    del trial[old_day]
                # Shift by random amount
                delta = rng.choice([-20, -12, -8, -5, -3, -2, -1, 1, 2, 3, 5, 8, 12, 20])
                new_day = max(0, min(max_day, old_day + delta))
                if len(trial.get(new_day, [])) < MAX_CONTRACTS_PER_DAY:
                    trial.setdefault(new_day, []).append(node)
                else:
                    # Put it back
                    trial.setdefault(old_day, []).append(node)

        elif op < 0.82:
            # PACK: insert a coordinated multi-seed attack
            if packs:
                pack = rng.choice(packs)
                day = rng.randint(0, max_day)
                slots = MAX_CONTRACTS_PER_DAY - len(trial.get(day, []))
                additions = [node for node in pack if node not in used]
                rng.shuffle(additions)
                if slots > 0 and additions:
                    trial.setdefault(day, []).extend(additions[:slots])

        elif op < 0.90:
            # CLUSTER ADD: add 2-3 nodes from same community
            if labels_by_size:
                label = rng.choice(labels_by_size[:30])
                members = [n for n in label_members[label]
                           if n not in used and graph.degree[n] <= 30]
                if len(members) >= 2:
                    size = rng.randint(2, min(3, len(members)))
                    nodes_to_add = rng.sample(members, size)
                    day = rng.randint(0, max_day)
                    slots = MAX_CONTRACTS_PER_DAY - len(trial.get(day, []))
                    if slots >= size:
                        trial.setdefault(day, []).extend(nodes_to_add)

        else:
            # SWAP: exchange seeds between two different days
            days_with_seeds = [d for d, nodes in trial.items() if len(nodes) >= 1]
            if len(days_with_seeds) >= 2:
                d1, d2 = rng.sample(days_with_seeds, 2)
                p1 = rng.randrange(len(trial[d1]))
                p2 = rng.randrange(len(trial[d2]))
                trial[d1][p1], trial[d2][p2] = trial[d2][p2], trial[d1][p1]

        # Repair: enforce budget/capacity constraints
        trial = fast_repair(graph, trial)

        try:
            trial_result = fast_simulate(graph, trial)
        except ValueError:
            continue

        delta = trial_result["profit"] - current_result["profit"]

        # Metropolis criterion
        if delta >= 0 or rng.random() < math.exp(delta / temperature):
            current = trial
            current_result = trial_result
            accept_count += 1

            if trial_result["profit"] > best_result["profit"]:
                best = {day: nodes[:] for day, nodes in trial.items()}
                best_result = dict(trial_result)
                improve_count += 1
                last_improve_step = step

                if verbose:
                    elapsed = time.time() - t0
                    gain = best_result['profit'] - fast_simulate(graph, start_strategy)["profit"] if step < 100 else 0
                    marker = "!!!" if best_result['profit'] >= 140000 else "**" if best_result['profit'] >= 135000 else ""
                    print(f"  [{elapsed:7.1f}s] step={step:>7d} NEW BEST: "
                          f"profit={best_result['profit']:>7,} "
                          f"viral={best_result['viral']:>4d} "
                          f"seeds={best_result['seeds']:>2d} "
                          f"cost={best_result['costs']:>6,} "
                          f"T={temperature:.1f} {marker}")

        # Periodic status
        if verbose and step > 0 and step % 50000 == 0:
            elapsed = time.time() - t0
            rate = step / elapsed
            eta = (iterations - step) / rate if rate > 0 else 0
            accept_pct = 100.0 * accept_count / step
            stale = step - last_improve_step
            stale_marker = " [STALE]" if stale > 100000 else ""
            print(f"  [{elapsed:7.1f}s] step={step:>7d}/{iterations} "
                  f"best={best_result['profit']:>7,} "
                  f"current={current_result['profit']:>7,} "
                  f"T={temperature:.0f} "
                  f"accept={accept_pct:.1f}% "
                  f"rate={rate:.0f}/s "
                  f"ETA={eta:.0f}s{stale_marker}")

    return best, best_result


# --- Greedy Local Search (polish after annealing) ----------------------------

def greedy_remove(graph: Graph, strategy: dict) -> dict:
    """Remove seeds that hurt profit."""
    current = {day: nodes[:] for day, nodes in strategy.items()}
    improved = True
    while improved:
        improved = False
        base_profit = fast_simulate(graph, current)["profit"]
        best_trial = None
        best_profit = base_profit
        for day, nodes in list(current.items()):
            for pos in range(len(nodes)):
                trial = {d: ns[:] for d, ns in current.items()}
                trial[day].pop(pos)
                if not trial[day]:
                    del trial[day]
                trial = fast_repair(graph, trial)
                try:
                    r = fast_simulate(graph, trial)
                except ValueError:
                    continue
                if r["profit"] > best_profit:
                    best_trial = trial
                    best_profit = r["profit"]
        if best_trial is not None:
            current = best_trial
            improved = True
    return current


def greedy_add(graph: Graph, strategy: dict, candidates: list, max_rounds=5) -> dict:
    """Greedily add best single seed."""
    current = {day: nodes[:] for day, nodes in strategy.items()}
    for _ in range(max_rounds):
        base_profit = fast_simulate(graph, current)["profit"]
        used = {node for nodes in current.values() for node in nodes}
        best_trial = None
        best_profit = base_profit

        for day in range(0, MAX_BUY_DAY + 1, 2):  # sample days
            if len(current.get(day, [])) >= MAX_CONTRACTS_PER_DAY:
                continue
            for node in candidates[:200]:
                if node in used:
                    continue
                trial = {d: ns[:] for d, ns in current.items()}
                trial.setdefault(day, []).append(node)
                trial = fast_repair(graph, trial)
                try:
                    r = fast_simulate(graph, trial)
                except ValueError:
                    continue
                if r["profit"] > best_profit:
                    best_trial = trial
                    best_profit = r["profit"]

        if best_trial is None:
            break
        current = best_trial
        print(f"    greedy_add: profit={best_profit}")
    return current


def shift_optimize(graph: Graph, strategy: dict, window=15) -> dict:
    """Optimize day assignments for each seed."""
    current = {day: nodes[:] for day, nodes in strategy.items()}
    improved = True
    while improved:
        improved = False
        base_profit = fast_simulate(graph, current)["profit"]
        for day, nodes in list(current.items()):
            for pos, node in enumerate(nodes):
                for new_day in range(max(0, day - window), min(DAYS - 1, day + window) + 1):
                    if new_day == day:
                        continue
                    if len(current.get(new_day, [])) >= MAX_CONTRACTS_PER_DAY:
                        continue
                    trial = {d: ns[:] for d, ns in current.items()}
                    trial[day].pop(pos)
                    if not trial[day]:
                        del trial[day]
                    trial.setdefault(new_day, []).append(node)
                    trial = fast_repair(graph, trial)
                    try:
                        r = fast_simulate(graph, trial)
                    except ValueError:
                        continue
                    if r["profit"] > base_profit:
                        current = trial
                        base_profit = r["profit"]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return current


# --- Main Pipeline -----------------------------------------------------------

def strategy_from_original_ids(graph: Graph, strat_orig: dict) -> dict:
    """Convert strategy with original node IDs to internal indices."""
    result = {}
    for day, nodes in strat_orig.items():
        internal = []
        for node_id in nodes:
            if node_id in graph.index:
                internal.append(graph.index[node_id])
        if internal:
            result[day] = internal
    return result


def strategy_to_original_ids(graph: Graph, strat: dict) -> dict:
    """Convert strategy with internal indices to original node IDs."""
    result = {}
    for day, nodes in strat.items():
        result[day] = [graph.ids[n] for n in nodes]
    return result


def write_submission(graph: Graph, strategy: dict, path: str):
    """Write submission.csv from internal-index strategy."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["day", "node_ids"])
        for day in range(DAYS):
            nodes = strategy.get(day, [])
            if nodes:
                writer.writerow([day, " ".join(str(graph.ids[n]) for n in nodes)])
            else:
                writer.writerow([day, "-1"])


def main():
    start_time = time.time()
    print("=" * 70)
    print("SIMULATED ANNEALING -- Network Influence Profit Maximization")
    print("=" * 70)

    # -- Load graph ---------------------------------------------------------
    # Try multiple paths (local dev + Kaggle dataset)
    possible_paths = [
        "data/marketing_edges.txt",
        "marketing_edges.txt",
        "/kaggle/input/network-influence-profit-challenge/marketing_edges.txt",
        "/kaggle/input/cu-optimal-seeding/marketing_edges.txt",
    ]
    # Also glob for any marketing_edges.txt under /kaggle/input/
    import glob
    possible_paths += glob.glob("/kaggle/input/**/marketing_edges.txt", recursive=True)
    edge_path = None
    for p in possible_paths:
        if Path(p).exists():
            edge_path = p
            break
    if edge_path is None:
        raise FileNotFoundError(
            f"marketing_edges.txt not found! Tried: {possible_paths[:4]}\n"
            "On Kaggle: add dataset with marketing_edges.txt, it will be at "
            "/kaggle/input/<dataset-name>/marketing_edges.txt"
        )

    print(f"\nLoading graph from {edge_path}...")
    graph = read_graph(edge_path)
    print(f"  Nodes: {len(graph.ids)}, Mean degree: {sum(graph.degree)/len(graph.degree):.1f}")

    # -- Community detection (multiple seeds for diversity) -----------------
    print("\nRunning community detection...")
    all_labels = []
    for lp_seed in [7, 13, 42, 99, 137]:
        labels = label_propagation(graph, seed=lp_seed)
        all_labels.append(labels)
    primary_labels = all_labels[0]
    print(f"  Communities: {len(set(primary_labels))}")

    # -- Build starting strategies -----------------------------------------
    print("\nPreparing starting strategies...")
    starting_strategies = []
    for name, strat_orig in KNOWN_BEST_SEEDS.items():
        strat = strategy_from_original_ids(graph, strat_orig)
        strat = fast_repair(graph, strat)
        try:
            r = fast_simulate(graph, strat)
            print(f"  {name}: profit={r['profit']:,} viral={r['viral']} seeds={r['seeds']}")
            starting_strategies.append((r["profit"], strat, name))
        except ValueError as e:
            print(f"  {name}: INVALID ({e})")

    starting_strategies.sort(key=lambda x: x[0], reverse=True)

    # -- Build candidate pools ---------------------------------------------
    print("\nBuilding candidate pool...")
    best_strat = starting_strategies[0][1] if starting_strategies else {}
    candidates = build_candidates(graph, primary_labels, best_strat)
    print(f"  Candidates: {len(candidates)}")

    # -- Build packs for multiple label sets -------------------------------
    print("Building seed packs...")
    rng = random.Random(42)
    all_packs = []
    for labels in all_labels:
        packs = build_packs(graph, labels, rng, n_packs=300)
        all_packs.extend(packs)
    print(f"  Packs: {len(all_packs)}")

    # -- Multi-restart annealing -------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"STARTING ANNEALING: {NUM_RESTARTS} restarts x {ITERATIONS_PER_RESTART:,} iterations")
    print(f"Target: 145,000+  |  Current best known: 139,450")
    print(f"Time budget: {TOTAL_TIME_BUDGET/3600:.1f}h")
    print(f"{'=' * 70}")

    global_best = None
    global_best_result = None
    restart_history = []

    for restart in range(NUM_RESTARTS):
        elapsed = time.time() - start_time
        remaining = TOTAL_TIME_BUDGET - elapsed
        if remaining < 60:
            print(f"\n  TIME UP ({elapsed/3600:.2f}h used). Stopping restarts.")
            break

        # Adapt iterations to remaining time (~1500 iter/s on CPU)
        iters = min(ITERATIONS_PER_RESTART, int(remaining * 1200))

        # Choose starting point: cycle through known + perturb global best
        if restart < len(starting_strategies):
            _, start_strat, name = starting_strategies[restart % len(starting_strategies)]
        else:
            start_strat = {day: nodes[:] for day, nodes in global_best.items()} if global_best else {}
            name = "global_best"

        # Vary annealing parameters per restart for diversity
        anneal_seed = 1000 + restart * 7919
        t_start_val = [3500, 4500, 5500, 7000, 9000][restart % 5]
        t_end_val = [20, 30, 50][restart % 3]
        cooling_val = [1.4, 1.6, 1.8, 2.0][restart % 4]

        # Use different label set for diversity
        restart_labels = all_labels[restart % len(all_labels)]
        restart_candidates = build_candidates(graph, restart_labels, start_strat)

        gb_str = f"global_best={global_best_result['profit']:,}" if global_best_result else "no global best yet"
        print(f"\n{'-' * 70}")
        print(f"[Restart {restart+1}/{NUM_RESTARTS}] start='{name}' | {iters:,} iters | "
              f"T={t_start_val}->{t_end_val} cool={cooling_val}")
        print(f"  {gb_str} | elapsed={elapsed:.0f}s | remaining={remaining:.0f}s")

        # Run annealing
        best, best_result = anneal(
            graph, start_strat, restart_candidates, all_packs, restart_labels,
            iterations=iters, seed=anneal_seed, max_day=MAX_BUY_DAY,
            t_start=t_start_val, t_end=t_end_val, cooling_power=cooling_val,
        )

        # Polish with local search
        print(f"  Polishing: remove -> shift -> add -> remove -> shift ...")
        best = greedy_remove(graph, best)
        best = shift_optimize(graph, best)
        best = greedy_add(graph, best, restart_candidates, max_rounds=3)
        best = greedy_remove(graph, best)
        best = shift_optimize(graph, best)

        polished_result = fast_simulate(graph, best)
        restart_history.append(polished_result['profit'])

        # Status indicator
        if polished_result['profit'] >= 145000:
            status = "TARGET HIT!!!"
        elif polished_result['profit'] >= 140000:
            status = "EXCELLENT"
        elif polished_result['profit'] >= 135000:
            status = "good"
        else:
            status = "below target"

        print(f"  Result: profit={polished_result['profit']:>7,} "
              f"viral={polished_result['viral']} seeds={polished_result['seeds']} "
              f"[{status}]")

        if global_best_result is None or polished_result["profit"] > global_best_result["profit"]:
            global_best = best
            global_best_result = polished_result
            print(f"  >>> NEW GLOBAL BEST: {global_best_result['profit']:,} <<<")

            # Save intermediate result
            write_submission(graph, global_best, "submission.csv")
            orig_strat = strategy_to_original_ids(graph, global_best)
            report = {
                "result": global_best_result,
                "strategy": {str(day): nodes for day, nodes in orig_strat.items()},
                "restart": restart,
                "elapsed_s": time.time() - start_time,
            }
            with open("best_result.json", "w") as f:
                json.dump(report, f, indent=2)

        # Summary table every 5 restarts
        if (restart + 1) % 5 == 0:
            print(f"\n  === SUMMARY after {restart+1} restarts ===")
            print(f"  Best: {max(restart_history):,} | Avg: {sum(restart_history)//len(restart_history):,} "
                  f"| Worst: {min(restart_history):,}")
            print(f"  Global best: {global_best_result['profit']:,}")

    # -- Final report ------------------------------------------------------
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"FINAL RESULT")
    print(f"{'=' * 70}")
    if global_best_result is None:
        print("  No valid result found!")
        return
    print(f"  Profit:  {global_best_result['profit']:>10,}")
    print(f"  Income:  {global_best_result['income']:>10,}")
    print(f"  Costs:   {global_best_result['costs']:>10,}")
    print(f"  Viral:   {global_best_result['viral']:>10,}")
    print(f"  Seeds:   {global_best_result['seeds']:>10}")
    print(f"  Active:  {global_best_result['active']:>10,}")
    print(f"  Time:    {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"\n  Restarts completed: {len(restart_history)}")
    if restart_history:
        print(f"  All restart profits: {restart_history}")

    # Verdict
    if global_best_result['profit'] >= 145000:
        print(f"\n  !!! TARGET 145K ACHIEVED !!!")
    elif global_best_result['profit'] >= 140000:
        print(f"\n  Close to target. Try more restarts or longer runs.")
    else:
        print(f"\n  Below target. Consider tuning parameters.")

    # Print strategy
    orig_strat = strategy_to_original_ids(graph, global_best)
    print(f"\nWinning Strategy:")
    for day in sorted(orig_strat.keys()):
        nodes = orig_strat[day]
        costs = [COST_PER_NEIGHBOR * graph.degree[graph.index[n]] for n in nodes]
        print(f"  Day {day:2d}: nodes={nodes} costs={costs} total={sum(costs)}")

    # Write final submission
    write_submission(graph, global_best, "submission.csv")
    print(f"\n>>> submission.csv written (profit={global_best_result['profit']:,}) <<<")

    # Save full report
    report = {
        "result": global_best_result,
        "strategy": {str(day): nodes for day, nodes in orig_strat.items()},
        "total_time_s": total_time,
        "restart_history": restart_history,
        "params": {
            "num_restarts": NUM_RESTARTS,
            "iterations_per_restart": ITERATIONS_PER_RESTART,
            "max_buy_day": MAX_BUY_DAY,
        }
    }
    with open("best_result.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Report: best_result.json")


if __name__ == "__main__":
    main()
