#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


INITIAL_BUDGET = 10_000
COST_PER_NEIGHBOR = 300
INCOME_PER_VIRAL = 50
MAX_CONTRACTS_PER_DAY = 10
DAYS = 60
THRESHOLD = 0.18


@dataclass
class Graph:
    ids: list[int]
    index: dict[int, int]
    adj: list[list[int]]
    degree: list[int]
    threshold: list[int]
    edges: list[tuple[int, int]]


def read_graph(path: Path) -> Graph:
    raw_edges: list[tuple[int, int]] = []
    nodes: set[int] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            a, b = map(int, line.split()[:2])
            if a == b:
                continue
            raw_edges.append((a, b))
            nodes.add(a)
            nodes.add(b)
    ids = sorted(nodes)
    index = {node: i for i, node in enumerate(ids)}
    adj_sets = [set() for _ in ids]
    for a, b in raw_edges:
        ia, ib = index[a], index[b]
        adj_sets[ia].add(ib)
        adj_sets[ib].add(ia)
    adj = [sorted(s) for s in adj_sets]
    degree = [len(v) for v in adj]
    threshold = [max(1, math.ceil(THRESHOLD * d)) for d in degree]
    edges = [(index[a], index[b]) for a, b in raw_edges]
    return Graph(ids=ids, index=index, adj=adj, degree=degree, threshold=threshold, edges=edges)


def to_indices(graph: Graph, strategy: dict[int, list[int]]) -> dict[int, list[int]]:
    return {
        day: [graph.index[node] for node in nodes if node in graph.index]
        for day, nodes in strategy.items()
    }


def simulate(graph: Graph, strategy: dict[int, list[int]], return_daily: bool = False):
    n = len(graph.ids)
    active = [False] * n
    balance = INITIAL_BUDGET
    income = 0
    costs = 0
    daily = []

    for day in range(DAYS):
        old = active[:]
        new = []
        for node in range(n):
            if old[node]:
                continue
            infected_neighbors = 0
            for neighbor in graph.adj[node]:
                if old[neighbor]:
                    infected_neighbors += 1
                    if infected_neighbors >= graph.threshold[node]:
                        new.append(node)
                        break
        for node in new:
            active[node] = True
        income += INCOME_PER_VIRAL * len(new)
        balance += INCOME_PER_VIRAL * len(new)

        bought = []
        for node in strategy.get(day, []):
            cost = COST_PER_NEIGHBOR * graph.degree[node]
            balance -= cost
            costs += cost
            bought.append(node)
            if balance < -1e-9:
                raise ValueError(f"negative balance on day {day}: {balance}")
            active[node] = True
        if return_daily:
            daily.append(
                {
                    "day": day,
                    "viral": len(new),
                    "bought": [graph.ids[node] for node in bought],
                    "balance": balance,
                    "active": sum(active),
                }
            )
    result = {
        "profit": income - costs,
        "income": income,
        "costs": costs,
        "balance": balance,
        "active": sum(active),
        "viral": income // INCOME_PER_VIRAL,
        "seeds": sum(len(v) for v in strategy.values()),
    }
    return (result, daily) if return_daily else result


def closure_from_active(graph: Graph, active_seed: set[int], max_days: int = DAYS) -> set[int]:
    active = [False] * len(graph.ids)
    for node in active_seed:
        active[node] = True
    for _ in range(max_days):
        old = active[:]
        new = []
        for node in range(len(graph.ids)):
            if old[node]:
                continue
            infected = 0
            for neighbor in graph.adj[node]:
                if old[neighbor]:
                    infected += 1
                    if infected >= graph.threshold[node]:
                        new.append(node)
                        break
        if not new:
            break
        for node in new:
            active[node] = True
    return {i for i, ok in enumerate(active) if ok}


def cascade_layers(graph: Graph, seeds: set[int], max_days: int = DAYS) -> list[list[int]]:
    active = [False] * len(graph.ids)
    counts = [0] * len(graph.ids)
    for node in seeds:
        active[node] = True
    frontier = []
    for node in seeds:
        for nb in graph.adj[node]:
            if active[nb]:
                continue
            counts[nb] += 1
    for node, count in enumerate(counts):
        if not active[node] and count >= graph.threshold[node]:
            frontier.append(node)

    layers = []
    for _ in range(max_days):
        if not frontier:
            break
        current = sorted(set(node for node in frontier if not active[node]))
        if not current:
            break
        layers.append(current)
        next_frontier = []
        for node in current:
            active[node] = True
        for node in current:
            for nb in graph.adj[node]:
                if active[nb]:
                    continue
                counts[nb] += 1
                if counts[nb] >= graph.threshold[nb]:
                    next_frontier.append(nb)
        frontier = next_frontier
    return layers


def seedset_score(graph: Graph, seeds: set[int], max_days: int = DAYS) -> dict[str, int]:
    layers = cascade_layers(graph, seeds, max_days=max_days)
    viral = sum(len(layer) for layer in layers)
    cost = sum(COST_PER_NEIGHBOR * graph.degree[node] for node in seeds)
    return {
        "profit": viral * INCOME_PER_VIRAL - cost,
        "viral": viral,
        "cost": cost,
        "active": viral + len(seeds),
        "depth": len(layers),
    }


def components(graph: Graph) -> list[list[int]]:
    seen = [False] * len(graph.ids)
    out = []
    for start in range(len(graph.ids)):
        if seen[start]:
            continue
        q = [start]
        seen[start] = True
        comp = []
        while q:
            node = q.pop()
            comp.append(node)
            for nb in graph.adj[node]:
                if not seen[nb]:
                    seen[nb] = True
                    q.append(nb)
        out.append(comp)
    out.sort(key=len, reverse=True)
    return out


def label_propagation(graph: Graph, iterations: int = 24, seed: int = 7) -> list[int]:
    rng = random.Random(seed)
    labels = list(range(len(graph.ids)))
    order = list(range(len(graph.ids)))
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
    groups = defaultdict(list)
    for node, label in enumerate(labels):
        groups[label].append(node)
    ordered = [label for label, _ in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))]
    remap = {label: i for i, label in enumerate(ordered)}
    return [remap[label] for label in labels]


def make_candidates(graph: Graph, labels: list[int]) -> list[int]:
    triangles = [0] * len(graph.ids)
    adj_sets = [set(v) for v in graph.adj]
    for node, nbs in enumerate(graph.adj):
        count = 0
        for i, a in enumerate(nbs):
            sa = adj_sets[a]
            for b in nbs[i + 1 :]:
                if b in sa:
                    count += 1
        triangles[node] = count

    community_size = Counter(labels)
    candidates = []
    for node in range(len(graph.ids)):
        deg = graph.degree[node]
        same = sum(1 for nb in graph.adj[node] if labels[nb] == labels[node])
        cheap = deg <= 33
        if cheap or deg <= 45 or same >= graph.threshold[node] * 2:
            candidates.append(node)

    def score(node: int) -> tuple[float, int]:
        deg = graph.degree[node]
        t = graph.threshold[node]
        low_neighbors = sum(1 for nb in graph.adj[node] if graph.threshold[nb] <= 2)
        same = sum(1 for nb in graph.adj[node] if labels[nb] == labels[node])
        value = (
            low_neighbors * 4
            + same * 1.5
            + triangles[node] * 0.25
            + community_size[labels[node]] * 0.04
            - deg * 2.1
            - t * 4
        )
        return (value, -deg)

    candidates.sort(key=score, reverse=True)
    return candidates[:900]


def greedy_plan(
    graph: Graph,
    candidates: list[int],
    days_to_buy: int,
    lookahead_days: int,
    min_gain: int,
    top_k: int,
) -> dict[int, list[int]]:
    strategy: dict[int, list[int]] = defaultdict(list)
    active: set[int] = set()
    bought: set[int] = set()
    balance = INITIAL_BUDGET

    for day in range(days_to_buy):
        # Real daily viral update before contracts.
        old = active.copy()
        new = []
        for node in range(len(graph.ids)):
            if node in old:
                continue
            infected = sum(1 for nb in graph.adj[node] if nb in old)
            if infected >= graph.threshold[node]:
                new.append(node)
        for node in new:
            active.add(node)
        balance += INCOME_PER_VIRAL * len(new)

        bought_today = 0
        while bought_today < MAX_CONTRACTS_PER_DAY:
            base_closure = None
            affordable = [
                node
                for node in candidates
                if node not in active
                and node not in bought
                and COST_PER_NEIGHBOR * graph.degree[node] <= balance
            ]
            if not affordable:
                break

            shortlist = []
            for node in affordable[:top_k]:
                frontier = sum(1 for nb in graph.adj[node] if nb not in active)
                near = sum(
                    1
                    for nb in graph.adj[node]
                    if nb not in active
                    and sum(1 for x in graph.adj[nb] if x in active) + 1 >= graph.threshold[nb]
                )
                same_active = sum(1 for nb in graph.adj[node] if nb in active)
                heuristic = near * 10 + frontier + same_active * 3 - graph.degree[node] * 1.7
                shortlist.append((heuristic, node))
            shortlist.sort(reverse=True)

            best_node = None
            best_net = -10**9
            best_gain = 0
            if base_closure is None:
                base_closure = closure_from_active(graph, active, lookahead_days)
            for _, node in shortlist[: min(90, len(shortlist))]:
                trial = closure_from_active(graph, active | {node}, lookahead_days)
                gain = len(trial - base_closure)
                net = gain * INCOME_PER_VIRAL - COST_PER_NEIGHBOR * graph.degree[node]
                if net > best_net:
                    best_net = net
                    best_node = node
                    best_gain = gain

            if best_node is None or best_gain < min_gain or best_net <= 0:
                break
            strategy[day].append(best_node)
            bought.add(best_node)
            active.add(best_node)
            balance -= COST_PER_NEIGHBOR * graph.degree[best_node]
            bought_today += 1
    return dict(strategy)


def cluster_bootstrap_plan(graph: Graph, labels: list[int], max_days: int = 8) -> dict[int, list[int]]:
    groups = defaultdict(list)
    for node, label in enumerate(labels):
        groups[label].append(node)

    packs = []
    for label, members in groups.items():
        member_set = set(members)
        cheap = [n for n in members if graph.degree[n] <= 8]
        cheap.sort(
            key=lambda n: (
                -sum(1 for nb in graph.adj[n] if nb in member_set and graph.threshold[nb] <= 2),
                graph.degree[n],
            )
        )
        for size in range(1, min(10, len(cheap)) + 1):
            seeds = cheap[:size]
            cost = sum(COST_PER_NEIGHBOR * graph.degree[n] for n in seeds)
            if cost > INITIAL_BUDGET:
                break
            closure = closure_from_active(graph, set(seeds), DAYS)
            viral = len(closure) - len(seeds)
            net = viral * INCOME_PER_VIRAL - cost
            packs.append((net, viral, -cost, label, seeds))
    packs.sort(reverse=True)

    strategy: dict[int, list[int]] = defaultdict(list)
    used = set()
    balance = INITIAL_BUDGET
    day = 0
    for net, viral, neg_cost, label, seeds in packs:
        if day >= max_days:
            break
        seeds = [node for node in seeds if node not in used]
        if not seeds:
            continue
        cost = sum(COST_PER_NEIGHBOR * graph.degree[n] for n in seeds)
        if net <= 0 or cost > balance or len(seeds) > MAX_CONTRACTS_PER_DAY:
            continue
        strategy[day].extend(seeds)
        used.update(seeds)
        balance -= cost
        day += 1
    return dict(strategy)


def random_cluster_seedsets(
    graph: Graph,
    labels: list[int],
    rounds: int = 18_000,
    seed: int = 13,
    budget: int = INITIAL_BUDGET,
    max_degree: int = 16,
) -> list[tuple[dict[str, int], set[int]]]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for node, label in enumerate(labels):
        groups[label].append(node)

    ranked_groups = []
    for label, members in groups.items():
        low = [n for n in members if graph.degree[n] <= max_degree]
        if not low:
            continue
        density_hint = sum(
            1
            for node in low
            for nb in graph.adj[node]
            if labels[nb] == label and graph.threshold[nb] <= 4
        )
        ranked_groups.append((len(members) + density_hint * 2, label, low))
    ranked_groups.sort(reverse=True)
    ranked_groups = ranked_groups[:45]

    def node_weight(node: int, label: int) -> float:
        deg = graph.degree[node]
        same = sum(1 for nb in graph.adj[node] if labels[nb] == label)
        vulnerable = sum(1 for nb in graph.adj[node] if graph.threshold[nb] <= 2)
        return max(0.2, same * 2.5 + vulnerable * 4 - deg * 1.4)

    best: list[tuple[dict[str, int], set[int]]] = []
    seen: set[tuple[int, ...]] = set()

    # Deterministic community prefixes: surprisingly strong for threshold cascades.
    for _, label, low in ranked_groups:
        ordered = sorted(low, key=lambda n: (-node_weight(n, label), graph.degree[n], graph.ids[n]))
        seeds = []
        cost = 0
        for node in ordered:
            node_cost = COST_PER_NEIGHBOR * graph.degree[node]
            if len(seeds) >= MAX_CONTRACTS_PER_DAY or cost + node_cost > budget:
                continue
            seeds.append(node)
            cost += node_cost
            key = tuple(sorted(seeds))
            if key not in seen:
                seen.add(key)
                score = seedset_score(graph, set(seeds))
                best.append((score, set(seeds)))

    for _ in range(rounds):
        _, label, low = rng.choice(ranked_groups)
        weighted = sorted(low, key=lambda n: rng.random() / node_weight(n, label))
        seeds = []
        cost = 0
        target_size = rng.randint(2, MAX_CONTRACTS_PER_DAY)
        for node in weighted:
            if len(seeds) >= target_size:
                break
            node_cost = COST_PER_NEIGHBOR * graph.degree[node]
            if cost + node_cost <= budget:
                seeds.append(node)
                cost += node_cost
        if not seeds:
            continue
        key = tuple(sorted(seeds))
        if key in seen:
            continue
        seen.add(key)
        score = seedset_score(graph, set(seeds))
        if score["profit"] > 0 or score["viral"] > 80:
            best.append((score, set(seeds)))

    best.sort(key=lambda item: (item[0]["profit"], item[0]["viral"], -item[0]["cost"]), reverse=True)
    return best[:160]


def sequential_pack_plan(
    graph: Graph,
    packs: list[tuple[dict[str, int], set[int]]],
    start_pack: set[int] | None = None,
    max_buy_day: int = 34,
) -> dict[int, list[int]]:
    strategy: dict[int, list[int]] = defaultdict(list)
    active: set[int] = set()
    bought: set[int] = set()
    balance = INITIAL_BUDGET

    if start_pack:
        cost = sum(COST_PER_NEIGHBOR * graph.degree[n] for n in start_pack)
        if len(start_pack) <= MAX_CONTRACTS_PER_DAY and cost <= balance:
            strategy[0].extend(sorted(start_pack))
            active.update(start_pack)
            bought.update(start_pack)
            balance -= cost

    for day in range(DAYS):
        if not (day == 0 and start_pack):
            old = active.copy()
            new = []
            for node in range(len(graph.ids)):
                if node in old:
                    continue
                infected = sum(1 for nb in graph.adj[node] if nb in old)
                if infected >= graph.threshold[node]:
                    new.append(node)
            for node in new:
                active.add(node)
            balance += INCOME_PER_VIRAL * len(new)

        if day > max_buy_day:
            continue

        slots = MAX_CONTRACTS_PER_DAY - len(strategy.get(day, []))
        while slots > 0:
            base = closure_from_active(graph, active, DAYS - day - 1)
            best_pack = None
            best_net = 0
            best_gain = 0
            best_cost = 0
            for _, original_seeds in packs:
                if original_seeds & bought:
                    continue
                seeds = set(node for node in original_seeds if node not in active and node not in bought)
                if not seeds or len(seeds) > slots:
                    continue
                cost = sum(COST_PER_NEIGHBOR * graph.degree[n] for n in seeds)
                if cost > balance:
                    continue
                trial = closure_from_active(graph, active | seeds, DAYS - day - 1)
                gain = len(trial - base)
                net = gain * INCOME_PER_VIRAL - cost
                if net > best_net:
                    best_pack = seeds
                    best_net = net
                    best_gain = gain
                    best_cost = cost
            if best_pack is None or best_gain < 10:
                break
            strategy[day].extend(sorted(best_pack))
            bought.update(best_pack)
            active.update(best_pack)
            balance -= best_cost
            slots -= len(best_pack)
    return dict(strategy)


def merge_to_valid(graph: Graph, strategies: list[dict[int, list[int]]]) -> dict[int, list[int]]:
    merged: dict[int, list[int]] = defaultdict(list)
    seen = set()
    for strategy in strategies:
        for day in sorted(strategy):
            for node in strategy[day]:
                if node in seen or len(merged[day]) >= MAX_CONTRACTS_PER_DAY:
                    continue
                merged[day].append(node)
                seen.add(node)
    # Drop anything that violates the actual budget.
    clean: dict[int, list[int]] = defaultdict(list)
    active = [False] * len(graph.ids)
    balance = INITIAL_BUDGET
    for day in range(DAYS):
        old = active[:]
        new = []
        for node in range(len(graph.ids)):
            if old[node]:
                continue
            infected = sum(1 for nb in graph.adj[node] if old[nb])
            if infected >= graph.threshold[node]:
                new.append(node)
        for node in new:
            active[node] = True
        balance += INCOME_PER_VIRAL * len(new)
        for node in merged.get(day, []):
            cost = COST_PER_NEIGHBOR * graph.degree[node]
            if not active[node] and cost <= balance and len(clean[day]) < MAX_CONTRACTS_PER_DAY:
                clean[day].append(node)
                active[node] = True
                balance -= cost
    return dict(clean)


def write_submission(graph: Graph, strategy: dict[int, list[int]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["day", "node_ids"])
        for day in range(DAYS):
            nodes = strategy.get(day, [])
            writer.writerow([day, " ".join(str(graph.ids[node]) for node in nodes) if nodes else "-1"])


def read_submission(graph: Graph, path: Path) -> dict[int, list[int]]:
    strategy: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            day = int(row["day"])
            raw = row["node_ids"].strip()
            if raw == "-1" or not raw:
                strategy[day] = []
            else:
                strategy[day] = [graph.index[int(node)] for node in raw.split()]
    return normalize_strategy(strategy)


def normalize_strategy(strategy: dict[int, list[int]]) -> dict[int, list[int]]:
    clean: dict[int, list[int]] = {}
    used = set()
    for day in range(DAYS):
        nodes = []
        for node in strategy.get(day, []):
            if node in used:
                continue
            if len(nodes) >= MAX_CONTRACTS_PER_DAY:
                break
            nodes.append(node)
            used.add(node)
        if nodes:
            clean[day] = nodes
    return clean


def copy_strategy(strategy: dict[int, list[int]]) -> dict[int, list[int]]:
    return {day: nodes[:] for day, nodes in strategy.items()}


def strategy_nodes(strategy: dict[int, list[int]]) -> set[int]:
    return {node for nodes in strategy.values() for node in nodes}


def repair_to_valid(graph: Graph, strategy: dict[int, list[int]]) -> dict[int, list[int]]:
    """Keep scheduled contracts in order, dropping duplicates, infected nodes, and over-budget buys."""
    strategy = normalize_strategy(strategy)
    clean: dict[int, list[int]] = defaultdict(list)
    active = [False] * len(graph.ids)
    balance = INITIAL_BUDGET
    used = set()
    for day in range(DAYS):
        old = active[:]
        new = []
        for node in range(len(graph.ids)):
            if old[node]:
                continue
            infected = sum(1 for nb in graph.adj[node] if old[nb])
            if infected >= graph.threshold[node]:
                new.append(node)
        for node in new:
            active[node] = True
        balance += INCOME_PER_VIRAL * len(new)

        for node in strategy.get(day, []):
            cost = COST_PER_NEIGHBOR * graph.degree[node]
            if node in used or active[node] or len(clean[day]) >= MAX_CONTRACTS_PER_DAY:
                continue
            if cost > balance:
                continue
            clean[day].append(node)
            used.add(node)
            active[node] = True
            balance -= cost
    return dict(clean)


def build_anneal_pools(
    graph: Graph,
    labels: list[int],
    seedsets: list[tuple[dict[str, int], set[int]]],
    later_seedsets: list[tuple[dict[str, int], set[int]]],
    candidates: list[int],
) -> tuple[list[int], list[set[int]]]:
    node_scores: Counter[int] = Counter()
    packs: list[set[int]] = []
    for rank, (score, seeds) in enumerate(seedsets[:120] + later_seedsets[:220]):
        if not seeds:
            continue
        packs.append(set(seeds))
        bonus = max(1, score["profit"] // 500 + 240 - rank)
        for node in seeds:
            node_scores[node] += bonus
            for nb in graph.adj[node]:
                if labels[nb] == labels[node] and graph.degree[nb] <= 60:
                    node_scores[nb] += max(1, bonus // 8)
    for rank, node in enumerate(candidates[:550]):
        node_scores[node] += max(1, 550 - rank)
    pool = [node for node, _ in node_scores.most_common(900)]
    return pool, packs


def anneal_strategy(
    graph: Graph,
    start: dict[int, list[int]],
    node_pool: list[int],
    pack_pool: list[set[int]],
    iterations: int,
    seed: int,
    max_day: int = 34,
) -> tuple[dict[int, list[int]], dict[str, int]]:
    rng = random.Random(seed)
    current = repair_to_valid(graph, start)
    current_result = simulate(graph, current)
    best = copy_strategy(current)
    best_result = dict(current_result)
    node_weights = [max(1, len(node_pool) - i) for i in range(len(node_pool))]

    def random_node(exclude: set[int]) -> int | None:
        for _ in range(60):
            node = rng.choices(node_pool, weights=node_weights, k=1)[0]
            if node not in exclude:
                return node
        available = [node for node in node_pool if node not in exclude]
        return rng.choice(available) if available else None

    for step in range(iterations):
        trial = copy_strategy(current)
        used = strategy_nodes(trial)
        op = rng.random()

        if op < 0.24:
            day = rng.randint(0, max_day)
            if len(trial.get(day, [])) < MAX_CONTRACTS_PER_DAY:
                node = random_node(used)
                if node is not None:
                    trial.setdefault(day, []).append(node)
        elif op < 0.42 and used:
            day = rng.choice([d for d, nodes in trial.items() if nodes])
            trial[day].pop(rng.randrange(len(trial[day])))
            if not trial[day]:
                trial.pop(day, None)
        elif op < 0.62 and used:
            day = rng.choice([d for d, nodes in trial.items() if nodes])
            pos = rng.randrange(len(trial[day]))
            old = trial[day][pos]
            used_without_old = used - {old}
            node = random_node(used_without_old)
            if node is not None:
                trial[day][pos] = node
        elif op < 0.78 and used:
            old_day = rng.choice([d for d, nodes in trial.items() if nodes])
            node = trial[old_day].pop(rng.randrange(len(trial[old_day])))
            if not trial[old_day]:
                trial.pop(old_day, None)
            new_day = max(0, min(max_day, old_day + rng.choice([-8, -5, -3, -2, -1, 1, 2, 3, 5, 8])))
            if len(trial.get(new_day, [])) < MAX_CONTRACTS_PER_DAY:
                trial.setdefault(new_day, []).append(node)
        elif pack_pool:
            pack = set(rng.choice(pack_pool))
            day = rng.randint(0, max_day)
            slots = MAX_CONTRACTS_PER_DAY - len(trial.get(day, []))
            additions = [node for node in pack if node not in used]
            rng.shuffle(additions)
            if slots > 0 and additions:
                trial.setdefault(day, []).extend(additions[:slots])

        trial = repair_to_valid(graph, trial)
        try:
            trial_result = simulate(graph, trial)
        except ValueError:
            continue

        delta = trial_result["profit"] - current_result["profit"]
        temperature = max(25.0, 5500.0 * (1.0 - step / max(1, iterations)) ** 1.7)
        if delta >= 0 or rng.random() < math.exp(delta / temperature):
            current = trial
            current_result = trial_result
            if trial_result["profit"] > best_result["profit"]:
                best = copy_strategy(trial)
                best_result = dict(trial_result)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "best_profit": best_result["profit"],
                            "active": best_result["active"],
                            "seeds": best_result["seeds"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    return best, best_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, default=Path("marketing_edges.txt"))
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start", type=Path, default=None, help="Existing submission to anneal from")
    parser.add_argument("--anneal", action="store_true", help="Run simulated annealing after heuristic search")
    parser.add_argument("--anneal-only", action="store_true", help="Only anneal from --start or current best")
    parser.add_argument("--iterations", type=int, default=2500)
    args = parser.parse_args()

    graph = read_graph(args.edges)
    labels = label_propagation(graph, seed=args.seed)
    candidates = make_candidates(graph, labels)

    seedsets = random_cluster_seedsets(graph, labels, seed=args.seed + 101)
    later_seedsets = random_cluster_seedsets(
        graph,
        labels,
        rounds=18_000 if args.anneal_only else 24_000,
        seed=args.seed + 202,
        budget=65_000,
        max_degree=45,
    )

    strategies = []
    if args.start and args.start.exists():
        strategies.append(read_submission(graph, args.start))
    if not args.anneal_only:
        if Path("submission.csv").exists() and args.out != Path("submission.csv"):
            strategies.append(read_submission(graph, Path("submission.csv")))
    for _, seeds in seedsets:
        strategies.append({0: list(seeds)})
    boot = cluster_bootstrap_plan(graph, labels)
    if not args.anneal_only:
        for _, seeds in seedsets[:20]:
            strategies.append(sequential_pack_plan(graph, later_seedsets, start_pack=seeds))
        strategies.append(boot)
        for days_to_buy in (1, 2, 3, 5, 8, 12, 20):
            for lookahead in (8, 16, 32, 59):
                for min_gain in (1, 2, 4, 8, 12):
                    plan = greedy_plan(
                        graph,
                        candidates,
                        days_to_buy=days_to_buy,
                        lookahead_days=lookahead,
                        min_gain=min_gain,
                        top_k=500,
                    )
                    strategies.append(plan)

    best = None
    best_result = None
    for strategy in strategies:
        for candidate in (strategy, merge_to_valid(graph, [strategy, boot]), merge_to_valid(graph, [boot, strategy])):
            try:
                result = simulate(graph, candidate)
            except ValueError:
                continue
            if best_result is None or result["profit"] > best_result["profit"]:
                best = candidate
                best_result = result

    assert best is not None and best_result is not None
    if args.anneal or args.anneal_only:
        node_pool, pack_pool = build_anneal_pools(graph, labels, seedsets, later_seedsets, candidates)
        best, best_result = anneal_strategy(
            graph,
            best,
            node_pool,
            pack_pool,
            iterations=args.iterations,
            seed=args.seed + 303,
        )

    result, daily = simulate(graph, best, return_daily=True)
    write_submission(graph, best, args.out)
    report = {
        "result": result,
        "contracts": {day: [graph.ids[n] for n in best.get(day, [])] for day in range(DAYS) if best.get(day)},
        "daily_nonzero": [row for row in daily if row["viral"] or row["bought"]],
    }
    Path(args.out.with_suffix(".report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    print(f"wrote {args.out} and {args.out.with_suffix('.report.json')}")


if __name__ == "__main__":
    main()
