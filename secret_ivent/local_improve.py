#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import solve_marketing as base


def fast_simulate(graph: base.Graph, strategy: dict[int, list[int]], return_daily: bool = False):
    n = len(graph.ids)
    active = bytearray(n)
    active_count = 0
    infected_counts = [0] * n
    ready: list[int] = []
    queued_next = bytearray(n)

    balance = base.INITIAL_BUDGET
    income = 0
    costs = 0
    daily = []

    def activate(node: int, next_ready: list[int]) -> bool:
        nonlocal active_count
        if active[node]:
            return False
        active[node] = 1
        active_count += 1
        for nb in graph.adj[node]:
            if active[nb]:
                continue
            infected_counts[nb] += 1
            if infected_counts[nb] >= graph.threshold[nb] and not queued_next[nb]:
                queued_next[nb] = 1
                next_ready.append(nb)
        return True

    for day in range(base.DAYS):
        next_ready: list[int] = []
        viral = []
        for node in ready:
            if not active[node] and infected_counts[node] >= graph.threshold[node]:
                viral.append(node)

        for node in viral:
            activate(node, next_ready)
        income += base.INCOME_PER_VIRAL * len(viral)
        balance += base.INCOME_PER_VIRAL * len(viral)

        bought = []
        for node in strategy.get(day, []):
            cost = base.COST_PER_NEIGHBOR * graph.degree[node]
            balance -= cost
            costs += cost
            bought.append(node)
            if balance < -1e-9:
                raise ValueError(f"negative balance on day {day}: {balance}")
            activate(node, next_ready)

        for node in ready:
            queued_next[node] = 0
        ready = next_ready

        if return_daily:
            daily.append(
                {
                    "day": day,
                    "viral": len(viral),
                    "bought": [graph.ids[node] for node in bought],
                    "balance": balance,
                    "active": active_count,
                }
            )

    result = {
        "profit": income - costs,
        "income": income,
        "costs": costs,
        "balance": balance,
        "active": active_count,
        "viral": income // base.INCOME_PER_VIRAL,
        "seeds": sum(len(v) for v in strategy.values()),
    }
    if return_daily:
        return result, daily, active
    return result


def fast_repair_to_valid(graph: base.Graph, strategy: dict[int, list[int]]) -> dict[int, list[int]]:
    strategy = base.normalize_strategy(strategy)
    n = len(graph.ids)
    active = bytearray(n)
    infected_counts = [0] * n
    ready: list[int] = []
    queued_next = bytearray(n)
    balance = base.INITIAL_BUDGET
    clean: dict[int, list[int]] = defaultdict(list)
    used = set()

    def activate(node: int, next_ready: list[int]) -> bool:
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

    for day in range(base.DAYS):
        next_ready: list[int] = []
        viral = [node for node in ready if not active[node] and infected_counts[node] >= graph.threshold[node]]
        for node in viral:
            activate(node, next_ready)
        balance += base.INCOME_PER_VIRAL * len(viral)

        for node in strategy.get(day, []):
            cost = base.COST_PER_NEIGHBOR * graph.degree[node]
            if node in used or active[node] or len(clean[day]) >= base.MAX_CONTRACTS_PER_DAY:
                continue
            if cost > balance:
                continue
            clean[day].append(node)
            used.add(node)
            balance -= cost
            activate(node, next_ready)

        for node in ready:
            queued_next[node] = 0
        ready = next_ready
    return dict(clean)


def assert_fast_matches(graph: base.Graph, strategy: dict[int, list[int]]) -> None:
    slow = base.simulate(graph, strategy)
    fast = fast_simulate(graph, strategy)
    if slow != fast:
        raise AssertionError(f"fast simulator mismatch\nslow={slow}\nfast={fast}")


def copy_strategy(strategy: dict[int, list[int]]) -> dict[int, list[int]]:
    return {day: nodes[:] for day, nodes in strategy.items() if nodes}


def evaluate(graph: base.Graph, strategy: dict[int, list[int]]) -> dict[str, int]:
    strategy = base.normalize_strategy(strategy)
    return fast_simulate(graph, strategy)


def can_add(strategy: dict[int, list[int]], day: int, node: int) -> bool:
    if day < 0 or day >= base.DAYS:
        return False
    if len(strategy.get(day, [])) >= base.MAX_CONTRACTS_PER_DAY:
        return False
    return all(node not in nodes for nodes in strategy.values())


def with_added(strategy: dict[int, list[int]], day: int, node: int) -> dict[int, list[int]]:
    trial = copy_strategy(strategy)
    trial.setdefault(day, []).append(node)
    return trial


def with_removed(strategy: dict[int, list[int]], day: int, pos: int) -> dict[int, list[int]]:
    trial = copy_strategy(strategy)
    trial[day].pop(pos)
    if not trial[day]:
        trial.pop(day, None)
    return trial


def final_inactive_components(graph: base.Graph, strategy: dict[int, list[int]]) -> list[list[int]]:
    _, _, active = fast_simulate(graph, strategy, return_daily=True)
    inactive = {node for node, ok in enumerate(active) if not ok}
    seen = set()
    comps = []
    for start in inactive:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in graph.adj[node]:
                if nb in inactive and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def activation_days(graph: base.Graph, strategy: dict[int, list[int]]) -> list[int | None]:
    n = len(graph.ids)
    active = bytearray(n)
    infected_counts = [0] * n
    ready: list[int] = []
    queued_next = bytearray(n)
    days: list[int | None] = [None] * n
    balance = base.INITIAL_BUDGET

    def activate(node: int, day: int, next_ready: list[int]) -> bool:
        if active[node]:
            return False
        active[node] = 1
        days[node] = day
        for nb in graph.adj[node]:
            if active[nb]:
                continue
            infected_counts[nb] += 1
            if infected_counts[nb] >= graph.threshold[nb] and not queued_next[nb]:
                queued_next[nb] = 1
                next_ready.append(nb)
        return True

    for day in range(base.DAYS):
        next_ready: list[int] = []
        viral = [node for node in ready if not active[node] and infected_counts[node] >= graph.threshold[node]]
        for node in viral:
            activate(node, day, next_ready)
        balance += base.INCOME_PER_VIRAL * len(viral)

        for node in strategy.get(day, []):
            cost = base.COST_PER_NEIGHBOR * graph.degree[node]
            balance -= cost
            if balance < -1e-9:
                raise ValueError(f"negative balance on day {day}: {balance}")
            activate(node, day, next_ready)

        for node in ready:
            queued_next[node] = 0
        ready = next_ready
    return days


def cluster_candidates(graph: base.Graph, strategy: dict[int, list[int]], labels: list[int]) -> list[int]:
    comps = final_inactive_components(graph, strategy)
    candidates: set[int] = set()
    comp_size_by_node = {}

    for comp in comps:
        comp_set = set(comp)
        for node in comp:
            comp_size_by_node[node] = len(comp)
        ranked = sorted(
            comp,
            key=lambda node: (
                -sum(1 for nb in graph.adj[node] if nb in comp_set and graph.threshold[nb] <= 3),
                graph.degree[node],
                graph.ids[node],
            ),
        )
        candidates.update(ranked[: min(70, len(ranked))])
        for node in ranked[: min(20, len(ranked))]:
            candidates.update(nb for nb in graph.adj[node] if graph.degree[nb] <= 45)

    label_counts = Counter(labels[node] for comp in comps[:8] for node in comp)
    for label, _ in label_counts.most_common(12):
        members = [node for node, node_label in enumerate(labels) if node_label == label and graph.degree[node] <= 55]
        members.sort(
            key=lambda node: (
                -sum(1 for nb in graph.adj[node] if labels[nb] == label and graph.threshold[nb] <= 4),
                graph.degree[node],
                graph.ids[node],
            )
        )
        candidates.update(members[:120])

    for node in range(len(graph.ids)):
        if graph.degree[node] <= 12:
            candidates.add(node)

    def score(node: int) -> tuple[float, int]:
        label = labels[node]
        same = sum(1 for nb in graph.adj[node] if labels[nb] == label)
        weak = sum(1 for nb in graph.adj[node] if graph.threshold[nb] <= 3)
        inactive_bonus = comp_size_by_node.get(node, 0)
        value = inactive_bonus * 2.8 + same * 1.6 + weak * 5.0 - graph.degree[node] * 1.25
        return (value, -graph.degree[node])

    return sorted(candidates, key=score, reverse=True)


def limited_candidates(candidates: list[int], limit: int) -> list[int]:
    return candidates[:limit] if limit > 0 else candidates


def best_single_addition(
    graph: base.Graph,
    strategy: dict[int, list[int]],
    candidates: list[int],
    days: range,
) -> tuple[int, int, dict[str, int]] | None:
    base_result = fast_simulate(graph, strategy)
    best = None
    used = {node for nodes in strategy.values() for node in nodes}
    active_day = activation_days(graph, strategy)
    for day in days:
        if len(strategy.get(day, [])) >= base.MAX_CONTRACTS_PER_DAY:
            continue
        for node in candidates:
            if node in used:
                continue
            if active_day[node] is not None and active_day[node] <= day:
                continue
            trial = with_added(strategy, day, node)
            try:
                result = fast_simulate(graph, trial)
            except ValueError:
                continue
            delta = result["profit"] - base_result["profit"]
            if delta > 0 and (best is None or delta > best[2]["profit"] - base_result["profit"]):
                best = (day, node, result)
    return best


def greedy_single_additions(
    graph: base.Graph,
    strategy: dict[int, list[int]],
    candidates: list[int],
    max_rounds: int,
    max_day: int,
) -> dict[int, list[int]]:
    current = copy_strategy(strategy)
    for round_idx in range(max_rounds):
        found = best_single_addition(graph, current, candidates, range(max_day + 1))
        if found is None:
            break
        day, node, result = found
        current.setdefault(day, []).append(node)
        print(
            json.dumps(
                {
                    "op": "add",
                    "round": round_idx,
                    "day": day,
                    "node": graph.ids[node],
                    "degree": graph.degree[node],
                    "result": result,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return current


def greedy_remove_bad(graph: base.Graph, strategy: dict[int, list[int]]) -> dict[int, list[int]]:
    current = copy_strategy(strategy)
    improved = True
    while improved:
        improved = False
        base_result = fast_simulate(graph, current)
        best_trial = None
        best_result = base_result
        for day, nodes in list(current.items()):
            for pos, _node in enumerate(nodes):
                trial = with_removed(current, day, pos)
                try:
                    result = fast_simulate(graph, trial)
                except ValueError:
                    continue
                if result["profit"] > best_result["profit"]:
                    best_trial = trial
                    best_result = result
        if best_trial is not None:
            current = best_trial
            improved = True
            print(json.dumps({"op": "remove", "result": best_result}, ensure_ascii=False), flush=True)
    return current


def shift_search(graph: base.Graph, strategy: dict[int, list[int]], window: int = 12) -> dict[int, list[int]]:
    current = copy_strategy(strategy)
    improved = True
    while improved:
        improved = False
        base_result = fast_simulate(graph, current)
        best = None
        for old_day, nodes in list(current.items()):
            for pos, node in enumerate(nodes):
                for new_day in range(max(0, old_day - window), min(base.DAYS - 1, old_day + window) + 1):
                    if new_day == old_day or len(current.get(new_day, [])) >= base.MAX_CONTRACTS_PER_DAY:
                        continue
                    trial = with_removed(current, old_day, pos)
                    if len(trial.get(new_day, [])) >= base.MAX_CONTRACTS_PER_DAY:
                        continue
                    trial.setdefault(new_day, []).append(node)
                    try:
                        result = fast_simulate(graph, trial)
                    except ValueError:
                        continue
                    if result["profit"] > base_result["profit"] and (
                        best is None or result["profit"] > best[3]["profit"]
                    ):
                        best = (old_day, new_day, node, result, trial)
        if best is not None:
            old_day, new_day, node, result, current = best
            improved = True
            print(
                json.dumps(
                    {
                        "op": "shift",
                        "node": graph.ids[node],
                        "from": old_day,
                        "to": new_day,
                        "result": result,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return current


def random_pack_from_cluster(
    graph: base.Graph,
    labels: list[int],
    label: int,
    rng: random.Random,
    max_cost: int,
    max_size: int,
) -> set[int]:
    members = [node for node, node_label in enumerate(labels) if node_label == label and graph.degree[node] <= 70]
    if not members:
        return set()

    def weight(node: int) -> float:
        same = sum(1 for nb in graph.adj[node] if labels[nb] == label)
        weak = sum(1 for nb in graph.adj[node] if graph.threshold[nb] <= 3)
        return max(0.2, same * 2.0 + weak * 4.0 - graph.degree[node] * 1.2)

    ordered = sorted(members, key=lambda node: rng.random() / weight(node))
    seeds = set()
    cost = 0
    for node in ordered:
        node_cost = base.COST_PER_NEIGHBOR * graph.degree[node]
        if len(seeds) >= max_size:
            break
        if cost + node_cost <= max_cost:
            seeds.add(node)
            cost += node_cost
    return seeds


def anneal(
    graph: base.Graph,
    strategy: dict[int, list[int]],
    candidates: list[int],
    labels: list[int],
    iterations: int,
    seed: int,
    max_day: int,
) -> dict[int, list[int]]:
    rng = random.Random(seed)
    labels_by_size = [label for label, _ in Counter(labels).most_common(80)]
    current = fast_repair_to_valid(graph, strategy)
    current_result = fast_simulate(graph, current)
    best = copy_strategy(current)
    best_result = current_result
    node_weights = [1.0 / math.sqrt(i + 1) for i in range(len(candidates))]

    for step in range(iterations):
        trial = copy_strategy(current)
        used = {node for nodes in trial.values() for node in nodes}
        op = rng.random()

        if op < 0.18:
            day = rng.randint(0, max_day)
            if len(trial.get(day, [])) < base.MAX_CONTRACTS_PER_DAY:
                for _ in range(40):
                    node = rng.choices(candidates, weights=node_weights, k=1)[0]
                    if node not in used:
                        trial.setdefault(day, []).append(node)
                        break
        elif op < 0.34 and used:
            day = rng.choice([day for day, nodes in trial.items() if nodes])
            trial[day].pop(rng.randrange(len(trial[day])))
            if not trial[day]:
                trial.pop(day, None)
        elif op < 0.56 and used:
            day = rng.choice([day for day, nodes in trial.items() if nodes])
            pos = rng.randrange(len(trial[day]))
            for _ in range(40):
                node = rng.choices(candidates, weights=node_weights, k=1)[0]
                if node not in used or node == trial[day][pos]:
                    trial[day][pos] = node
                    break
        elif op < 0.76 and used:
            old_day = rng.choice([day for day, nodes in trial.items() if nodes])
            pos = rng.randrange(len(trial[old_day]))
            node = trial[old_day].pop(pos)
            if not trial[old_day]:
                trial.pop(old_day, None)
            new_day = max(0, min(max_day, old_day + rng.choice([-15, -9, -5, -3, -2, -1, 1, 2, 3, 5, 9, 15])))
            if len(trial.get(new_day, [])) < base.MAX_CONTRACTS_PER_DAY:
                trial.setdefault(new_day, []).append(node)
        else:
            label = rng.choice(labels_by_size)
            pack = random_pack_from_cluster(
                graph,
                labels,
                label,
                rng,
                max_cost=rng.choice([9000, 15000, 30000, 60000]),
                max_size=rng.randint(2, 10),
            )
            if pack:
                day = rng.randint(0, max_day)
                slots = base.MAX_CONTRACTS_PER_DAY - len(trial.get(day, []))
                additions = [node for node in pack if node not in used]
                rng.shuffle(additions)
                if slots > 0:
                    trial.setdefault(day, []).extend(additions[:slots])

        trial = fast_repair_to_valid(graph, trial)
        try:
            result = fast_simulate(graph, trial)
        except ValueError:
            continue

        delta = result["profit"] - current_result["profit"]
        temp = max(35.0, 4200.0 * (1.0 - step / max(1, iterations)) ** 1.55)
        if delta >= 0 or rng.random() < math.exp(delta / temp):
            current = trial
            current_result = result
            if result["profit"] > best_result["profit"]:
                best = copy_strategy(trial)
                best_result = result
                print(
                    json.dumps(
                        {
                            "op": "anneal_best",
                            "step": step,
                            "result": best_result,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, default=Path("marketing_edges.txt"))
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--max-day", type=int, default=52)
    parser.add_argument("--single-rounds", type=int, default=4)
    parser.add_argument("--candidate-limit", type=int, default=500)
    args = parser.parse_args()

    graph = base.read_graph(args.edges)
    strategy = base.read_submission(graph, args.start)
    assert_fast_matches(graph, strategy)
    labels = base.label_propagation(graph, seed=args.seed % 1000)
    all_candidates = cluster_candidates(graph, strategy, labels)
    candidates = limited_candidates(all_candidates, args.candidate_limit)
    print(
        json.dumps(
            {
                "start": fast_simulate(graph, strategy),
                "candidate_count": len(candidates),
                "candidate_count_all": len(all_candidates),
                "inactive_components": [len(comp) for comp in final_inactive_components(graph, strategy)[:12]],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    strategy = greedy_remove_bad(graph, strategy)
    strategy = shift_search(graph, strategy)
    strategy = greedy_single_additions(graph, strategy, candidates, args.single_rounds, args.max_day)
    strategy = anneal(graph, strategy, candidates, labels, args.iterations, args.seed, args.max_day)
    strategy = greedy_remove_bad(graph, strategy)
    strategy = shift_search(graph, strategy)
    strategy = greedy_single_additions(graph, strategy, candidates, 2, args.max_day)
    result, daily, _ = fast_simulate(graph, strategy, return_daily=True)
    base.write_submission(graph, strategy, args.out)
    report = {
        "result": result,
        "contracts": {day: [graph.ids[n] for n in strategy.get(day, [])] for day in range(base.DAYS) if strategy.get(day)},
        "daily_nonzero": [row for row in daily if row["viral"] or row["bought"]],
    }
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["result"], ensure_ascii=False), flush=True)
    print(f"wrote {args.out} and {args.out.with_suffix('.report.json')}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
