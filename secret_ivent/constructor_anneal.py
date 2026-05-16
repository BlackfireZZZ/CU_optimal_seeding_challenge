#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import local_improve as li
import solve_marketing as base


def clone(strategy: dict[int, list[int]]) -> dict[int, list[int]]:
    return {day: nodes[:] for day, nodes in strategy.items() if nodes}


def used_nodes(strategy: dict[int, list[int]]) -> set[int]:
    return {node for nodes in strategy.values() for node in nodes}


def build_communities(graph: base.Graph, seeds: list[int]) -> list[dict]:
    communities = []
    seen_keys = set()
    for seed in seeds:
        labels = base.label_propagation(graph, iterations=60, seed=seed)
        groups = defaultdict(list)
        for node, label in enumerate(labels):
            groups[label].append(node)
        for label, members in groups.items():
            if len(members) < 3:
                continue
            member_set = set(members)
            key = tuple(sorted(members))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            weak = {
                node: sum(1 for nb in graph.adj[node] if graph.threshold[nb] <= 3)
                for node in members
            }
            internal = {
                node: sum(1 for nb in graph.adj[node] if nb in member_set)
                for node in members
            }
            ranked = sorted(
                members,
                key=lambda node: (
                    -internal[node] * 2 - weak[node] * 4 + graph.degree[node] * 1.25,
                    graph.degree[node],
                    graph.ids[node],
                ),
            )
            cheap = sorted(
                [node for node in members if graph.degree[node] <= 80],
                key=lambda node: (graph.degree[node], -internal[node], graph.ids[node]),
            )
            communities.append(
                {
                    "seed": seed,
                    "label": label,
                    "size": len(members),
                    "members": members,
                    "ranked": ranked[:700],
                    "cheap": cheap[:500],
                    "internal": internal,
                    "weak": weak,
                }
            )
    communities.sort(key=lambda c: c["size"], reverse=True)
    return communities


def component_communities(graph: base.Graph, strategy: dict[int, list[int]]) -> list[dict]:
    comps = li.final_inactive_components(graph, strategy)
    out = []
    for idx, comp in enumerate(comps):
        if len(comp) < 3:
            continue
        comp_set = set(comp)
        internal = {node: sum(1 for nb in graph.adj[node] if nb in comp_set) for node in comp}
        weak = {node: sum(1 for nb in graph.adj[node] if graph.threshold[nb] <= 3) for node in comp}
        ranked = sorted(
            comp,
            key=lambda node: (-internal[node] * 2 - weak[node] * 4 + graph.degree[node] * 1.25, graph.degree[node]),
        )
        cheap = sorted([node for node in comp if graph.degree[node] <= 80], key=lambda node: (graph.degree[node], -internal[node]))
        out.append(
            {
                "seed": 10_000 + idx,
                "label": idx,
                "size": len(comp),
                "members": comp,
                "ranked": ranked[:500],
                "cheap": cheap[:500],
                "internal": internal,
                "weak": weak,
            }
        )
    return out


def weighted_pack(graph: base.Graph, community: dict, rng: random.Random, used: set[int], slots: int, cost_cap: int) -> list[int]:
    if slots <= 0:
        return []
    pool = [node for node in community["ranked"] if node not in used and graph.degree[node] <= 120]
    if not pool:
        return []

    def weight(node: int) -> float:
        return max(
            0.1,
            community["internal"].get(node, 0) * 2.0
            + community["weak"].get(node, 0) * 4.5
            - graph.degree[node] * 1.2,
        )

    target = rng.randint(1, min(slots, 10, len(pool)))
    if rng.random() < 0.2:
        ordered = [node for node in community["cheap"] if node in pool]
    else:
        ordered = sorted(pool, key=lambda node: rng.random() / weight(node))
    pack = []
    cost = 0
    for node in ordered:
        if len(pack) >= target:
            break
        node_cost = base.COST_PER_NEIGHBOR * graph.degree[node]
        if cost + node_cost <= cost_cap:
            pack.append(node)
            cost += node_cost
    return pack


def repair_fast(graph: base.Graph, strategy: dict[int, list[int]]) -> dict[int, list[int]]:
    return li.fast_repair_to_valid(graph, strategy)


def mutate(
    graph: base.Graph,
    strategy: dict[int, list[int]],
    communities: list[dict],
    rng: random.Random,
    max_day: int,
) -> dict[int, list[int]]:
    trial = clone(strategy)
    op = rng.random()
    nonempty = [day for day, nodes in trial.items() if nodes]
    used = used_nodes(trial)

    if op < 0.18 and nonempty:
        day = rng.choice(nonempty)
        pos = rng.randrange(len(trial[day]))
        trial[day].pop(pos)
        if not trial[day]:
            trial.pop(day, None)
    elif op < 0.38 and nonempty:
        day = rng.choice(nonempty)
        remove_count = rng.randint(1, min(len(trial[day]), 4))
        removed = rng.sample(trial[day], remove_count)
        trial[day] = [node for node in trial[day] if node not in removed]
        if not trial[day]:
            trial.pop(day, None)
        used = used_nodes(trial)
        comm = rng.choice(communities)
        old_cost = sum(base.COST_PER_NEIGHBOR * graph.degree[node] for node in removed)
        cap = max(300, old_cost + rng.choice([-9000, -6000, -3000, 0, 3000, 9000, 18000]))
        pack = weighted_pack(graph, comm, rng, used, base.MAX_CONTRACTS_PER_DAY - len(trial.get(day, [])), cap)
        if pack:
            trial.setdefault(day, []).extend(pack)
    elif op < 0.58:
        day = rng.randint(0, max_day)
        used = used_nodes(trial)
        comm = rng.choice(communities)
        cap = rng.choice([600, 1200, 2400, 4800, 9000, 15000, 30000])
        pack = weighted_pack(graph, comm, rng, used, base.MAX_CONTRACTS_PER_DAY - len(trial.get(day, [])), cap)
        if pack:
            trial.setdefault(day, []).extend(pack)
    elif op < 0.78 and nonempty:
        old_day = rng.choice(nonempty)
        node = trial[old_day].pop(rng.randrange(len(trial[old_day])))
        if not trial[old_day]:
            trial.pop(old_day, None)
        new_day = max(0, min(max_day, old_day + rng.choice([-12, -8, -5, -3, -2, -1, 1, 2, 3, 5, 8, 12])))
        if len(trial.get(new_day, [])) < base.MAX_CONTRACTS_PER_DAY:
            trial.setdefault(new_day, []).append(node)
    elif nonempty:
        # Replace one node by a small cheap constructor from a community containing or near it.
        day = rng.choice(nonempty)
        old = rng.choice(trial[day])
        trial[day] = [node for node in trial[day] if node != old]
        if not trial[day]:
            trial.pop(day, None)
        used = used_nodes(trial)
        related = [
            c for c in communities[:250]
            if old in c["members"] or any(nb in c["members"] for nb in graph.adj[old][:20])
        ]
        comm = rng.choice(related or communities)
        cap = max(300, base.COST_PER_NEIGHBOR * graph.degree[old] + rng.choice([-6000, -3000, 0, 3000, 9000]))
        pack = weighted_pack(graph, comm, rng, used, base.MAX_CONTRACTS_PER_DAY - len(trial.get(day, [])), cap)
        if pack:
            trial.setdefault(day, []).extend(pack)

    return repair_fast(graph, trial)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, default=Path("marketing_edges.txt"))
    parser.add_argument("--start", type=Path, default=Path("best_sub.csv"))
    parser.add_argument("--out", type=Path, default=Path("secret_ivent/constructor_anneal_best.csv"))
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--max-day", type=int, default=52)
    args = parser.parse_args()

    graph = base.read_graph(args.edges)
    start = base.read_submission(graph, args.start)
    rng = random.Random(args.seed)
    communities = build_communities(graph, [3, 7, 11, 17, 23, 31, 43, 59, 79, 101, 131, 167])
    communities.extend(component_communities(graph, start))
    communities.sort(key=lambda c: c["size"], reverse=True)
    communities = communities[:520]

    current = repair_fast(graph, start)
    current_result = li.fast_simulate(graph, current)
    best = clone(current)
    best_result = dict(current_result)
    print(json.dumps({"start": best_result, "communities": len(communities)}, ensure_ascii=False), flush=True)

    for step in range(args.iterations):
        trial = mutate(graph, current, communities, rng, args.max_day)
        try:
            result = li.fast_simulate(graph, trial)
        except ValueError:
            continue
        delta = result["profit"] - current_result["profit"]
        temp = max(20.0, 6500.0 * (1.0 - step / max(1, args.iterations)) ** 1.7)
        if delta >= 0 or rng.random() < math.exp(delta / temp):
            current = trial
            current_result = result
            if result["profit"] > best_result["profit"]:
                best = clone(trial)
                best_result = dict(result)
                print(
                    json.dumps(
                        {"step": step, "best": best_result, "contracts": {
                            day: [graph.ids[n] for n in best.get(day, [])]
                            for day in range(base.DAYS)
                            if best.get(day)
                        }},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                base.write_submission(graph, best, args.out)
                base.write_submission(graph, best, Path("best_sub.csv"))
                report = {
                    "result": best_result,
                    "contracts": {
                        day: [graph.ids[n] for n in best.get(day, [])]
                        for day in range(base.DAYS)
                        if best.get(day)
                    },
                }
                args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                if best_result["profit"] >= 146000:
                    break

    base.write_submission(graph, best, args.out)
    base.write_submission(graph, best, Path("best_sub.csv"))
    result, daily = base.simulate(graph, best, return_daily=True)
    for out in [args.out, Path("best_sub.csv"), Path("secret_ivent/best_sub.csv")]:
        base.write_submission(graph, best, out)
        out.with_suffix(".report.json").write_text(
            json.dumps(
                {
                    "result": result,
                    "contracts": {
                        day: [graph.ids[n] for n in best.get(day, [])]
                        for day in range(base.DAYS)
                        if best.get(day)
                    },
                    "daily_nonzero": [row for row in daily if row["viral"] or row["bought"]],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps({"final": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
