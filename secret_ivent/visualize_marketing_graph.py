#!/usr/bin/env python3
"""Generate an interactive HTML visualization for marketing_edges.txt.

The output is dependency-free and embeds all nodes/edges into a canvas app.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc949",
    "#af7aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
    "#2f6f73",
    "#b07aa1",
    "#8cd17d",
    "#fabfd2",
]


def read_edges(path: Path) -> tuple[list[int], list[tuple[int, int]], dict[int, set[int]]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    edges: list[tuple[int, int]] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                a_raw, b_raw = line.split()[:2]
                a, b = int(a_raw), int(b_raw)
            except ValueError as exc:
                raise ValueError(f"Cannot parse edge at line {line_no}: {line!r}") from exc
            if a == b:
                continue
            edges.append((a, b))
            adjacency[a].add(b)
            adjacency[b].add(a)

    nodes = sorted(adjacency)
    return nodes, edges, adjacency


def connected_components(nodes: list[int], adjacency: dict[int, set[int]]) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in nodes:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    components.sort(key=len, reverse=True)
    return components


def label_propagation(
    nodes: list[int],
    adjacency: dict[int, set[int]],
    iterations: int = 18,
    seed: int = 7,
) -> dict[int, int]:
    rng = random.Random(seed)
    labels = {node: node for node in nodes}
    order = nodes[:]

    for _ in range(iterations):
        rng.shuffle(order)
        changed = 0
        for node in order:
            counts = Counter(labels[neighbor] for neighbor in adjacency[node])
            if not counts:
                continue
            best_count = max(counts.values())
            best_labels = [label for label, count in counts.items() if count == best_count]
            best = min(best_labels)
            if labels[node] != best:
                labels[node] = best
                changed += 1
        if changed == 0:
            break

    communities: dict[int, list[int]] = defaultdict(list)
    for node, label in labels.items():
        communities[label].append(node)

    ordered_labels = [
        label
        for label, members in sorted(
            communities.items(), key=lambda item: (-len(item[1]), min(item[1]))
        )
    ]
    remap = {label: index for index, label in enumerate(ordered_labels)}
    return {node: remap[label] for node, label in labels.items()}


def layout_nodes(
    nodes: list[int],
    adjacency: dict[int, set[int]],
    community_by_node: dict[int, int],
) -> dict[int, dict[str, float | int | str]]:
    communities: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        communities[community_by_node[node]].append(node)

    ordered = sorted(communities.items(), key=lambda item: (-len(item[1]), item[0]))
    total = len(nodes)
    outer_radius = 250 + 18 * math.sqrt(total)
    positions: dict[int, dict[str, float | int | str]] = {}

    for index, (community, members) in enumerate(ordered):
        angle = 2 * math.pi * index / max(1, len(ordered))
        ring = outer_radius * (0.55 + 0.45 * ((index % 4) / 3 if len(ordered) > 3 else 1))
        cx = math.cos(angle) * ring
        cy = math.sin(angle) * ring
        members.sort(key=lambda n: (-len(adjacency[n]), n))
        local_radius = max(34.0, 8.0 * math.sqrt(len(members)))

        for local_index, node in enumerate(members):
            theta = 2 * math.pi * local_index / max(1, len(members))
            shell = 0.35 + 0.65 * math.sqrt((local_index + 1) / max(1, len(members)))
            jitter = ((node * 1103515245 + 12345) % 1000) / 1000 - 0.5
            x = cx + math.cos(theta + jitter * 0.25) * local_radius * shell
            y = cy + math.sin(theta + jitter * 0.25) * local_radius * shell
            degree = len(adjacency[node])
            positions[node] = {
                "id": node,
                "x": round(x, 3),
                "y": round(y, 3),
                "degree": degree,
                "community": community,
                "color": PALETTE[community % len(PALETTE)],
            }

    return positions


def html_template(nodes_json: str, edges_json: str, stats_json: str) -> str:
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Marketing graph</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f7f3;
      color: #202124;
    }}
    body {{
      margin: 0;
      overflow: hidden;
    }}
    #graph {{
      display: block;
      width: 100vw;
      height: 100vh;
      background: #fbfaf6;
      cursor: grab;
    }}
    #graph:active {{ cursor: grabbing; }}
    .panel {{
      position: fixed;
      left: 16px;
      top: 16px;
      display: grid;
      gap: 10px;
      width: min(360px, calc(100vw - 32px));
      padding: 14px;
      border: 1px solid #d9d7cf;
      border-radius: 8px;
      background: rgba(255, 255, 252, 0.94);
      box-shadow: 0 12px 34px rgba(39, 35, 27, 0.12);
      backdrop-filter: blur(8px);
    }}
    .title {{
      font-size: 15px;
      font-weight: 720;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }}
    .stat {{
      border: 1px solid #e4e1d8;
      border-radius: 6px;
      padding: 8px;
      background: #fff;
    }}
    .stat strong {{
      display: block;
      font-size: 17px;
    }}
    .stat span {{
      color: #69665e;
      font-size: 12px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 8px;
    }}
    input, button {{
      height: 34px;
      border: 1px solid #cbc8bf;
      border-radius: 6px;
      background: #fff;
      color: #202124;
      font: inherit;
      font-size: 13px;
    }}
    input {{
      min-width: 0;
      padding: 0 10px;
    }}
    button {{
      padding: 0 10px;
      cursor: pointer;
    }}
    button:hover {{ background: #f0eee7; }}
    .hint {{
      color: #69665e;
      font-size: 12px;
      line-height: 1.35;
    }}
    #status {{
      font-size: 13px;
      min-height: 18px;
    }}
  </style>
</head>
<body>
  <canvas id="graph"></canvas>
  <section class="panel">
    <div class="title">Marketing graph: все связи</div>
    <div class="stats">
      <div class="stat"><strong id="nodeCount"></strong><span>узлов</span></div>
      <div class="stat"><strong id="edgeCount"></strong><span>связей</span></div>
      <div class="stat"><strong id="communityCount"></strong><span>групп</span></div>
    </div>
    <div class="controls">
      <input id="search" type="search" placeholder="ID узла, например 236">
      <button id="fit">Центр</button>
      <button id="toggle">Связи</button>
    </div>
    <div id="status"></div>
    <div class="hint">Колесо мыши масштабирует, перетаскивание двигает граф. Наведите на узел или введите ID, чтобы подсветить его соседей.</div>
  </section>
  <script>
    const nodes = {nodes_json};
    const edges = {edges_json};
    const stats = {stats_json};
    const canvas = document.getElementById('graph');
    const ctx = canvas.getContext('2d');
    const byId = new Map(nodes.map((node, index) => [node.id, {{...node, index}}]));
    const neighborMap = new Map(nodes.map(node => [node.id, new Set()]));
    for (const [a, b] of edges) {{
      neighborMap.get(a)?.add(b);
      neighborMap.get(b)?.add(a);
    }}

    let width = 0;
    let height = 0;
    let scale = 1;
    let tx = 0;
    let ty = 0;
    let hovered = null;
    let selected = null;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let showEdges = true;

    const search = document.getElementById('search');
    const status = document.getElementById('status');
    document.getElementById('nodeCount').textContent = stats.nodes.toLocaleString('ru-RU');
    document.getElementById('edgeCount').textContent = stats.edges.toLocaleString('ru-RU');
    document.getElementById('communityCount').textContent = stats.communities.toLocaleString('ru-RU');

    function resize() {{
      const ratio = window.devicePixelRatio || 1;
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      canvas.style.width = width + 'px';
      canvas.style.height = height + 'px';
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }}

    function fit() {{
      const xs = nodes.map(n => n.x);
      const ys = nodes.map(n => n.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const graphW = maxX - minX || 1;
      const graphH = maxY - minY || 1;
      scale = Math.min((width - 80) / graphW, (height - 80) / graphH);
      scale = Math.max(0.08, Math.min(scale, 4));
      tx = width / 2 - ((minX + maxX) / 2) * scale;
      ty = height / 2 - ((minY + maxY) / 2) * scale;
      draw();
    }}

    function screenToWorld(x, y) {{
      return {{x: (x - tx) / scale, y: (y - ty) / scale}};
    }}

    function nodeAt(x, y) {{
      const p = screenToWorld(x, y);
      let best = null;
      let bestDistance = 10 / scale;
      for (const node of nodes) {{
        const dx = node.x - p.x;
        const dy = node.y - p.y;
        const distance = Math.hypot(dx, dy);
        const radius = 2.1 + Math.sqrt(node.degree) * 0.22;
        if (distance <= radius + bestDistance) {{
          best = node;
          bestDistance = distance;
        }}
      }}
      return best;
    }}

    function activeNode() {{
      return selected || hovered;
    }}

    function draw() {{
      ctx.clearRect(0, 0, width, height);
      ctx.save();
      ctx.translate(tx, ty);
      ctx.scale(scale, scale);
      ctx.lineCap = 'round';

      const active = activeNode();
      const activeNeighbors = active ? neighborMap.get(active.id) : null;

      if (showEdges) {{
        ctx.lineWidth = Math.max(0.12, 0.55 / Math.sqrt(scale));
        ctx.globalAlpha = active ? 0.06 : 0.075;
        ctx.strokeStyle = '#6f7480';
        ctx.beginPath();
        for (const [a, b] of edges) {{
          if (active && !(a === active.id || b === active.id)) continue;
          const source = byId.get(a);
          const target = byId.get(b);
          ctx.moveTo(source.x, source.y);
          ctx.lineTo(target.x, target.y);
        }}
        ctx.stroke();
      }}

      for (const node of nodes) {{
        const isActive = active && node.id === active.id;
        const isNeighbor = activeNeighbors && activeNeighbors.has(node.id);
        const radius = (isActive ? 5.5 : 2.2) + Math.sqrt(node.degree) * (isActive ? 0.45 : 0.18);
        ctx.globalAlpha = active ? (isActive || isNeighbor ? 1 : 0.18) : 0.9;
        ctx.fillStyle = isActive ? '#111827' : node.color;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }}

      if (active) {{
        ctx.globalAlpha = 1;
        ctx.fillStyle = '#111827';
        ctx.font = `${{Math.max(11, 13 / scale)}}px Inter, system-ui, sans-serif`;
        ctx.fillText(`ID ${{active.id}}`, active.x + 9 / scale, active.y - 9 / scale);
      }}

      ctx.restore();
      const shown = active
        ? `Узел ${{active.id}}: степень ${{active.degree}}, соседей ${{neighborMap.get(active.id).size}}, группа ${{active.community}}`
        : 'Показаны все ребра из marketing_edges.txt';
      status.textContent = shown;
    }}

    canvas.addEventListener('mousedown', event => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
    }});
    window.addEventListener('mouseup', () => dragging = false);
    window.addEventListener('mousemove', event => {{
      if (dragging) {{
        tx += event.clientX - lastX;
        ty += event.clientY - lastY;
        lastX = event.clientX;
        lastY = event.clientY;
        draw();
        return;
      }}
      hovered = nodeAt(event.clientX, event.clientY);
      draw();
    }});
    canvas.addEventListener('click', event => {{
      selected = nodeAt(event.clientX, event.clientY);
      if (selected) search.value = selected.id;
      draw();
    }});
    canvas.addEventListener('wheel', event => {{
      event.preventDefault();
      const before = screenToWorld(event.clientX, event.clientY);
      const factor = event.deltaY < 0 ? 1.12 : 0.89;
      scale = Math.max(0.04, Math.min(14, scale * factor));
      tx = event.clientX - before.x * scale;
      ty = event.clientY - before.y * scale;
      draw();
    }}, {{passive: false}});
    search.addEventListener('input', () => {{
      const value = Number(search.value.trim());
      selected = Number.isFinite(value) && byId.has(value) ? byId.get(value) : null;
      if (selected) {{
        scale = Math.max(scale, 0.95);
        tx = width / 2 - selected.x * scale;
        ty = height / 2 - selected.y * scale;
      }}
      draw();
    }});
    document.getElementById('fit').addEventListener('click', fit);
    document.getElementById('toggle').addEventListener('click', () => {{
      showEdges = !showEdges;
      draw();
    }});
    window.addEventListener('resize', resize);
    resize();
    fit();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path(__file__).with_name("marketing_edges.txt"),
        help="Path to marketing_edges.txt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("marketing_graph.html"),
        help="Output HTML file",
    )
    args = parser.parse_args()

    nodes, edges, adjacency = read_edges(args.edges)
    components = connected_components(nodes, adjacency)
    communities = label_propagation(nodes, adjacency)
    positioned_nodes = layout_nodes(nodes, adjacency, communities)
    rows = [positioned_nodes[node] for node in nodes]
    community_count = len({node["community"] for node in rows})
    stats = {
        "nodes": len(nodes),
        "edges": len(edges),
        "components": len(components),
        "largest_component": len(components[0]) if components else 0,
        "communities": community_count,
    }

    args.out.write_text(
        html_template(
            json.dumps(rows, separators=(",", ":")),
            json.dumps(edges, separators=(",", ":")),
            json.dumps(stats, separators=(",", ":")),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
