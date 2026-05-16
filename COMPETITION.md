# Network Influence Profit Challenge

## Competition Goal

Plan a marketing campaign for a new pocket device in a social network.

You can sign advertising contracts with people. After that, the product spreads through the network: if a person has enough friends already using the device, they start using it too the next day.

**Objective:** Maximize total campaign profit over 60 days.

You must decide:
- with whom to sign contracts
- on which days
- when it's better to wait

---

## Problem Description

You are running a marketing campaign for a new pocket device.

The society is modeled as an **undirected social network**. Nodes = people, edges = acquaintances. The graph is based on the **SNAP Facebook network**.

Initially, you can sign advertising contracts with selected people. That person immediately starts using the product and promotes it among their friends.

**Contract cost depends on number of friends:**
$$cost(i) = 300 \times N(i)$$
where $N(i)$ = number of friends of person $i$ in the graph.

---

## Viral Spread Mechanics

Each day the product can spread through the network.

**Rule:** If at least **18%** of a person's friends already use the product, that person starts using it **the next day**.

**Revenue:** **50 rubles** for each new **virally acquired** user.

> Important: people with whom you signed contracts do NOT directly generate revenue.

---

## Campaign Parameters

| Parameter | Value |
|-----------|-------|
| Initial budget | 10,000 rubles |
| Contract cost | 300 × degree(i) rubles |
| Income per viral user | 50 rubles |
| Viral threshold | 18% of friends |
| Max contracts per day | 10 |
| Campaign duration | 60 days |

**Budget constraint:** Cannot sign contracts if insufficient funds at that moment. Revenue from viral users can be used in subsequent days.

---

## Scoring

$$\text{profit} = \text{income} - \text{costs\_contracts}$$

where:
- `income` = revenue from virally acquired users
- `costs_contracts` = total cost of all advertising contracts

Starting budget (10,000 rubles) is used only to validate the strategy is feasible — it is **not** counted in the final profit.

Leaderboard is sorted by descending profit.

---

## Submission Format

CSV file with columns:

```csv
day,node_ids
0,3057 1270
1,-1
2,-1
...
59,-1
```

- `node_ids`: space-separated list of node IDs for that day's contracts
- `-1`: no contracts on this day
- Each day: at most **10** contracts

---

## Baseline Example

```python
initial_budget = 10000
contract_cost_per_neighbor = 300
income_per_affected = 50
max_contracts_per_day = 10
campaign_duration = 60
threshold = 0.18

G = nx.read_edgelist('marketing_edges.txt')  # undirected graph

# Viral spread simulation (one step):
def update_affected(G):
    for node in non_affected:
        neighbors = list(G.neighbors(node))
        affected_neighbors = sum(statuses[n] for n in neighbors)
        if affected_neighbors / len(neighbors) >= threshold:
            G.nodes[node]['status'] = True
            new_affected.append(node)
    return new_affected
```

Baseline example `{0: [1,2], 1: [117]}` yields **Profit = -8900**.

---

## Graph Data

- **File:** `data/marketing_edges.txt`
- **Format:** space-separated edges `node1 node2`, one per line
- **Edges:** 84,070
- **Nodes:** 3,953
- **Type:** Undirected (SNAP Facebook network)

**Baseline notebook:** `Marketing_Campaign.ipynb`
