---
title: Thermal Chaos Arena
emoji: 🔥
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Thermal Chaos Arena

Thermal Chaos Arena is an original AI infrastructure-control arena for testing whether an agent can operate a fragile GPU cluster without greedily causing collapse.

The agent must run critical AI jobs on a 10-node, 2-zone GPU cluster while managing:

- heat accumulation
- limited energy
- hidden hardware wear
- random cooling failures
- cascading node failures
- job deadlines
- maintenance tradeoffs
- reward-hacking guardrails

The core question:

> Can an AI learn not to be greedy when short-term throughput creates long-term infrastructure failure?

## Why this project

Modern AI systems depend on GPU clusters, but most demos treat compute as infinite. I wanted to build a small world where compute is fragile. The interesting behavior is not “schedule every job immediately”; it is knowing when to inspect, cool, throttle, repair, or even sacrifice a node to prevent a cascade.

## What is technically happening

Each episode is a partially observable control problem:

- The agent sees temperatures, free GPUs, job deadlines, energy, and visible node status.
- It does **not** directly see hidden hardware wear unless it spends an action inspecting.
- Heat and load increase wear over time.
- Chaos events can reduce cooling or spike hidden wear.
- Failed nodes push thermal stress onto neighbours, which can trigger cascades.
- Guardrails penalize risky/reward-hacking behavior.

## Actions

The simulator supports eight action types:

- `PLACE_JOB`
- `COOL_ZONE`
- `THROTTLE_NODE`
- `INSPECT_NODE`
- `REPAIR_NODE`
- `MIGRATE_JOB`
- `SACRIFICE_NODE`
- `WAIT`

## Built-in agents

- **Greedy Throughput** — tries to complete high-value jobs immediately, often causing collapse.
- **Balanced Operator** — avoids very hot nodes and uses cooling when needed.
- **Thermal SRE** — inspects, throttles, cools, and sacrifices nodes to prevent cascades.

## Demo features

- Run a full episode in the browser.
- Compare baseline agents on the same seed and chaos level.
- Export replay JSON.
- Inspect any replay frame to see how decisions produced collapse or survival.

## What I chose to cut

I cut real RL training, Kubernetes integration, accounts, and persistent storage. The smallest interesting version is the arena itself: a working control environment with hidden state, chaos, baselines, scoring, and replay. That proves the idea without over-scoping.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Activate AI Fellows write-up draft

I built Thermal Chaos Arena, an AI infrastructure-control simulator where agents operate a fragile GPU cluster under heat, energy limits, hidden hardware wear, random chaos events, and cascading failures. The agent must complete critical jobs, but greedy scheduling can overheat nodes, increase hidden degradation, trigger failures, and collapse the cluster. The demo lets you run baseline agents, compare outcomes, and inspect replay frames.

I built it because AI products usually treat compute as invisible and infinite. I wanted a small world where compute has consequences, and where good judgment means knowing when not to maximize throughput.

With another 10 hours, I would add an actual LLM policy loop, PPO/GRPO training, richer replay charts, and scenario authoring.

I cut Kubernetes realism, real cloud APIs, accounts, persistence, and full RL training. The smallest interesting version is a working arena with hidden state, chaos, baselines, guardrails, scoring, and replay.

Word count: 133
