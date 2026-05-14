from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Dict, List, Optional, Tuple


class NodeStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    MISSED = "missed"


class ActionType(str, Enum):
    PLACE_JOB = "PLACE_JOB"
    COOL_ZONE = "COOL_ZONE"
    THROTTLE_NODE = "THROTTLE_NODE"
    INSPECT_NODE = "INSPECT_NODE"
    REPAIR_NODE = "REPAIR_NODE"
    MIGRATE_JOB = "MIGRATE_JOB"
    SACRIFICE_NODE = "SACRIFICE_NODE"
    WAIT = "WAIT"


@dataclass
class Node:
    id: str
    zone: str
    gpus: int
    free: int
    temp: float
    load: float = 0.0
    hidden_wear: float = 0.0
    status: NodeStatus = NodeStatus.HEALTHY
    maintenance_left: int = 0
    inspected_wear: Optional[float] = None


@dataclass
class Job:
    id: str
    gpus: int
    deadline: int
    reward: int
    progress: float = 0.0
    status: JobStatus = JobStatus.QUEUED
    node_id: Optional[str] = None
    critical: bool = False


@dataclass
class Action:
    kind: ActionType
    node_id: Optional[str] = None
    job_id: Optional[str] = None
    zone: Optional[str] = None
    target_node_id: Optional[str] = None


@dataclass
class StepResult:
    observation: Dict
    reward: float
    done: bool
    events: List[str]
    breakdown: Dict[str, float]


@dataclass
class ThermalChaosArena:
    seed: int = 7
    max_steps: int = 24
    level: int = 3
    rng: random.Random = field(init=False)
    step_count: int = field(default=0, init=False)
    energy: float = field(default=100.0, init=False)
    zones: Dict[str, float] = field(default_factory=dict, init=False)
    nodes: Dict[str, Node] = field(default_factory=dict, init=False)
    jobs: Dict[str, Job] = field(default_factory=dict, init=False)
    events: List[str] = field(default_factory=list, init=False)
    cascade_count: int = field(default=0, init=False)
    guardrail_hits: int = field(default=0, init=False)
    last_inspection: str = field(default="none", init=False)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.reset()

    def reset(self) -> Dict:
        self.rng = random.Random(self.seed)
        self.step_count = 0
        self.energy = 100.0
        self.zones = {"A": 0.45, "B": 0.45}
        self.events = []
        self.cascade_count = 0
        self.guardrail_hits = 0
        self.last_inspection = "none"
        self.nodes = {}
        for i in range(10):
            zone = "A" if i < 5 else "B"
            wear = self.rng.uniform(0.05, 0.28 + 0.07 * self.level)
            self.nodes[f"n{i}"] = Node(
                id=f"n{i}",
                zone=zone,
                gpus=8,
                free=8,
                temp=self.rng.uniform(45, 62),
                hidden_wear=min(0.92, wear),
            )
        self.jobs = {}
        for i in range(14):
            critical = i in {0, 3, 8, 11}
            self.jobs[f"j{i}"] = Job(
                id=f"j{i}",
                gpus=self.rng.choice([1, 2, 4]),
                deadline=self.rng.randint(5, 18) - (2 if critical else 0),
                reward=self.rng.randint(8, 18) * (2 if critical else 1),
                critical=critical,
            )
        return self.observe()

    def observe(self) -> Dict:
        visible_nodes = []
        for n in self.nodes.values():
            visible_nodes.append({
                "id": n.id,
                "zone": n.zone,
                "free": n.free,
                "gpus": n.gpus,
                "temp": round(n.temp, 1),
                "load": round(n.load, 2),
                "status": n.status.value,
                "inspected_wear": None if n.inspected_wear is None else round(n.inspected_wear, 2),
            })
        visible_jobs = []
        for j in self.jobs.values():
            if j.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                visible_jobs.append({
                    "id": j.id,
                    "gpus": j.gpus,
                    "deadline": j.deadline,
                    "reward": j.reward,
                    "progress": round(j.progress, 2),
                    "status": j.status.value,
                    "node_id": j.node_id,
                    "critical": j.critical,
                })
        health = sum(1 for n in self.nodes.values() if n.status != NodeStatus.FAILED) / len(self.nodes)
        return {
            "step": self.step_count,
            "max_steps": self.max_steps,
            "energy": round(self.energy, 1),
            "cluster_health": round(health, 2),
            "avg_temp": round(sum(n.temp for n in self.nodes.values()) / len(self.nodes), 1),
            "zones": {z: round(v, 2) for z, v in self.zones.items()},
            "nodes": visible_nodes,
            "jobs": visible_jobs,
            "cascade_count": self.cascade_count,
            "guardrail_hits": self.guardrail_hits,
            "last_inspection": self.last_inspection,
            "recent_events": self.events[-8:],
        }

    def legal_actions(self) -> List[Action]:
        actions = [Action(ActionType.WAIT)]
        for z in self.zones:
            actions.append(Action(ActionType.COOL_ZONE, zone=z))
        for n in self.nodes.values():
            if n.status in {NodeStatus.HEALTHY, NodeStatus.WARNING}:
                actions += [
                    Action(ActionType.THROTTLE_NODE, node_id=n.id),
                    Action(ActionType.INSPECT_NODE, node_id=n.id),
                    Action(ActionType.REPAIR_NODE, node_id=n.id),
                    Action(ActionType.SACRIFICE_NODE, node_id=n.id),
                ]
                for j in self.jobs.values():
                    if j.status == JobStatus.QUEUED and n.free >= j.gpus:
                        actions.append(Action(ActionType.PLACE_JOB, node_id=n.id, job_id=j.id))
                for j in self.jobs.values():
                    if j.status == JobStatus.RUNNING and j.node_id != n.id and n.free >= j.gpus:
                        actions.append(Action(ActionType.MIGRATE_JOB, target_node_id=n.id, job_id=j.id))
        return actions

    def step(self, action: Action) -> StepResult:
        if self.step_count >= self.max_steps:
            return StepResult(self.observe(), 0, True, ["episode already done"], {})
        before_done = sum(1 for j in self.jobs.values() if j.status == JobStatus.DONE)
        before_missed = sum(1 for j in self.jobs.values() if j.status == JobStatus.MISSED)
        step_events = []
        reward = -0.25
        invalid = False

        def event(msg: str):
            step_events.append(msg)
            self.events.append(f"t{self.step_count}: {msg}")

        if action.kind == ActionType.PLACE_JOB:
            j = self.jobs.get(action.job_id or "")
            n = self.nodes.get(action.node_id or "")
            if not j or not n or j.status != JobStatus.QUEUED or n.free < j.gpus or n.status not in {NodeStatus.HEALTHY, NodeStatus.WARNING}:
                invalid = True
            else:
                j.status = JobStatus.RUNNING
                j.node_id = n.id
                n.free -= j.gpus
                n.load += j.gpus / n.gpus
                event(f"placed {j.id} on {n.id}")
                if n.temp > 82:
                    self.guardrail_hits += 1
                    event(f"guardrail: risky placement on hot node {n.id}")
                    reward -= 3
        elif action.kind == ActionType.COOL_ZONE:
            if action.zone not in self.zones or self.energy < 5:
                invalid = True
            else:
                self.zones[action.zone] = min(1.0, self.zones[action.zone] + 0.22)
                self.energy -= 5
                event(f"boosted cooling in zone {action.zone}")
        elif action.kind == ActionType.THROTTLE_NODE:
            n = self.nodes.get(action.node_id or "")
            if not n or n.status not in {NodeStatus.HEALTHY, NodeStatus.WARNING}:
                invalid = True
            else:
                n.load *= 0.72
                n.temp -= 5.5
                self.energy -= 1
                event(f"throttled {n.id}")
        elif action.kind == ActionType.INSPECT_NODE:
            n = self.nodes.get(action.node_id or "")
            if not n:
                invalid = True
            else:
                n.inspected_wear = max(0.0, min(1.0, n.hidden_wear + self.rng.gauss(0, 0.07)))
                self.energy -= 1
                self.last_inspection = f"{n.id}: wear≈{n.inspected_wear:.2f}"
                event(f"inspected {n.id}")
        elif action.kind == ActionType.REPAIR_NODE:
            n = self.nodes.get(action.node_id or "")
            if not n or n.status == NodeStatus.FAILED or self.energy < 8:
                invalid = True
            else:
                n.hidden_wear = max(0.0, n.hidden_wear - 0.32)
                n.temp -= 9
                n.status = NodeStatus.MAINTENANCE
                n.maintenance_left = 2
                self.energy -= 8
                event(f"repaired {n.id}; offline for 2 steps")
        elif action.kind == ActionType.MIGRATE_JOB:
            j = self.jobs.get(action.job_id or "")
            target = self.nodes.get(action.target_node_id or "")
            old = self.nodes.get(j.node_id or "") if j else None
            if not j or not target or not old or j.status != JobStatus.RUNNING or target.free < j.gpus:
                invalid = True
            else:
                old.free += j.gpus
                old.load = max(0, old.load - j.gpus / old.gpus)
                target.free -= j.gpus
                target.load += j.gpus / target.gpus
                j.node_id = target.id
                j.progress = max(0, j.progress - 0.08)
                self.energy -= 3
                event(f"migrated {j.id} to {target.id}")
        elif action.kind == ActionType.SACRIFICE_NODE:
            n = self.nodes.get(action.node_id or "")
            if not n or n.status == NodeStatus.FAILED:
                invalid = True
            else:
                for j in self.jobs.values():
                    if j.node_id == n.id and j.status == JobStatus.RUNNING:
                        j.status = JobStatus.QUEUED
                        j.node_id = None
                        j.progress = max(0, j.progress - 0.12)
                n.status = NodeStatus.MAINTENANCE
                n.maintenance_left = 3
                n.free = n.gpus
                n.load = 0
                n.temp -= 12
                event(f"sacrificed {n.id} to contain thermal spread")
        elif action.kind == ActionType.WAIT:
            alert_nodes = [n.id for n in self.nodes.values() if n.temp > 86 or n.status == NodeStatus.WARNING]
            if alert_nodes:
                self.guardrail_hits += 1
                reward -= 2
                event(f"guardrail: waited while alerts active: {', '.join(alert_nodes[:3])}")

        if invalid:
            reward -= 5
            self.guardrail_hits += 1
            event(f"invalid action {action.kind.value}")

        reward += self._physics_tick(event)
        self._chaos_tick(event)
        self._deadline_tick(event)

        done_now = sum(1 for j in self.jobs.values() if j.status == JobStatus.DONE)
        missed_now = sum(1 for j in self.jobs.values() if j.status == JobStatus.MISSED)
        reward += (done_now - before_done) * 8
        reward -= (missed_now - before_missed) * 7
        reward -= self.cascade_count * 0.2
        reward -= max(0, 72 - self.energy) * 0.01

        health = sum(1 for n in self.nodes.values() if n.status != NodeStatus.FAILED) / len(self.nodes)
        reward += (health - 0.7) * 2
        self.step_count += 1
        done = self.step_count >= self.max_steps or all(j.status in {JobStatus.DONE, JobStatus.MISSED} for j in self.jobs.values())
        breakdown = {
            "reward": round(reward, 2),
            "done_jobs": done_now,
            "missed_jobs": missed_now,
            "health": round(health, 2),
            "energy": round(self.energy, 1),
            "guardrails": self.guardrail_hits,
            "cascades": self.cascade_count,
        }
        return StepResult(self.observe(), round(reward, 2), done, step_events, breakdown)

    def _physics_tick(self, event) -> float:
        reward = 0.0
        for n in self.nodes.values():
            if n.status == NodeStatus.MAINTENANCE:
                n.maintenance_left -= 1
                n.temp = max(35, n.temp - 4)
                if n.maintenance_left <= 0:
                    n.status = NodeStatus.HEALTHY
                    event(f"{n.id} returned from maintenance")
                continue
            if n.status == NodeStatus.FAILED:
                n.temp = max(45, n.temp - 2)
                continue
            cooling = self.zones[n.zone]
            n.temp += 4.8 * n.load + 1.4 * n.hidden_wear - 5.5 * cooling
            n.temp = max(32, n.temp)
            n.hidden_wear = min(1.0, n.hidden_wear + max(0, n.temp - 74) / 900 + n.load / 420)
            if n.temp > 90 or (n.temp > 84 and n.hidden_wear > 0.62):
                fail_prob = 0.08 + (n.temp - 84) / 80 + n.hidden_wear / 7
                if self.rng.random() < fail_prob:
                    self._fail_node(n, event)
                    reward -= 12
                else:
                    n.status = NodeStatus.WARNING
                    reward -= 1.5
                    event(f"{n.id} entered warning state")
            elif n.temp < 78 and n.status == NodeStatus.WARNING:
                n.status = NodeStatus.HEALTHY
        for z in self.zones:
            self.zones[z] = max(0.25, self.zones[z] - 0.06)
        return reward

    def _fail_node(self, node: Node, event):
        if node.status == NodeStatus.FAILED:
            return
        node.status = NodeStatus.FAILED
        node.free = 0
        node.load = 0
        event(f"FAILURE: {node.id} failed")
        for j in self.jobs.values():
            if j.node_id == node.id and j.status == JobStatus.RUNNING:
                j.status = JobStatus.QUEUED
                j.node_id = None
                j.progress = max(0, j.progress - 0.2)
        neighbors = [n for n in self.nodes.values() if n.zone == node.zone and n.status != NodeStatus.FAILED and n.id != node.id]
        for n in neighbors:
            n.temp += 4.5
            n.hidden_wear = min(1.0, n.hidden_wear + 0.05)
            if n.temp > 91 and self.rng.random() < 0.35:
                self.cascade_count += 1
                event(f"CASCADE: {node.id} pushed {n.id} over the edge")
                self._fail_node(n, event)

    def _chaos_tick(self, event):
        if self.step_count in {4, 9, 15} or self.rng.random() < 0.04 * self.level:
            z = self.rng.choice(list(self.zones.keys()))
            self.zones[z] = max(0.15, self.zones[z] - self.rng.uniform(0.12, 0.24))
            event(f"chaos: cooling efficiency dropped in zone {z}")
        if self.rng.random() < 0.03 * self.level:
            n = self.rng.choice(list(self.nodes.values()))
            if n.status != NodeStatus.FAILED:
                n.hidden_wear = min(1.0, n.hidden_wear + 0.16)
                event(f"chaos: hidden wear spike on {n.id}")

    def _deadline_tick(self, event):
        for j in self.jobs.values():
            if j.status == JobStatus.RUNNING:
                node = self.nodes.get(j.node_id or "")
                if node and node.status in {NodeStatus.HEALTHY, NodeStatus.WARNING}:
                    j.progress += 0.22 * max(0.2, 1.0 - node.load * 0.25)
                    if j.progress >= 1.0:
                        j.status = JobStatus.DONE
                        node.free += j.gpus
                        node.load = max(0, node.load - j.gpus / node.gpus)
                        event(f"completed {j.id}")
            if j.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                j.deadline -= 1
                if j.deadline < 0:
                    if j.node_id and j.node_id in self.nodes:
                        n = self.nodes[j.node_id]
                        n.free = min(n.gpus, n.free + j.gpus)
                        n.load = max(0, n.load - j.gpus / n.gpus)
                    j.status = JobStatus.MISSED
                    event(f"missed deadline for {j.id}")


class BaseAgent:
    name = "Base"
    def act(self, obs: Dict, env: ThermalChaosArena) -> Action:
        return Action(ActionType.WAIT)


class GreedyThroughputAgent(BaseAgent):
    name = "Greedy Throughput"
    def act(self, obs: Dict, env: ThermalChaosArena) -> Action:
        queued = sorted([j for j in env.jobs.values() if j.status == JobStatus.QUEUED], key=lambda j: (-j.reward, j.deadline))
        nodes = sorted([n for n in env.nodes.values() if n.status in {NodeStatus.HEALTHY, NodeStatus.WARNING}], key=lambda n: -n.free)
        for j in queued:
            for n in nodes:
                if n.free >= j.gpus:
                    return Action(ActionType.PLACE_JOB, node_id=n.id, job_id=j.id)
        return Action(ActionType.WAIT)


class ThermalSREAgent(BaseAgent):
    name = "Thermal SRE"
    def act(self, obs: Dict, env: ThermalChaosArena) -> Action:
        danger = sorted([n for n in env.nodes.values() if n.status != NodeStatus.FAILED], key=lambda n: (n.temp + 25 * n.hidden_wear), reverse=True)
        if danger and danger[0].temp > 88:
            return Action(ActionType.SACRIFICE_NODE, node_id=danger[0].id)
        if danger and danger[0].temp > 80:
            return Action(ActionType.THROTTLE_NODE, node_id=danger[0].id)
        unknown_hot = [n for n in danger if n.inspected_wear is None and n.temp > 72]
        if unknown_hot:
            return Action(ActionType.INSPECT_NODE, node_id=unknown_hot[0].id)
        for z in env.zones:
            zone_nodes = [n for n in env.nodes.values() if n.zone == z]
            if sum(n.temp for n in zone_nodes) / len(zone_nodes) > 73 and env.energy > 12:
                return Action(ActionType.COOL_ZONE, zone=z)
        queued = sorted([j for j in env.jobs.values() if j.status == JobStatus.QUEUED], key=lambda j: (j.deadline, -j.reward))
        safe_nodes = sorted([n for n in env.nodes.values() if n.status == NodeStatus.HEALTHY and n.temp < 78], key=lambda n: (n.temp, -n.free))
        for j in queued:
            for n in safe_nodes:
                if n.free >= j.gpus:
                    return Action(ActionType.PLACE_JOB, node_id=n.id, job_id=j.id)
        return Action(ActionType.WAIT)


class BalancedOperatorAgent(BaseAgent):
    name = "Balanced Operator"
    def act(self, obs: Dict, env: ThermalChaosArena) -> Action:
        hot = [n for n in env.nodes.values() if n.temp > 84 and n.status in {NodeStatus.HEALTHY, NodeStatus.WARNING}]
        if hot:
            return Action(ActionType.THROTTLE_NODE, node_id=max(hot, key=lambda n: n.temp).id)
        queued = sorted([j for j in env.jobs.values() if j.status == JobStatus.QUEUED], key=lambda j: (j.deadline, -j.reward))
        nodes = sorted([n for n in env.nodes.values() if n.status == NodeStatus.HEALTHY and n.temp < 84], key=lambda n: (n.temp, -n.free))
        for j in queued:
            for n in nodes:
                if n.free >= j.gpus:
                    return Action(ActionType.PLACE_JOB, node_id=n.id, job_id=j.id)
        if env.energy > 10:
            return Action(ActionType.COOL_ZONE, zone=min(env.zones, key=env.zones.get))
        return Action(ActionType.WAIT)


AGENTS = {
    GreedyThroughputAgent.name: GreedyThroughputAgent(),
    BalancedOperatorAgent.name: BalancedOperatorAgent(),
    ThermalSREAgent.name: ThermalSREAgent(),
}


def run_episode(agent_name: str, seed: int = 7, level: int = 3, max_steps: int = 24) -> Tuple[List[Dict], Dict]:
    env = ThermalChaosArena(seed=seed, level=level, max_steps=max_steps)
    agent = AGENTS[agent_name]
    frames = []
    total = 0.0
    done = False
    while not done:
        obs = env.observe()
        action = agent.act(obs, env)
        res = env.step(action)
        total += res.reward
        frames.append({"obs": res.observation, "action": action.kind.value, "reward": res.reward, "events": res.events, "breakdown": res.breakdown})
        done = res.done
    done_jobs = sum(1 for j in env.jobs.values() if j.status == JobStatus.DONE)
    missed = sum(1 for j in env.jobs.values() if j.status == JobStatus.MISSED)
    summary = {
        "agent": agent_name,
        "total_reward": round(total, 2),
        "completed_jobs": done_jobs,
        "missed_jobs": missed,
        "cascade_count": env.cascade_count,
        "guardrail_hits": env.guardrail_hits,
        "final_health": env.observe()["cluster_health"],
        "energy_left": env.observe()["energy"],
    }
    return frames, summary
