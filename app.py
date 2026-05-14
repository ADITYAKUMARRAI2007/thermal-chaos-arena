from __future__ import annotations

import html
import json
from collections import Counter
from typing import Dict, List, Tuple

import gradio as gr

from thermal_arena import AGENTS, run_episode


def esc(value) -> str:
    return html.escape(str(value))


def temp_color(temp: float, status: str) -> str:
    if status == "failed":
        return "#ff3b67"
    if status == "maintenance":
        return "#8b9bb4"
    if temp >= 88:
        return "#ff5c33"
    if temp >= 80:
        return "#ffb703"
    return "#38f8b4"


def progress_bar(value: float, color: str = "#38f8b4", max_value: float = 1.0) -> str:
    pct = max(0, min(100, (value / max_value) * 100))
    return f"<div class='bar'><i style='width:{pct:.1f}%;background:{color}'></i></div>"


def hero_html() -> str:
    return """
    <section class="hero">
      <div class="hero-copy">
        <div class="kicker">GPU cluster control · hidden-state chaos simulator</div>
        <h1>Thermal Chaos Arena</h1>
        <p>Run AI workloads on a fragile GPU cluster. Greedy scheduling looks smart for a few steps — then heat, hidden wear, cooling failures, and cascades punish it.</p>
        <div class="hero-pills">
          <span>10 nodes</span><span>2 thermal zones</span><span>hidden degradation</span><span>chaos events</span><span>guardrails</span><span>replay inspector</span>
        </div>
      </div>
      <div class="hero-card">
        <b>Core question</b>
        <p>Can an AI agent learn when <em>not</em> to maximize throughput?</p>
      </div>
    </section>
    """


def node_html(obs: Dict) -> str:
    zone_groups = {"A": [], "B": []}
    for node in obs["nodes"]:
        zone_groups.setdefault(node["zone"], []).append(node)
    lanes = []
    for zone, nodes in zone_groups.items():
        cooling = obs.get("zones", {}).get(zone, 0)
        cards = []
        for n in nodes:
            temp = float(n["temp"])
            used = n["gpus"] - n["free"]
            color = temp_color(temp, n["status"])
            wear = n["inspected_wear"] if n["inspected_wear"] is not None else "hidden"
            cards.append(
                f"""
                <div class="node-card" style="--accent:{color}">
                  <div class="node-top"><b>{esc(n['id'])}</b><span>{esc(n['status'])}</span></div>
                  <div class="temp">{temp:.1f}°C</div>
                  <div class="mini-row"><span>GPU {used}/{n['gpus']}</span><span>load {n['load']}</span></div>
                  {progress_bar(used, color, n['gpus'])}
                  <div class="wear">wear: {esc(wear)}</div>
                </div>
                """
            )
        lanes.append(
            f"""
            <div class="zone-lane">
              <div class="zone-head"><b>Zone {esc(zone)}</b><span>cooling {cooling}</span></div>
              <div class="node-grid">{''.join(cards)}</div>
            </div>
            """
        )
    return "<div class='topology'>" + "".join(lanes) + "</div>"


def jobs_html(obs: Dict) -> str:
    rows = []
    for j in sorted(obs["jobs"], key=lambda x: (x["status"] != "running", x["deadline"], -x["reward"])):
        critical = "<span class='critical'>critical</span>" if j["critical"] else ""
        deadline_class = "danger" if j["deadline"] <= 2 else "warn" if j["deadline"] <= 5 else ""
        rows.append(
            f"""
            <tr>
              <td><b>{esc(j['id'])}</b> {critical}</td><td>{esc(j['status'])}</td><td>{j['gpus']}</td>
              <td class="{deadline_class}">{j['deadline']}</td><td>{progress_bar(float(j['progress']))}</td>
              <td>{esc(j['node_id'] or '-')}</td><td>{j['reward']}</td>
            </tr>
            """
        )
    return """
    <table class='jobs'><thead><tr><th>Job</th><th>Status</th><th>GPUs</th><th>Deadline</th><th>Progress</th><th>Node</th><th>Value</th></tr></thead><tbody>
    """ + "".join(rows) + "</tbody></table>"


def metrics_html(obs: Dict, summary: Dict | None = None) -> str:
    items = [
        ("Energy", obs["energy"], "battery budget left"),
        ("Health", f"{obs['cluster_health'] * 100:.0f}%", "non-failed nodes"),
        ("Avg temp", f"{obs['avg_temp']}°C", "thermal pressure"),
        ("Cascades", obs["cascade_count"], "linked failures"),
        ("Guardrails", obs["guardrail_hits"], "unsafe choices"),
        ("Step", f"{obs['step']}/{obs['max_steps']}", "episode progress"),
    ]
    if summary:
        items = [
            ("Reward", summary["total_reward"], "final score"),
            ("Completed", summary["completed_jobs"], "finished jobs"),
            ("Missed", summary["missed_jobs"], "deadline failures"),
        ] + items
    return "<div class='metric-grid'>" + "".join(
        f"<div class='metric'><b>{esc(v)}</b><span>{esc(k)}</span><small>{esc(sub)}</small></div>" for k, v, sub in items
    ) + "</div>"


def event_html(events: List[str]) -> str:
    if not events:
        return "<div class='events empty'>No events yet. Run an episode to generate a failure/survival trace.</div>"
    return "<div class='events'>" + "".join(f"<p>{esc(e)}</p>" for e in events[-16:]) + "</div>"


def sparkline(values: List[float], color: str, label: str, suffix: str = "") -> str:
    if not values:
        return ""
    width, height, pad = 460, 96, 10
    lo, hi = min(values), max(values)
    if hi == lo:
        hi += 1
    points = []
    for i, value in enumerate(values):
        x = pad + (width - 2 * pad) * (i / max(1, len(values) - 1))
        y = height - pad - (height - 2 * pad) * ((value - lo) / (hi - lo))
        points.append(f"{x:.1f},{y:.1f}")
    return f"""
    <div class="chart-card">
      <div class="chart-head"><b>{esc(label)}</b><span>{values[-1]:.1f}{suffix}</span></div>
      <svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">
        <polyline fill="none" stroke="{color}" stroke-width="4" points="{' '.join(points)}" />
      </svg>
    </div>
    """


def analytics_html(frames: List[Dict], summary: Dict) -> str:
    temps = [float(f["obs"]["avg_temp"]) for f in frames]
    energy = [float(f["obs"]["energy"]) for f in frames]
    health = [float(f["obs"]["cluster_health"]) * 100 for f in frames]
    rewards = [float(f["reward"]) for f in frames]
    actions = Counter(f["action"] for f in frames)
    action_rows = "".join(
        f"<div><span>{esc(action)}</span>{progress_bar(count, '#7c5cff', max(actions.values()))}<b>{count}</b></div>"
        for action, count in actions.most_common()
    )
    if summary["cascade_count"] > 0:
        verdict = "The cluster suffered cascading failure. This run proves why raw throughput is not enough."
        cls = "danger-card"
    elif summary["guardrail_hits"] > 0 or summary["missed_jobs"] > 2:
        verdict = "The cluster survived, but with unsafe decisions. Better policies should inspect and cool earlier."
        cls = "warn-card"
    else:
        verdict = "Clean survival run: the agent preserved infrastructure while completing work."
        cls = "safe-card"
    return f"""
    <div class="analytics">
      <div class="verdict {cls}"><b>Run diagnosis</b><p>{esc(verdict)}</p></div>
      <div class="charts">
        {sparkline(temps, '#ffb703', 'Average temperature', '°C')}
        {sparkline(energy, '#38f8b4', 'Energy remaining')}
        {sparkline(health, '#7c5cff', 'Cluster health', '%')}
        {sparkline(rewards, '#ff5c8a', 'Step reward')}
      </div>
      <div class="action-mix"><h3>Policy fingerprint</h3>{action_rows}</div>
    </div>
    """


def decision_trace_html(frames: List[Dict]) -> str:
    rows = []
    for idx, frame in enumerate(frames[-12:]):
        event = frame["events"][0] if frame["events"] else "quiet step"
        rows.append(
            f"<tr><td>{idx + max(0, len(frames)-12) + 1}</td><td><code>{esc(frame['action'])}</code></td><td>{frame['reward']}</td><td>{esc(event)}</td></tr>"
        )
    return "<table class='jobs trace'><thead><tr><th>Step</th><th>Action</th><th>Reward</th><th>First event</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def render_frame(frame: Dict) -> Tuple[str, str, str, str]:
    obs = frame["obs"]
    return metrics_html(obs, frame.get("summary")), node_html(obs), jobs_html(obs), event_html(obs.get("recent_events", []))


def run(agent: str, seed: int, level: int, max_steps: int):
    frames, summary = run_episode(agent, int(seed), int(level), int(max_steps))
    if frames:
        frames[-1]["summary"] = summary
    replay = json.dumps({"frames": frames, "summary": summary}, indent=2)
    metrics, nodes, jobs, events = render_frame(frames[-1])
    analytics = analytics_html(frames, summary)
    trace = decision_trace_html(frames)
    return metrics, analytics, nodes, jobs, events, trace, replay, summary


def compare(seed: int, level: int, max_steps: int):
    summaries = []
    for agent in AGENTS:
        _, summary = run_episode(agent, int(seed), int(level), int(max_steps))
        summaries.append(summary)
    summaries.sort(key=lambda x: x["total_reward"], reverse=True)
    best = summaries[0]
    rows = "".join(
        f"<tr><td>{i+1}</td><td><b>{esc(s['agent'])}</b></td><td>{s['total_reward']}</td><td>{s['completed_jobs']}</td><td>{s['missed_jobs']}</td><td>{s['cascade_count']}</td><td>{s['guardrail_hits']}</td><td>{s['final_health']}</td><td>{s['energy_left']}</td></tr>"
        for i, s in enumerate(summaries)
    )
    table = f"""
    <div class="verdict safe-card"><b>Best policy on this seed: {esc(best['agent'])}</b><p>It scored {best['total_reward']} with {best['completed_jobs']} completed jobs and {best['cascade_count']} cascades.</p></div>
    <table class='jobs'><thead><tr><th>#</th><th>Agent</th><th>Reward</th><th>Done</th><th>Missed</th><th>Cascades</th><th>Guardrails</th><th>Health</th><th>Energy</th></tr></thead><tbody>{rows}</tbody></table>
    """
    return table, json.dumps(summaries, indent=2)


def replay_frame(replay_json: str, index: int):
    data = json.loads(replay_json)
    frames = data["frames"]
    index = max(0, min(int(index), len(frames) - 1))
    metrics, nodes, jobs, events = render_frame(frames[index])
    return metrics, nodes, jobs, events, f"Showing frame {index + 1}/{len(frames)} · action={frames[index]['action']} · reward={frames[index]['reward']}"


CSS = """
:root { --bg:#061019; --panel:#0d1b2a; --panel2:#10243a; --line:#1f3b58; --text:#eaf6ff; --muted:#9fb4ca; --cyan:#38f8b4; --violet:#7c5cff; --red:#ff3b67; --amber:#ffb703; }
.gradio-container { background: radial-gradient(circle at 12% 0%, rgba(56,248,180,.16), transparent 28rem), radial-gradient(circle at 100% 8%, rgba(124,92,255,.2), transparent 30rem), var(--bg) !important; color: var(--text) !important; max-width: 1180px !important; }
footer { display:none !important; }
.hero { display:grid; grid-template-columns: 1.6fr .8fr; gap:18px; padding:34px; border:1px solid var(--line); background:linear-gradient(135deg, rgba(16,36,58,.92), rgba(6,16,25,.88)); border-radius:28px; box-shadow:0 20px 70px rgba(0,0,0,.35); margin-bottom:18px; }
.kicker { color:var(--cyan); text-transform:uppercase; letter-spacing:.14em; font-size:12px; font-weight:900; }
.hero h1 { font-size:64px; line-height:.9; margin:10px 0 14px; letter-spacing:-.06em; }
.hero p { color:#cfe0f6; font-size:18px; line-height:1.55; }
.hero-pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
.hero-pills span, .critical { border:1px solid rgba(56,248,180,.35); color:#bfffea; background:rgba(56,248,180,.08); border-radius:999px; padding:6px 10px; font-size:12px; font-weight:800; }
.hero-card { border:1px solid rgba(255,255,255,.12); border-radius:22px; padding:22px; background:rgba(0,0,0,.22); align-self:center; }
.hero-card b { color:var(--amber); }
.metric-grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(128px,1fr)); gap:12px; margin:12px 0; }
.metric { background:linear-gradient(180deg, rgba(16,36,58,.98), rgba(8,22,35,.98)); border:1px solid var(--line); border-radius:18px; padding:16px; }
.metric b { display:block; font-size:26px; color:var(--cyan); }
.metric span { display:block; font-weight:900; margin-top:4px; }
.metric small { color:var(--muted); }
.topology { display:grid; gap:14px; }
.zone-lane { border:1px solid var(--line); border-radius:22px; padding:14px; background:rgba(13,27,42,.78); }
.zone-head { display:flex; justify-content:space-between; color:var(--cyan); margin-bottom:10px; }
.node-grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
.node-card { background:rgba(6,16,25,.9); border:1px solid var(--accent); border-left:5px solid var(--accent); border-radius:16px; padding:12px; box-shadow: inset 0 0 24px rgba(255,255,255,.03); }
.node-top { display:flex; justify-content:space-between; color:#eaf6ff; }
.node-top span, .wear, .mini-row { color:var(--muted); font-size:12px; }
.temp { font-size:28px; font-weight:950; color:var(--accent); margin:8px 0; }
.mini-row { display:flex; justify-content:space-between; }
.bar { height:8px; background:#07131f; border-radius:999px; overflow:hidden; margin:7px 0; border:1px solid rgba(255,255,255,.08); }
.bar i { display:block; height:100%; border-radius:999px; }
.jobs { width:100%; border-collapse:collapse; background:rgba(13,27,42,.92); border:1px solid var(--line); border-radius:18px; overflow:hidden; margin-top:12px; }
.jobs th,.jobs td { border-bottom:1px solid var(--line); padding:10px; text-align:left; vertical-align:middle; }
.jobs th { color:var(--cyan); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
.jobs code { color:#d8d1ff; background:rgba(124,92,255,.16); padding:4px 7px; border-radius:7px; }
.danger { color:var(--red); font-weight:900; } .warn { color:var(--amber); font-weight:900; }
.events { background:rgba(13,27,42,.92); border:1px solid var(--line); border-radius:18px; padding:14px; max-height:310px; overflow:auto; }
.events p { margin:7px 0; color:#dcecff; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:13px; }
.empty { color:var(--muted); }
.analytics { display:grid; gap:14px; }
.charts { display:grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap:12px; }
.chart-card, .verdict, .action-mix { background:rgba(13,27,42,.92); border:1px solid var(--line); border-radius:18px; padding:14px; }
.chart-head { display:flex; justify-content:space-between; margin-bottom:8px; color:#dcecff; }
.chart-head span { color:var(--cyan); font-weight:900; }
.chart-card svg { width:100%; height:96px; background:#07131f; border-radius:12px; }
.verdict b { font-size:18px; } .verdict p { margin:6px 0 0; color:#dcecff; }
.safe-card { border-color:rgba(56,248,180,.45); } .warn-card { border-color:rgba(255,183,3,.55); } .danger-card { border-color:rgba(255,59,103,.55); }
.action-mix h3 { margin:0 0 8px; }
.action-mix div div { display:grid; grid-template-columns: 150px 1fr 32px; align-items:center; gap:10px; margin:8px 0; color:var(--muted); }
#component-0 .tabs { border-radius:20px; }
@media (max-width: 760px) { .hero { grid-template-columns:1fr; padding:22px; } .hero h1 { font-size:42px; } .charts { grid-template-columns:1fr; } }
"""

with gr.Blocks(css=CSS, title="Thermal Chaos Arena") as demo:
    gr.HTML(hero_html())
    gr.Markdown(
        """
        Try **Thermal SRE** vs **Greedy Throughput** on the same seed. The point is not pretty scheduling — it is whether a policy understands that compute is fragile.
        Suggested stress seeds: `7`, `19`, `42`, `88`.
        """
    )
    with gr.Tab("Mission Control"):
        with gr.Row():
            agent = gr.Dropdown(list(AGENTS.keys()), value="Thermal SRE", label="Agent policy")
            seed = gr.Number(value=7, precision=0, label="Seed")
            level = gr.Slider(1, 5, value=4, step=1, label="Chaos level")
            max_steps = gr.Slider(8, 40, value=28, step=1, label="Max steps")
        run_btn = gr.Button("▶ Run arena episode", variant="primary")
        metrics = gr.HTML()
        analytics = gr.HTML()
        with gr.Row():
            nodes = gr.HTML(label="Cluster topology")
        with gr.Row():
            jobs = gr.HTML(label="Workload queue")
        with gr.Row():
            events = gr.HTML(label="Event log")
            trace = gr.HTML(label="Decision trace")
        with gr.Accordion("Replay JSON / API payload", open=False):
            replay = gr.Code(label="Replay JSON", language="json")
            summary = gr.JSON(label="Summary")
        run_btn.click(run, [agent, seed, level, max_steps], [metrics, analytics, nodes, jobs, events, trace, replay, summary])
    with gr.Tab("Policy Benchmarks"):
        gr.Markdown("Run every built-in policy against the exact same chaos seed. This makes the demo feel like a real benchmark, not a canned animation.")
        with gr.Row():
            c_seed = gr.Number(value=7, precision=0, label="Seed")
            c_level = gr.Slider(1, 5, value=4, step=1, label="Chaos level")
            c_steps = gr.Slider(8, 40, value=28, step=1, label="Max steps")
        c_btn = gr.Button("Compare policies", variant="primary")
        table = gr.HTML()
        raw = gr.Code(language="json", label="Raw comparison")
        c_btn.click(compare, [c_seed, c_level, c_steps], [table, raw])
    with gr.Tab("Replay Inspector"):
        gr.Markdown("Paste replay JSON from Mission Control and scrub through the run. This is useful for explaining exactly why a cluster survived or collapsed.")
        replay_in = gr.Code(language="json", label="Replay JSON")
        idx = gr.Slider(0, 39, value=0, step=1, label="Frame index")
        replay_btn = gr.Button("Render frame", variant="primary")
        r_status = gr.Textbox(label="Frame status")
        r_metrics = gr.HTML()
        r_nodes = gr.HTML()
        r_jobs = gr.HTML()
        r_events = gr.HTML()
        replay_btn.click(replay_frame, [replay_in, idx], [r_metrics, r_nodes, r_jobs, r_events, r_status])
    gr.Markdown(
        """
        ### What makes this more than a toy scheduler
        The agent sees visible temperatures and queue pressure, but not true hardware wear. It can spend actions inspecting, cooling, throttling, repairing, migrating, or sacrificing nodes. Heat creates delayed damage. Chaos can weaken a zone at the worst time. Failed nodes push stress onto neighbours. Guardrails penalize policies that look good by abusing unsafe behavior.
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
