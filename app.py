from __future__ import annotations

import json
from typing import Dict, List, Tuple

import gradio as gr

from thermal_arena import AGENTS, ThermalChaosArena, run_episode


def node_html(obs: Dict) -> str:
    cards = []
    for n in obs["nodes"]:
        temp = n["temp"]
        if n["status"] == "failed":
            color = "#ff4d6d"
        elif temp > 84 or n["status"] == "warning":
            color = "#ffb703"
        else:
            color = "#4ade80"
        cards.append(
            f"""
            <div class='node-card' style='border-color:{color}'>
              <div><b>{n['id']}</b> <span>{n['zone']}</span></div>
              <div class='temp'>{temp}°C</div>
              <div>GPU {n['gpus'] - n['free']}/{n['gpus']} · load {n['load']}</div>
              <div>{n['status']} · wear {n['inspected_wear'] if n['inspected_wear'] is not None else 'hidden'}</div>
            </div>
            """
        )
    return "<div class='node-grid'>" + "".join(cards) + "</div>"


def jobs_html(obs: Dict) -> str:
    rows = []
    for j in sorted(obs["jobs"], key=lambda x: (x["status"] != "running", x["deadline"])):
        critical = "🔥" if j["critical"] else ""
        rows.append(
            f"<tr><td>{critical}{j['id']}</td><td>{j['status']}</td><td>{j['gpus']}</td><td>{j['deadline']}</td><td>{j['progress']}</td><td>{j['node_id'] or '-'}</td><td>{j['reward']}</td></tr>"
        )
    return """
    <table class='jobs'><thead><tr><th>Job</th><th>Status</th><th>GPUs</th><th>Deadline</th><th>Progress</th><th>Node</th><th>Value</th></tr></thead><tbody>
    """ + "".join(rows) + "</tbody></table>"


def metrics_html(obs: Dict, summary: Dict | None = None) -> str:
    items = [
        ("Step", f"{obs['step']}/{obs['max_steps']}"),
        ("Energy", obs["energy"]),
        ("Health", obs["cluster_health"]),
        ("Avg temp", f"{obs['avg_temp']}°C"),
        ("Cascades", obs["cascade_count"]),
        ("Guardrails", obs["guardrail_hits"]),
    ]
    if summary:
        items = [("Reward", summary["total_reward"]), ("Completed", summary["completed_jobs"]), ("Missed", summary["missed_jobs"])] + items
    return "<div class='metric-grid'>" + "".join(f"<div><b>{v}</b><span>{k}</span></div>" for k, v in items) + "</div>"


def event_html(events: List[str]) -> str:
    return "<div class='events'>" + "".join(f"<p>{e}</p>" for e in events[-12:]) + "</div>"


def render_frame(frame: Dict) -> Tuple[str, str, str, str]:
    obs = frame["obs"]
    return metrics_html(obs, frame.get("summary")), node_html(obs), jobs_html(obs), event_html(obs.get("recent_events", []))


def run(agent: str, seed: int, level: int, max_steps: int):
    frames, summary = run_episode(agent, int(seed), int(level), int(max_steps))
    if frames:
        frames[-1]["summary"] = summary
    replay = json.dumps({"frames": frames, "summary": summary}, indent=2)
    metrics, nodes, jobs, events = render_frame(frames[-1])
    return metrics, nodes, jobs, events, replay, summary


def compare(seed: int, level: int, max_steps: int):
    summaries = []
    for agent in AGENTS:
        _, summary = run_episode(agent, int(seed), int(level), int(max_steps))
        summaries.append(summary)
    summaries.sort(key=lambda x: x["total_reward"], reverse=True)
    rows = "".join(
        f"<tr><td>{i+1}</td><td>{s['agent']}</td><td>{s['total_reward']}</td><td>{s['completed_jobs']}</td><td>{s['missed_jobs']}</td><td>{s['cascade_count']}</td><td>{s['guardrail_hits']}</td><td>{s['final_health']}</td></tr>"
        for i, s in enumerate(summaries)
    )
    table = "<table class='jobs'><thead><tr><th>#</th><th>Agent</th><th>Reward</th><th>Done</th><th>Missed</th><th>Cascades</th><th>Guardrails</th><th>Health</th></tr></thead><tbody>" + rows + "</tbody></table>"
    return table, json.dumps(summaries, indent=2)


def replay_frame(replay_json: str, index: int):
    data = json.loads(replay_json)
    frames = data["frames"]
    index = max(0, min(int(index), len(frames) - 1))
    return (*render_frame(frames[index]), f"Showing frame {index + 1}/{len(frames)}")


CSS = """
.gradio-container { background: #071018 !important; color: #eaf6ff !important; }
.node-grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(155px,1fr)); gap:10px; }
.node-card { background:#0f1d2b; border:2px solid #4ade80; border-radius:14px; padding:12px; color:#dcecff; }
.node-card span { color:#86efac; float:right; }
.temp { font-size:26px; font-weight:900; margin:8px 0; }
.metric-grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(110px,1fr)); gap:10px; }
.metric-grid div { background:#102033; border:1px solid #1d3955; border-radius:14px; padding:14px; }
.metric-grid b { display:block; font-size:24px; color:#75ffe1; }
.metric-grid span { color:#9fb4ca; }
.jobs { width:100%; border-collapse:collapse; background:#0f1d2b; border-radius:14px; overflow:hidden; }
.jobs th,.jobs td { border-bottom:1px solid #1d3955; padding:9px; text-align:left; }
.jobs th { color:#75ffe1; }
.events { background:#0f1d2b; border:1px solid #1d3955; border-radius:14px; padding:12px; max-height:260px; overflow:auto; }
.events p { margin:6px 0; color:#dcecff; }
"""

with gr.Blocks(css=CSS, title="Thermal Chaos Arena") as demo:
    gr.Markdown(
        """
        # 🔥 Thermal Chaos Arena
        **An original AI infrastructure-control arena.** Agents must run critical AI jobs on a fragile GPU cluster while heat, hidden hardware wear, energy limits, and chaos events try to collapse it.

        The core test: can an AI learn *not* to be greedy when short-term throughput creates long-term failure?
        """
    )
    with gr.Tab("Live Episode"):
        with gr.Row():
            agent = gr.Dropdown(list(AGENTS.keys()), value="Thermal SRE", label="Agent")
            seed = gr.Number(value=7, precision=0, label="Seed")
            level = gr.Slider(1, 5, value=3, step=1, label="Chaos level")
            max_steps = gr.Slider(8, 40, value=24, step=1, label="Max steps")
        run_btn = gr.Button("Run full episode")
        metrics = gr.HTML()
        nodes = gr.HTML()
        jobs = gr.HTML()
        events = gr.HTML()
        replay = gr.Code(label="Replay JSON", language="json")
        summary = gr.JSON(label="Summary")
        run_btn.click(run, [agent, seed, level, max_steps], [metrics, nodes, jobs, events, replay, summary])
    with gr.Tab("Agent Comparison"):
        with gr.Row():
            c_seed = gr.Number(value=7, precision=0, label="Seed")
            c_level = gr.Slider(1, 5, value=3, step=1, label="Chaos level")
            c_steps = gr.Slider(8, 40, value=24, step=1, label="Max steps")
        c_btn = gr.Button("Compare baseline agents")
        table = gr.HTML()
        raw = gr.Code(language="json", label="Raw comparison")
        c_btn.click(compare, [c_seed, c_level, c_steps], [table, raw])
    with gr.Tab("Replay Inspector"):
        gr.Markdown("Paste replay JSON from a live episode, then scrub through the collapse chain.")
        replay_in = gr.Code(language="json", label="Replay JSON")
        idx = gr.Slider(0, 39, value=0, step=1, label="Frame index")
        replay_btn = gr.Button("Render frame")
        r_metrics = gr.HTML()
        r_nodes = gr.HTML()
        r_jobs = gr.HTML()
        r_events = gr.HTML()
        r_status = gr.Textbox(label="Status")
        replay_btn.click(replay_frame, [replay_in, idx], [r_metrics, r_nodes, r_jobs, r_events, r_status])
    gr.Markdown(
        """
        ## Why this is not just a toy scheduler
        The agent cannot see hidden hardware wear directly. Greedy placement can look optimal for a few steps, then heat increases wear, chaos weakens cooling, nodes fail, neighbours absorb thermal pressure, and a cascade begins. Guardrails penalize reward-hacking patterns like waiting through alerts or placing jobs on dangerously hot nodes.
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
