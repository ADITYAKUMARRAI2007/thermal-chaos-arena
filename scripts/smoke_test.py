from thermal_arena import AGENTS, run_episode, ThermalChaosArena


def main():
    assert len(AGENTS) == 3
    env = ThermalChaosArena(seed=1, level=3, max_steps=6)
    obs = env.reset()
    assert len(obs["nodes"]) == 10
    assert len(obs["jobs"]) > 0
    for name in AGENTS:
        frames, summary = run_episode(name, seed=3, level=3, max_steps=8)
        assert frames, name
        assert "total_reward" in summary
        assert summary["completed_jobs"] + summary["missed_jobs"] <= 14
        print(name, summary)
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
