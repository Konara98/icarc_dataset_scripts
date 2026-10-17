"""Measure the RL interface and episode cost for each RESCO scenario.

For each scenario: observation size, number of actions, decisions per episode,
wall-clock per episode (random policy), and the fixed-time baseline's waiting time.
"""
import os
try:
    import sumo as _sumo
    os.environ.setdefault("SUMO_HOME", _sumo.SUMO_HOME)   # pip eclipse-sumo sets this for you
except ImportError:
    pass

import os, sys, time, statistics
import numpy as np
import sumo_rl
from sumo_rl import SumoEnvironment

RESCO = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "RESCO")
WINDOW = {"cologne": (25200, 28800), "ingolstadt": (57600, 61200)}

def env_for(s, fixed=False, seed=0):
    d = os.path.join(RESCO, s)
    begin, end = WINDOW["cologne" if s.startswith("cologne") else "ingolstadt"]
    return SumoEnvironment(
        net_file=os.path.join(d, f"{s}.net.xml"),
        route_file=[os.path.join(d, f) for f in os.listdir(d) if f.endswith(".rou.xml")][0],
        begin_time=begin, num_seconds=end - begin, delta_time=5,
        yellow_time=2, min_green=5, fixed_ts=fixed, sumo_seed=seed,
        reward_fn="diff-waiting-time", use_gui=False, single_agent=False,
        sumo_warnings=False, additional_sumo_cmd="--no-step-log",
    )

def run(s, fixed, seed=0):
    env = env_for(s, fixed, seed)
    obs = env.reset()
    agents = list(obs.keys())
    shape = {a: env.observation_spaces(a).shape[0] for a in agents}
    nact = {a: env.action_spaces(a).n for a in agents}
    rng = np.random.default_rng(seed)
    t0 = time.time(); steps = 0; wait = []
    done = {"__all__": False}
    while not done["__all__"]:
        act = {a: int(rng.integers(nact[a])) for a in agents}
        obs, rew, done, info = env.step(act)
        steps += 1
        wait.append(info.get("system_mean_waiting_time", float("nan")))
    dt = time.time() - t0
    env.close()
    return agents, shape, nact, steps, dt, statistics.mean(wait)

if __name__ == "__main__":
    for s in sys.argv[1:] or ["cologne1", "ingolstadt1", "cologne3", "ingolstadt7"]:
        a, shp, na, steps, dt, w = run(s, fixed=False)
        _, _, _, steps_f, dt_f, w_f = run(s, fixed=True)
        print(f"\n== {s}")
        print(f"  agents (junctions)   : {len(a)}")
        print(f"  observation size     : {sorted(set(shp.values()))}  per junction")
        print(f"  actions (green phases): {sorted(set(na.values()))}  per junction")
        print(f"  decisions / episode  : {steps} per junction  -> {steps*len(a)} total")
        print(f"  wall-clock / episode : random {dt:.1f} s   fixed-time {dt_f:.1f} s")
        print(f"  mean waiting time    : random {w:.1f} s   fixed-time {w_f:.1f} s")
