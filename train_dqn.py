"""Train a DQN agent on one RESCO scenario and compare it to the fixed-time baseline.

Only works out-of-the-box for SINGLE-junction scenarios (cologne1, ingolstadt1),
since it uses SumoEnvironment(single_agent=True), which gives a plain Gym env
that stable-baselines3 can train on directly. cologne3 / ingolstadt7 have
multiple junctions (multi-agent) and need a different setup -- ask once this
one is working.

Run:  python train_dqn.py cologne1
Needs: pip install eclipse-sumo sumo-rl torch stable-baselines3
"""
import os
try:
    import sumo as _sumo
    os.environ.setdefault("SUMO_HOME", _sumo.SUMO_HOME)   # pip eclipse-sumo sets this for you
except ImportError:
    pass

import os, sys, glob, xml.etree.ElementTree as ET
import sumo_rl
from sumo_rl import SumoEnvironment
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

RESCO = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "RESCO")
SPLITS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")
WINDOW = {"cologne": (25200, 28800), "ingolstadt": (57600, 61200)}


def make_env(scenario, route_file, seed=0):
    d = os.path.join(RESCO, scenario)
    begin, end = WINDOW["cologne" if scenario.startswith("cologne") else "ingolstadt"]
    return SumoEnvironment(
        net_file=os.path.join(d, f"{scenario}.net.xml"),
        route_file=route_file,
        begin_time=begin, num_seconds=end - begin, delta_time=5,
        yellow_time=2, min_green=5, sumo_seed=seed,
        reward_fn="diff-waiting-time",
        single_agent=True,           # <- one junction only; gives a plain Gym env
        use_gui=False, sumo_warnings=False, additional_sumo_cmd="--no-step-log",
    )


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "cologne1"
    train_files = sorted(glob.glob(os.path.join(SPLITS, scenario, "train", "*.rou.xml")))
    if not train_files:
        sys.exit(f"No train files found for {scenario} -- run make_splits.py first.")

    print(f"Training DQN on {scenario}: {len(train_files)} demand variants available.")

    # Simplest possible loop: train a fixed number of episodes, cycling
    # through the training demand files so the agent sees varied traffic
    # (0.8x-1.2x scale) rather than memorising one fixed hour.
    TOTAL_EPISODES = 50          # start small to confirm it runs end-to-end;
                                  # your make_splits.py comment says 200 was
                                  # the budget used for the reference numbers
    model = None
    for ep in range(TOTAL_EPISODES):
        rou = train_files[ep % len(train_files)]
        env = Monitor(make_env(scenario, rou, seed=ep))
        if model is None:
            model = DQN("MlpPolicy", env, verbose=0, learning_rate=1e-3,
                        buffer_size=50_000, learning_starts=500,
                        train_freq=1, target_update_interval=500)
        else:
            model.set_env(env)
        # one episode's worth of steps (num_seconds / delta_time)
        steps_per_ep = (WINDOW["cologne" if scenario.startswith("cologne") else "ingolstadt"][1]
                        - WINDOW["cologne" if scenario.startswith("cologne") else "ingolstadt"][0]) // 5
        model.learn(total_timesteps=steps_per_ep, reset_num_timesteps=False)
        env.close()
        if (ep + 1) % 10 == 0:
            print(f"  finished episode {ep+1}/{TOTAL_EPISODES}  (file: {os.path.basename(rou)})")

    model.save(f"dqn_{scenario}")
    print(f"\nSaved trained model to dqn_{scenario}.zip")

    # ---- quick evaluation on the REAL hour (test set) ----------------------
    real_rou = glob.glob(os.path.join(SPLITS, scenario, "test", f"{scenario}_real.rou.xml"))[0]
    env = make_env(scenario, real_rou, seed=42)
    obs, _ = env.reset()
    done = False
    total_wait = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        total_wait.append(info.get("system_mean_waiting_time", float("nan")))
    env.close()
    print(f"\nDQN on real hour -- mean waiting time: {sum(total_wait)/len(total_wait):.1f} s")
    print("Compare this number to baselines.py's 'fixed' and 'actuated' rows for the same scenario.")
