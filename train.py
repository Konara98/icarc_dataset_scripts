"""Train one RL traffic-light controller (DQN, Double DQN, PPO or A2C).

What one run does (Step 6 of the plan):
  1. Each episode is one simulated hour, using the next training demand file.
  2. Every 5 s each traffic light picks a green phase and is rewarded for
     reducing the waiting time on its approaches.
  3. Every --val-every episodes the current policy is run WITHOUT exploration
     on the 5 validation files. If its average total waiting time is the best
     so far, it is saved as best_model.zip. Test files are never used here.
  4. After --episodes episodes, final_model.zip is saved as well.

All traffic lights in a scenario share one policy network (see common.Adapter),
so the same script handles cologne1, ingolstadt1 and the 3-light cologne3.

  python train.py --scenario cologne1 --algo dqn --seed 0
  python train.py --scenario cologne3 --algo ppo --seed 3
  python train.py --scenario cologne1 --algo dqn --seed 0 --net 16 16          # smaller network
  python train.py --scenario cologne1 --algo dqn --seed 0 --train-set real     # ablation: real hour only

Use run_all.py to run many algorithms x seeds in parallel.

Output: results/<scenario>/<algo>_<net>_<train-set>/seed<k>/
  best_model.zip   chosen on validation  -> used for the test results
  final_model.zip  after the last episode
  train_log.csv    one row per training episode (learning curve)
  val_log.csv      one row per validation check
  config.json      every setting, so the run can be reported and repeated
"""
import common as C          # first: sets SUMO_HOME / libsumo before sumo-rl loads

import argparse, csv, json, os, platform, statistics, time
import torch as th
import stable_baselines3

# Starting hyper-parameters. Tune them on the VALIDATION files only, and give
# every algorithm the same tuning budget, or the comparison is not fair.
DQN_HP = dict(learning_rate=1e-3, buffer_size=50_000, learning_starts=1_000, batch_size=64,
              gamma=0.99, train_freq=1, gradient_steps=1, target_update_interval=500,
              exploration_fraction=0.1, exploration_final_eps=0.02)
HYPER = {
    "dqn": DQN_HP,
    "ddqn": DQN_HP,
    "ppo": dict(learning_rate=3e-4, n_steps=720, batch_size=144, n_epochs=10, gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.01),
    "a2c": dict(learning_rate=7e-4, n_steps=5, gamma=0.99, gae_lambda=1.0, ent_coef=0.01),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=C.SCENARIOS)
    ap.add_argument("--algo", required=True, choices=list(C.ALGOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--net", type=int, nargs="+", default=[64, 64], help="hidden layer sizes")
    ap.add_argument("--train-set", choices=["variants", "real"], default="variants",
                    help="variants = the 20 training files; real = only the real hour (ablation)")
    ap.add_argument("--lr", type=float, help="override the learning rate")
    a = ap.parse_args()
    th.set_num_threads(1)       # several runs share the CPU; one thread each is fastest overall

    s = a.scenario
    train_files = C.split_files(s, "train") if a.train_set == "variants" else [C.real_hour_file(s)]
    val_files = C.split_files(s, "val")
    if not train_files or not val_files:
        raise SystemExit(f"No split files for {s} -- run make_splits.py first.")
    out = C.run_dir(s, a.algo, a.net, a.train_set, a.seed)
    os.makedirs(out, exist_ok=True)
    name = f"[{s} {C.run_tag(a.algo, a.net, a.train_set)} seed{a.seed}]"

    venv = C.TrafficVecEnv(s, train_files, a.seed)
    hp = dict(HYPER[a.algo], **({"learning_rate": a.lr} if a.lr else {}))
    model = C.ALGOS[a.algo]("MlpPolicy", venv, seed=a.seed, device="cpu", verbose=0,
                            policy_kwargs=dict(net_arch=list(a.net), activation_fn=th.nn.ReLU), **hp)

    config = dict(vars(a), hyperparameters=hp, traffic_lights=venv.ad.ids, inputs=venv.ad.obs_dim,
                  actions=venv.ad.n_actions, yellow_s=C.plan_yellow(s), decision_s=C.DELTA,
                  min_green_s=C.MIN_GREEN, train_files=[os.path.basename(f) for f in train_files],
                  val_files=[os.path.basename(f) for f in val_files],
                  versions=dict(python=platform.python_version(), torch=th.__version__,
                                stable_baselines3=stable_baselines3.__version__),
                  started=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    tlog = open(os.path.join(out, "train_log.csv"), "w", newline="")
    vlog = open(os.path.join(out, "val_log.csv"), "w", newline="")
    tw, vw = csv.writer(tlog), csv.writer(vlog)
    tw.writerow(["episode", "route_file", "reward", "avg_total_wait", "avg_wait", "avg_trip",
                 "avg_stops", "finished", "vehicles", "elapsed_s"])
    vw.writerow(["episode", "val_avg_total_wait", "val_avg_trip", "val_finished_pct", "best", "elapsed_s"])
    t0, best = time.time(), {"score": float("inf"), "episode": None}

    def validate(ep):
        policy = lambda x: model.predict(x, deterministic=True)[0]
        ms = [C.run_episode(s, f, "rl", sumo_seed=C.VAL_SUMO_SEED + i, policy=policy)
              for i, f in enumerate(val_files)]
        score = statistics.mean(m["avg_total_wait"] for m in ms)
        is_best = score < best["score"]
        if is_best:
            best.update(score=score, episode=ep)
            model.save(os.path.join(out, "best_model"))
        vw.writerow([ep, round(score, 3), round(statistics.mean(m["avg_trip"] for m in ms), 3),
                     round(100 * sum(m["finished"] for m in ms) / sum(m["vehicles"] for m in ms), 2),
                     int(is_best), round(time.time() - t0)])
        vlog.flush()
        print(f"{name} validation after ep {ep}: total wait {score:.1f} s"
              f"{'  <- best so far, saved' if is_best else ''}", flush=True)

    def on_episode_end(st):
        ep = st["episode"]
        tw.writerow([ep, st["route_file"], round(st["reward"], 3), round(st["avg_total_wait"], 3),
                     round(st["avg_wait"], 3), round(st["avg_trip"], 3), round(st["avg_stops"], 3),
                     st["finished"], st["vehicles"], round(time.time() - t0)])
        tlog.flush()
        print(f"{name} ep {ep}/{a.episodes} {st['route_file']}: total wait {st['avg_total_wait']:.1f} s,"
              f" reward {st['reward']:.1f}, {time.time() - t0:.0f} s elapsed", flush=True)
        if ep % a.val_every == 0 or ep == a.episodes:
            validate(ep)

    venv.on_episode_end = on_episode_end
    try:
        model.learn(total_timesteps=a.episodes * C.steps_per_episode(s) * venv.num_envs)
    finally:
        venv.close()
        tlog.close()
        vlog.close()
    model.save(os.path.join(out, "final_model"))

    config.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), train_seconds=round(time.time() - t0),
                  best_val_avg_total_wait=best["score"], best_episode=best["episode"])
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"{name} done in {(time.time() - t0) / 60:.0f} min; best validation {best['score']:.1f} s"
          f" at episode {best['episode']}")


if __name__ == "__main__":
    main()
