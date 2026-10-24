"""Per-intersection performance on cologne3 (shared vs. independent vs. masked policy).

For every cologne3 test file, runs fixed-time control and each available A2C
variant, and records the mean number of halted vehicles (queue) on the
incoming lanes of each of the three signals over the hour.

  python per_intersection.py --workers 6

Output: results/cologne3_per_intersection.csv (one row per controller x seed x file x signal)
and a printed table: mean queue per signal, mean +- s.d. over seeds.
"""
import common as C

import argparse, csv, glob, os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import torch as th

SCENARIO = "cologne3"
VARIANTS = ["a2c_64x64_variants", "a2c-ind_64x64_variants", "a2c-mask_64x64_variants"]


def job(spec):
    th.set_num_threads(1)
    queues = {}

    def on_step(env):
        for t in env.ts_ids:
            queues.setdefault(t, []).append(env.traffic_signals[t].get_total_queued())

    policy = None
    if spec["model"]:
        algo = C.parse_tag(spec["controller"])[0]
        policy = C.argmax_policy(C.export_policy(C.ALGOS[algo].load(spec["model"], device="cpu")))
    m = C.run_episode(SCENARIO, spec["path"], "rl" if policy else "fixed", sumo_seed=spec["sumo_seed"],
                      policy=policy, on_step=on_step)
    return [dict(controller=spec["controller"], seed=spec["seed"], route_file=os.path.basename(spec["path"]),
                 signal=t, mean_queue=round(sum(q) / len(q), 3), avg_total_wait=round(m["avg_total_wait"], 3))
            for t, q in queues.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    files = C.split_files(SCENARIO, "test")
    specs = []
    for i, f in enumerate(files):
        base = dict(path=f, sumo_seed=C.TEST_SUMO_SEED + i)       # same SUMO seeds as evaluate.py
        specs.append(dict(base, controller="fixed", seed=0, model=None))
        for v in VARIANTS:
            for model in sorted(glob.glob(os.path.join(C.RESULTS, SCENARIO, v, "seed*", "best_model.zip"))):
                seed = int(os.path.basename(os.path.dirname(model))[4:])
                specs.append(dict(base, controller=v, seed=seed, model=model))
    print(f"{len(specs)} episodes, {a.workers} workers")
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for k, fut in enumerate(as_completed([pool.submit(job, s) for s in specs]), 1):
            rows += fut.result()
            print(f"  [{k}/{len(specs)}]", flush=True)
    out = os.path.join(C.RESULTS, f"{SCENARIO}_per_intersection.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    df = pd.DataFrame(rows)
    per_seed = df.groupby(["controller", "seed", "signal"]).mean_queue.mean().reset_index()
    summ = per_seed.groupby(["controller", "signal"]).mean_queue.agg(["mean", "std"]).round(2)
    print("\nMean queue (halted vehicles on incoming lanes), mean +- s.d. over seeds:")
    print(summ.unstack("signal").to_string())
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
