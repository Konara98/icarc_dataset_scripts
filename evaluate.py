"""Test every controller on every test file with the same metrics (Step 7).

Controllers
  fixed     the city's signal plan stored in the .net.xml
  actuated  the same plan; detectors stretch or cut each green
            (tune it on the validation files first with --tune-actuated)
  random    a random green phase every 5 s, 5 seeds (sanity check)
  RL        every trained run found in results/<scenario>/<tag>/seed<k>/best_model.zip.
            Run as FP32 and, with --int8, as an 8-bit quantised copy: the
            version a small roadside computer would actually run.

On a given test file every controller gets the same SUMO seed, so differences
come from the controller and not from random driver behaviour.

  python evaluate.py --tune-actuated --scenarios cologne1 ingolstadt1 cologne3   # once, before testing
  python evaluate.py --scenarios cologne1 ingolstadt1 cologne3 --int8 --workers 6

Output in results/:
  <scenario>/actuated_params.json  chosen actuated settings (from --tune-actuated)
  test_episodes.csv                one row per controller x seed x test file (raw data)
  test_summary.csv                 per scenario x demand level x controller:
                                   mean and std over seeds (the table for the paper)
Rows already in test_episodes.csv are skipped, so the script can be stopped and re-run.
"""
import common as C          # first: sets SUMO_HOME / libsumo before sumo-rl loads

import argparse, csv, glob, json, os, re, statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import torch as th

KEY = ["scenario", "controller", "seed", "precision", "route_file"]
METRICS = ["vehicles", "finished", "avg_total_wait", "avg_wait", "avg_entry_wait", "avg_trip", "avg_stops"]
COLUMNS = KEY + ["algo", "net", "train_set", "demand"] + METRICS
RANDOM_SEEDS = range(5)
ACTUATED_GRID = [dict(max_gap=g, max_scale=m) for g in (2.0, 3.0, 4.0) for m in (1.5, 2.0, 3.0)]

_policies = {}


def demand_level(route_file):
    m = re.search(r"_x(\d\.\d)_", route_file)
    return f"x{m.group(1)}" if m else "real"


def load_policy(model_path, algo, precision):
    """Load a trained model as a plain network (FP32 or INT8); cached per worker."""
    key = (model_path, precision)
    if key not in _policies:
        net = C.export_policy(C.ALGOS[algo].load(model_path, device="cpu"))
        _policies[key] = C.argmax_policy(C.quantize_int8(net) if precision == "int8" else net)
    return _policies[key]


def run_job(job):
    th.set_num_threads(1)
    policy = None
    if job["controller"] not in ("fixed", "actuated", "random"):
        policy = load_policy(job["model"], job["algo"], job["precision"])
    m = C.run_episode(job["scenario"], job["path"], "rl" if policy else job["controller"],
                      sumo_seed=job["sumo_seed"], rng_seed=job["seed"], policy=policy,
                      actuated=job.get("actuated"))
    row = {k: job.get(k, "") for k in COLUMNS if k not in METRICS}
    row.update({k: (round(m[k], 3) if isinstance(m[k], float) else m[k]) for k in METRICS})
    return row


def run_parallel(jobs, workers, on_row):
    if workers <= 1:
        for j in jobs:
            on_row(j, run_job(j))
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, j): j for j in jobs}
        for f in as_completed(futures):
            on_row(futures[f], f.result())


def actuated_params(scenario):
    path = os.path.join(C.RESULTS, scenario, "actuated_params.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)["best"]
    return dict(max_gap=3.0, max_scale=2.0)


def tune_actuated(scenarios, workers):
    """Pick actuated settings on the VALIDATION files, so actuated is a fair, tuned baseline."""
    for s in scenarios:
        files = C.split_files(s, "val")
        jobs = [dict(scenario=s, controller="actuated", seed=0, precision="", path=f,
                     route_file=os.path.basename(f), demand=demand_level(f),
                     sumo_seed=C.VAL_SUMO_SEED + i, actuated=p, cfg=k)
                for k, p in enumerate(ACTUATED_GRID) for i, f in enumerate(files)]
        jobs += [dict(scenario=s, controller="fixed", seed=0, precision="", path=f,
                      route_file=os.path.basename(f), demand=demand_level(f),
                      sumo_seed=C.VAL_SUMO_SEED + i, cfg=-1) for i, f in enumerate(files)]
        scores = {}
        run_parallel(jobs, workers, lambda j, r: scores.setdefault(j["cfg"], []).append(r["avg_total_wait"]))
        mean = {k: statistics.mean(v) for k, v in scores.items()}
        best = min((k for k in mean if k >= 0), key=mean.get)
        print(f"\n== {s}: actuated tuning on {len(files)} validation files (avg total wait, s)")
        print(f"  fixed-time                     {mean[-1]:7.1f}")
        for k, p in enumerate(ACTUATED_GRID):
            print(f"  max_gap {p['max_gap']:.0f}  max green x{p['max_scale']:.1f}   {mean[k]:7.1f}"
                  f"{'   <- chosen' if k == best else ''}")
        os.makedirs(os.path.join(C.RESULTS, s), exist_ok=True)
        with open(os.path.join(C.RESULTS, s, "actuated_params.json"), "w") as fh:
            json.dump(dict(best=ACTUATED_GRID[best], fixed_val_score=mean[-1],
                           grid=[dict(p, val_score=mean[k]) for k, p in enumerate(ACTUATED_GRID)]),
                      fh, indent=2)


def build_jobs(scenarios, split, int8, which):
    jobs = []
    for s in scenarios:
        act = actuated_params(s)
        runs = sorted(glob.glob(os.path.join(C.RESULTS, s, "*", "seed*", f"{which}_model.zip")))
        for i, f in enumerate(C.split_files(s, split)):
            base = dict(scenario=s, path=f, route_file=os.path.basename(f), demand=demand_level(f),
                        sumo_seed=(C.TEST_SUMO_SEED if split == "test" else C.VAL_SUMO_SEED) + i)
            jobs.append(dict(base, controller="fixed", seed=0, precision=""))
            jobs.append(dict(base, controller="actuated", seed=0, precision="", actuated=act))
            jobs += [dict(base, controller="random", seed=k, precision="") for k in RANDOM_SEEDS]
            for model in runs:
                tag = os.path.basename(os.path.dirname(os.path.dirname(model)))
                algo, net, train_set = C.parse_tag(tag)
                seed = int(os.path.basename(os.path.dirname(model))[4:])
                for prec in ["fp32"] + (["int8"] if int8 else []):
                    jobs.append(dict(base, controller=tag, algo=algo, net="x".join(map(str, net)),
                                     train_set=train_set, seed=seed, precision=prec, model=model))
    return jobs


def summarise(episodes_csv, out_csv):
    df = pd.read_csv(episodes_csv)
    df["precision"] = df["precision"].fillna("")
    df["finished_pct"] = 100 * df["finished"] / df["vehicles"]
    cols = ["avg_total_wait", "avg_wait", "avg_entry_wait", "avg_trip", "avg_stops", "finished_pct"]
    df = pd.concat([df, df.assign(demand="all")])
    # average over the files of one demand level for each seed, then mean/std across seeds
    per_seed = df.groupby(["scenario", "demand", "controller", "precision", "seed"])[cols].mean()
    summ = per_seed.groupby(level=[0, 1, 2, 3]).agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ["seeds"] = per_seed.groupby(level=[0, 1, 2, 3]).size()
    summ = summ.reset_index()
    fixed = summ[summ.controller == "fixed"].set_index(["scenario", "demand"])["avg_total_wait_mean"]
    summ["vs_fixed_pct"] = [100 * (fixed[(r.scenario, r.demand)] - r.avg_total_wait_mean) / fixed[(r.scenario, r.demand)]
                            for r in summ.itertuples()]
    order = {"fixed": 0, "actuated": 1, "random": 2}
    summ["_o"] = summ.controller.map(lambda c: order.get(c, 3))
    summ = summ.sort_values(["scenario", "demand", "_o", "controller", "precision"]).drop(columns="_o")
    summ.round(3).to_csv(out_csv, index=False)

    for (s, d), g in summ.groupby(["scenario", "demand"], sort=False):
        print(f"\n== {s}  demand {d}   (mean +- std over seeds; vs fixed = % less total waiting)")
        print(f"  {'controller':28} {'prec':5} {'total wait s':>14} {'trip s':>8} {'stops':>6} {'finished %':>10} {'vs fixed':>9}")
        for r in g.itertuples():
            sd = "" if pd.isna(r.avg_total_wait_std) else f"+-{r.avg_total_wait_std:.1f}"
            print(f"  {r.controller:28} {r.precision:5} {r.avg_total_wait_mean:7.1f}{sd:>7} {r.avg_trip_mean:8.1f}"
                  f" {r.avg_stops_mean:6.2f} {r.finished_pct_mean:10.1f} {r.vs_fixed_pct:+8.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=C.SCENARIOS, choices=C.SCENARIOS)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--int8", action="store_true", help="also test INT8-quantised RL policies")
    ap.add_argument("--model", choices=["best", "final"], default="best")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--tune-actuated", action="store_true", help="tune actuated on validation, then stop")
    a = ap.parse_args()

    if a.tune_actuated:
        tune_actuated(a.scenarios, a.workers)
        return

    os.makedirs(C.RESULTS, exist_ok=True)
    episodes_csv = os.path.join(C.RESULTS, f"{a.split}_episodes.csv")
    done = set()
    if os.path.exists(episodes_csv):
        with open(episodes_csv) as fh:
            done = {tuple(r[k] for k in KEY) for r in csv.DictReader(fh)}
    jobs = [j for j in build_jobs(a.scenarios, a.split, a.int8, a.model)
            if tuple(str(j[k]) for k in KEY) not in done]
    print(f"{len(jobs)} episodes to run ({len(done)} already done), {a.workers} workers")

    new = not os.path.exists(episodes_csv)
    with open(episodes_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        count = [0]

        def on_row(job, row):
            w.writerow(row)
            fh.flush()
            count[0] += 1
            print(f"  [{count[0]}/{len(jobs)}] {row['scenario']} {row['route_file']:26} "
                  f"{row['controller']} {row['precision']} seed{row['seed']}: "
                  f"total wait {row['avg_total_wait']:.1f} s", flush=True)

        run_parallel(jobs, a.workers, on_row)

    summarise(episodes_csv, os.path.join(C.RESULTS, f"{a.split}_summary.csv"))
    print(f"\nwritten: {episodes_csv}\n         {os.path.join(C.RESULTS, f'{a.split}_summary.csv')}")


if __name__ == "__main__":
    main()
