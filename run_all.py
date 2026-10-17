"""Run many training jobs in parallel: scenarios x algorithms x seeds.

  python run_all.py --scenarios cologne1 ingolstadt1 cologne3 --workers 6    # main study: 4 algos x 5 seeds
  python run_all.py --scenarios cologne1 --algos dqn --net 16 16             # smaller network
  python run_all.py --scenarios cologne1 --algos dqn --train-set real        # ablation: real hour only

A run that already finished all its episodes is skipped, so re-running after a
crash, a reboot or a Colab disconnect only does the missing runs (an unfinished
run starts again from episode 1). Each run's console output is saved as
train.out in its results folder.
"""
import common as C

import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))


def finished(out, episodes):
    """A run counts as done only if it completed the requested number of episodes
    (so a short pilot run with the same name is redone, not mistaken for a full run)."""
    try:
        with open(os.path.join(out, "config.json")) as fh:
            c = json.load(fh)
        return "finished" in c and c["episodes"] == episodes and os.path.exists(os.path.join(out, "final_model.zip"))
    except (OSError, ValueError, KeyError):
        return False


def run(job, extra):
    s, algo, seed, net, train_set = job
    out = C.run_dir(s, algo, net, train_set, seed)
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "train.py"), "--scenario", s, "--algo", algo,
           "--seed", str(seed), "--train-set", train_set, "--net", *map(str, net), *extra]
    t = time.time()
    with open(os.path.join(out, "train.out"), "w") as fh:
        code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    return code, (time.time() - t) / 60


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["cologne1", "ingolstadt1"], choices=C.SCENARIOS)
    ap.add_argument("--algos", nargs="+", default=list(C.ALGOS), choices=list(C.ALGOS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--net", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--train-set", choices=["variants", "real"], default="variants")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()

    jobs = [(s, algo, seed, a.net, a.train_set) for s in a.scenarios for algo in a.algos for seed in a.seeds]
    todo = [j for j in jobs if not finished(C.run_dir(j[0], j[1], j[3], j[4], j[2]), a.episodes)]
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already finished, {len(todo)} to do, {a.workers} at a time")
    extra = ["--episodes", str(a.episodes), "--val-every", str(a.val_every)]
    failed = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(run, j, extra): j for j in todo}
        for i, f in enumerate(as_completed(futures), 1):
            j = futures[f]
            code, minutes = f.result()
            label = f"{j[0]} {C.run_tag(j[1], j[3], j[4])} seed{j[2]}"
            print(f"[{i}/{len(todo)}] {label}: {'ok' if code == 0 else f'FAILED (exit {code})'} in {minutes:.0f} min",
                  flush=True)
            if code:
                failed.append(label)
    if failed:
        print("\nFailed runs (see train.out in each results folder):\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
