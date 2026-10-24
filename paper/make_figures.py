"""Build the paper's figures and print the numbers used in its tables.

Run from the repository root after training and evaluate.py:
    python paper/make_figures.py
Writes paper/figures/learning_curves.pdf and prints the table values.
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "figures")
SCEN = ["cologne1", "ingolstadt1", "cologne3"]
ALGOS = ["dqn", "ddqn", "ppo", "a2c"]
NAMES = {"dqn": "DQN", "ddqn": "Double DQN", "ppo": "PPO", "a2c": "A2C"}
COLORS = {"dqn": "#2a78d6", "ddqn": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100"}  # fixed categorical order
FAIL_RATIO = 2.0   # an episode "fails" if its total wait is more than 2x fixed-time on the same file


def learning_curves():
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.8))
    for ax, s in zip(axes, SCEN):
        fixed = json.load(open(os.path.join(RES, s, "actuated_params.json")))["fixed_val_score"]
        for a in ALGOS:
            logs = [pd.read_csv(f) for f in glob.glob(os.path.join(RES, s, f"{a}_64x64_variants", "seed*", "val_log.csv"))]
            # median seed and interquartile band: robust to the occasional gridlocked checkpoint
            g = pd.concat(logs).groupby("episode")["val_avg_total_wait"]
            q1, med, q3 = g.quantile(0.25), g.median(), g.quantile(0.75)
            ax.fill_between(med.index, q1, q3, color=COLORS[a], alpha=0.15, lw=0)
            ax.plot(med.index, med, color=COLORS[a], lw=1.4, label=NAMES[a])
        ax.axhline(fixed, color="#52514e", lw=1, ls="--", label="Fixed-time")
        # log scale: shows the whole range, including gridlocked checkpoints, without clipping
        ax.set_yscale("log")
        ax.set_ylim(3, 1500)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(10, 200)
        ax.set_title(s, fontsize=8, loc="left")
        ax.set_xlabel("Training episode")
        ax.grid(axis="y", color="#e6e5e0", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Waiting time (s)")   # template: words, not symbols, in axis labels
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.9], w_pad=1.0)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "learning_curves.pdf"), bbox_inches="tight")
    print("written", os.path.join(OUT, "learning_curves.pdf"))


def tables():
    e = pd.read_csv(os.path.join(RES, "test_episodes.csv"))
    fixed_ep = e[e.controller == "fixed"].set_index(["scenario", "route_file"])["avg_total_wait"]
    e["fail"] = [w > FAIL_RATIO * fixed_ep[(s, f)] for s, f, w in zip(e.scenario, e.route_file, e.avg_total_wait)]
    main = e.controller.str.endswith("_64x64_variants")
    e["algo"] = np.where(main, e.controller.str.split("_").str[0], e.controller)   # main-study runs only
    print("\nMain table: mean +- sd over seeds, median over seeds, failed episodes / 50")
    for s in SCEN:
        d = e[e.scenario == s]
        for c in ["fixed", "actuated", "random"] + ALGOS:
            g = d[d.algo == c]
            ps = g.groupby("seed").avg_total_wait.mean()
            sd = f"+-{ps.std():.1f}" if len(ps) > 1 else ""
            print(f"  {s:12} {c:9} {ps.mean():7.1f}{sd:8} median {ps.median():6.1f}   fails {int(g.fail.sum())}/{len(g)}")
    print("\nReduction vs fixed-time (%) of the mean, by demand level")
    summ = pd.read_csv(os.path.join(RES, "test_summary.csv"))
    for s in SCEN:
        t = summ[summ.scenario == s].pivot_table(index="controller", columns="demand", values="vs_fixed_pct")
        print(s); print(t[["x0.7", "x1.0", "x1.3", "real"]].round(1).to_string())
    print("\nWelch t-test on per-seed means")
    rl = e[main]
    ps = rl.groupby(["scenario", "algo", "seed"]).avg_total_wait.mean()
    for s in SCEN:
        f = e[(e.scenario == s) & (e.controller == "fixed")].avg_total_wait.mean()
        for a in ALGOS:
            p_fixed = stats.ttest_1samp(ps[s][a], f).pvalue
            p_a2c = stats.ttest_ind(ps[s]["a2c"], ps[s][a], equal_var=False).pvalue if a != "a2c" else np.nan
            print(f"  {s:12} {a:5} vs fixed p={p_fixed:.4f}   a2c vs {a} p={p_a2c:.4f}")


def instability():
    print("\nValidation checks after episode 50 that are worse than 2x fixed-time (% of checks)")
    for s in SCEN:
        fixed = json.load(open(os.path.join(RES, s, "actuated_params.json")))["fixed_val_score"]
        row = []
        for a in ALGOS:
            v = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(RES, s, f"{a}_64x64_variants", "seed*", "val_log.csv")))
            v = v[v.episode > 50]
            row.append(f"{a} {100 * (v.val_avg_total_wait > 2 * fixed).mean():4.0f}%")
        print(f"  {s:12}", "  ".join(row))


def ablation():
    """A2C full framework vs. -C1 (real hour only) and -C2 (independent policy per light)."""
    e = pd.read_csv(os.path.join(RES, "test_episodes.csv"))
    fixed_ep = e[e.controller == "fixed"].set_index(["scenario", "route_file"])["avg_total_wait"]
    e["fail"] = [w > FAIL_RATIO * fixed_ep[(s, f)] for s, f, w in zip(e.scenario, e.route_file, e.avg_total_wait)]
    variants = {"full (C1+C2)": "a2c_64x64_variants", "-C1 real hour only": "a2c_64x64_real",
                "-C2 independent": "a2c-ind_64x64_variants", "+ masking": "a2c-mask_64x64_variants"}
    print("\nAblation (A2C): unseen = 0.7x and 1.3x files; mean +- sd over seeds; failures / 50")
    for s in ["cologne1", "cologne3"]:
        full = e[(e.scenario == s) & (e.controller == variants["full (C1+C2)"])].groupby("seed").avg_total_wait.mean()
        for name, c in variants.items():
            d = e[(e.scenario == s) & (e.controller == c)]
            if d.empty:
                continue
            un = d[d.demand.isin(["x0.7", "x1.3"])].groupby("seed").avg_total_wait.mean()
            re = d[d.demand == "real"].groupby("seed").avg_total_wait.mean()
            al = d.groupby("seed").avg_total_wait.mean()
            p = stats.mannwhitneyu(full, al).pvalue if c != variants["full (C1+C2)"] else np.nan
            print(f"  {s:9} {name:20} unseen {un.mean():6.1f} +- {un.std():5.1f}   real {re.mean():6.1f} +- {re.std():5.1f}"
                  f"   all {al.mean():6.1f} +- {al.std():5.1f}   fails {int(d.fail.sum())}/{len(d)}   MWU vs full p={p:.3f}")


if __name__ == "__main__":
    learning_curves()
    tables()
    instability()
    ablation()
