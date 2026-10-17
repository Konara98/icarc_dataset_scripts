# Traffic-signal control with RL on RESCO: scripts and research guide

These scripts train four reinforcement-learning (RL) controllers (DQN, Double DQN, PPO, A2C)
to run traffic lights in three real-city SUMO scenarios (cologne1, ingolstadt1, cologne3).
They then test the controllers against fixed-time, actuated and random control on traffic
the controllers never saw, and measure whether the trained networks are small and fast
enough for a roadside computer.

## The research questions

| | Question | Where the answer comes from |
|---|---|---|
| RQ1 | Which RL algorithm controls the lights best? | `test_summary.csv`, demand `all` |
| RQ2 | Does it still work on unseen traffic levels (0.7x, 1.3x) and the real hour? | `test_summary.csv`, demand `x0.7`, `x1.3`, `real` vs `x1.0` |
| RQ3 | Can the policy run on cheap roadside hardware, and does 8-bit compression change its decisions? | `policy_hardware.csv` + the `int8` rows of `test_summary.csv` |
| Ablation | Does training on demand variants (vs the real hour only) help? Does a smaller network hurt? | runs with `--train-set real` and `--net 16 16` |

## 1. Install (once)

    pip install eclipse-sumo sumo-rl torch stable-baselines3 libsumo

`libsumo` is optional but runs SUMO inside Python, about 5x faster. The scripts use it
automatically. Set `USE_TRACI=1` to switch it off. Use the same setting for every run,
because the two can give slightly different numbers for the same seed.

## 2. How it works, in plain words

* **One episode = one simulated hour** of traffic from one demand file.
* Every **5 s** each traffic light's agent sees 21 numbers for cologne1: which phase is green,
  whether the minimum green has passed, and how full and how queued each incoming lane is.
  It then **picks the next green phase**. If the phase changes, SUMO shows yellow first,
  with the **same yellow length as the city's plan** (5 s cologne1, 3 s others), so RL gets
  no unfair advantage.
* **Reward:** how much the waiting time on that light's approaches went down since the last decision.
* **cologne3 has 3 lights.** One shared network controls all three. Each light's input is
  padded to the same length and tagged with which light it is (`common.Adapter`).
* **Training** cycles through the 20 training files. Every 10 episodes the current policy is
  tried on the 5 validation files with no random exploration. The best one is saved
  (`best_model.zip`). The **test files are never used during training or model selection.**
* **Main metric = average total waiting per vehicle**: time stopped on the road plus time
  waiting to enter the map. Cars still driving or still waiting to enter at the end of the
  hour are counted too. Without this, a controller could look good by blocking an entrance
  so cars never enter.

## 3. Doing the research, step by step

### Step A: data (already done)

    python make_splits.py           # builds splits/<scenario>/train|val|test (35 files each)
    python inspect_scenarios.py     # facts about each scenario

### Step B: give the baselines a fair chance (about 5 min)

    python evaluate.py --tune-actuated --scenarios cologne1 ingolstadt1 cologne3

This tries 9 actuated settings on the **validation** files and keeps the best one in
`results/<scenario>/actuated_params.json`. A reviewer will not accept an untuned actuated baseline.

### Step C: pilot, one short run to check everything (about 5–10 min)

    python train.py --scenario cologne1 --algo dqn --seed 0 --episodes 20

Open `results/cologne1/dqn_64x64_variants/seed0/train_log.csv` and plot `avg_total_wait`
against `episode`. It should go down. `val_log.csv` shows the validation checks.
Delete this folder before the main run, because `run_all.py` skips runs that already
have a `final_model.zip`.

### Step D: tune hyper-parameters (validation only)

The starting values are in `train.py` (`HYPER`). If you tune, **only look at `val_log.csv`**
and give every algorithm the **same tuning budget**, for example the same 3 learning rates
each (`--lr`). Write down what you tried; it goes in the paper.

### Step E: main training, 4 algorithms x 5 seeds x 3 scenarios

    python run_all.py --scenarios cologne1 ingolstadt1 cologne3 --workers 6

This is 60 runs of 200 episodes each. Short test runs took 5–10 s per episode for cologne1 and
ingolstadt1 and 15–35 s for cologne3. Validation adds roughly a third on top. So expect roughly
**half a day to a day on 6 cores**. Treat this as an estimate and check the times from the pilot.
If the PC restarts, run the same command again; finished runs are skipped.

### Step F: test everything once

    python evaluate.py --scenarios cologne1 ingolstadt1 cologne3 --int8 --workers 6

This runs every controller on every test file:
fixed, tuned actuated, random (5 seeds), and every trained run as FP32 and INT8.
On a given test file all controllers use the same SUMO seed.

### Step G: hardware

    python measure_policy.py

This gives size (KB) and time per decision for FP32 and INT8 networks. For the paper, also
run it **on the real target device** (copy this folder and `results/`). Numbers from a desktop
CPU are not roadside numbers. On a desktop CPU, INT8 is *smaller but slower* for networks
this tiny.

### Step H: ablations (pick the best algorithm from Step F; call it ALG)

    python run_all.py --scenarios cologne1 ingolstadt1 --algos ALG --train-set real    # no demand variants
    python run_all.py --scenarios cologne1 ingolstadt1 --algos ALG --net 16 16         # tiny network
    python run_all.py --scenarios cologne1 ingolstadt1 --algos ALG --net 256 256       # big network
    python evaluate.py --scenarios cologne1 ingolstadt1 --int8
    python measure_policy.py

Each configuration gets its own folder (`<algo>_<net>_<train-set>`), so all of them appear
side by side in the same summary table.

## 4. Reading the results

`results/test_summary.csv` has one row per scenario x demand level x controller x precision:

| column | meaning |
|---|---|
| `demand` | `x0.7` / `x1.3` = traffic levels never seen in training, `x1.0` = seen level (new variants), `real` = the original hour, `all` = all 10 test files |
| `avg_total_wait_mean` / `_std` | main result, seconds per vehicle; mean and std over the 5 training seeds |
| `avg_trip_mean`, `avg_stops_mean` | travel time and number of stops per vehicle |
| `finished_pct_mean` | % of vehicles that completed their trip; a drop means the network jammed |
| `vs_fixed_pct` | % less total waiting than fixed-time (positive = better) |

* **RQ1:** compare the algorithms on `all`. Report mean ± std over 5 seeds. Call a difference
  real only if it holds across seeds (for example a Welch t-test or Mann–Whitney U on the 5 per-seed values).
* **RQ2 (generalisation):** compare each controller's `vs_fixed_pct` at `x1.0` with `x0.7` and `x1.3`.
  A controller that generalises keeps its advantage. Be precise in the paper: the test
  changes the *amount* and *timing* of traffic, not *where cars travel* (every variant is
  built from the same real trips).
* **RQ3:** compare `int8` with `fp32` rows, especially at `x1.3`. A small change in decisions
  can matter a lot in heavy traffic.

`results/test_episodes.csv` holds every single test episode if you need to redo the statistics.

## 5. Rules that keep the comparison fair

1. Never use test files for tuning or for choosing a model. `train.py` never reads them.
2. Same decision interval, yellow, minimum green and SUMO version for every controller.
3. Same tuning budget for every algorithm, and tune actuated too (Step B).
4. 5 seeds for every RL configuration; report mean ± std, not the best seed.
5. Use `best_model.zip` (chosen on validation) for the test results.

## 6. Files

| file | what it does |
|---|---|
| `common.py` | shared settings, environment, metrics, Double DQN, INT8; used by everything below |
| `train.py` | trains one run (algorithm, scenario, seed) with validation and best-model saving |
| `run_all.py` | runs many `train.py` jobs in parallel; skips finished ones |
| `evaluate.py` | tunes actuated (`--tune-actuated`) and tests all controllers on all test files |
| `measure_policy.py` | size and speed of trained policies, FP32 vs INT8 |
| `make_splits.py`, `inspect_scenarios.py`, `inspect_network.py` | data preparation and description |
| `baselines.py`, `time_episodes.py`, `train_dqn.py` | older first versions, replaced by `evaluate.py` and `train.py` |
