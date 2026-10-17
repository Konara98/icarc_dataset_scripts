"""Build train / validation / test traffic-demand files for each RESCO scenario.

Why: each RESCO scenario ships ONE hour of real traffic demand. Training and
testing on the same hour would be like testing a classifier on its training
images. So we generate variants of that hour, keep them in separate sets, and
never let the test set's variants be seen during training.

How each variant is made from the real hour:
  * demand scale   - keep a random fraction (scale < 1) or add duplicated trips
                     (scale > 1), so total traffic goes up or down
  * departure jitter - every trip's start time moves by up to +/-60 s
  * random seed    - each variant has its own seed, so no two are identical

Test contains scales never used in training (0.7 and 1.3) plus the unmodified
real hour, so it checks the learner on lighter, heavier, and real traffic.

Run:  python make_splits.py   ->  writes splits/<scenario>/{train,val,test}/*.rou.xml
                                  and prints the table of counts
"""
import os
try:
    import sumo as _sumo
    os.environ.setdefault("SUMO_HOME", _sumo.SUMO_HOME)   # pip eclipse-sumo sets this for you
except ImportError:
    pass

import os, glob, copy, random, csv, shutil, xml.etree.ElementTree as ET
import sumo_rl

RESCO = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "RESCO")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")
SCENARIOS = ["cologne1", "ingolstadt1", "cologne3"]
WINDOW = {"cologne": (25200, 28800), "ingolstadt": (57600, 61200)}
JITTER = 60.0
DECISION_EVERY = 5          # seconds between RL decisions
TRAIN_EPISODES = 200        # training budget per run (confirmed by learning curves)
SEEDS = 5                   # independent training runs per configuration

SPLITS = {
    #        scale values                                 variants per scale  seed base
    "train": ([0.8, 0.9, 1.0, 1.1, 1.2],                  4,                  1000),
    "val":   ([0.8, 0.9, 1.0, 1.1, 1.2],                  1,                  2000),
    "test":  ([0.7, 1.0, 1.3],                            3,                  3000),
}

def load(s):
    d = os.path.join(RESCO, s)
    rou = glob.glob(os.path.join(d, "*.rou.xml"))[0]
    root = ET.parse(rou).getroot()
    vtypes = [e for e in root if e.tag == "vType"]
    trips = [e for e in root if e.tag in ("trip", "vehicle")]
    return vtypes, trips

def variant(vtypes, trips, scale, seed, begin, end):
    rng = random.Random(seed)
    out = []
    for t in trips:
        n = int(scale) + (1 if rng.random() < scale - int(scale) else 0)
        for k in range(n):
            c = copy.deepcopy(t)
            if k:
                c.set("id", f"{t.get('id')}_d{k}")
            dep = float(t.get("depart"))
            if begin <= dep < end:              # only jitter trips inside the hour
                dep = min(max(dep + rng.uniform(-JITTER, JITTER), begin), end - 1)
            c.set("depart", f"{dep:.2f}")
            c.attrib.pop("arrival", None)
            out.append(c)
    out.sort(key=lambda e: float(e.get("depart")))
    root = ET.Element("routes")
    for v in vtypes:
        root.append(copy.deepcopy(v))
    root.extend(out)
    in_win = sum(1 for e in out if begin <= float(e.get("depart")) < end)
    return ET.ElementTree(root), in_win

rows = []
for s in SCENARIOS:
    begin, end = WINDOW["cologne" if s.startswith("cologne") else "ingolstadt"]
    vtypes, trips = load(s)
    real = sum(1 for t in trips if begin <= float(t.get("depart")) < end)
    for split, (scales, per, base) in SPLITS.items():
        d = os.path.join(OUT, s, split); os.makedirs(d, exist_ok=True)
        files, vehicles = 0, []
        if split == "test":                    # the unmodified real hour is a test file
            shutil.copy(glob.glob(os.path.join(RESCO, s, "*.rou.xml"))[0],
                        os.path.join(d, f"{s}_real.rou.xml"))
            files += 1; vehicles.append(real)
        for sc in scales:
            for i in range(per):
                seed = base + int(sc * 100) * 10 + i
                tree, n = variant(vtypes, trips, sc, seed, begin, end)
                tree.write(os.path.join(d, f"{s}_x{sc:.1f}_v{i}.rou.xml"))
                files += 1; vehicles.append(n)
        rows.append((s, split, files, min(vehicles), max(vehicles), sum(vehicles), real))

# ---- decisions ("records") per split, per training run -------------------
steps = int((WINDOW['cologne'][1] - WINDOW['cologne'][0]) / DECISION_EVERY)   # 720
print(f"\nDecisions per junction per 1-hour episode: {steps}\n")
print(f"{'scenario':12} {'split':6} {'files':>5} {'veh/file min-max':>18} {'vehicles total':>15}  real hour")
for s, split, f, lo, hi, tot, real in rows:
    print(f"{s:12} {split:6} {f:5d} {lo:>8d} - {hi:<7d} {tot:15d}  {real}")

with open(os.path.join(OUT, "split_summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["scenario", "split", "demand_files", "vehicles_min", "vehicles_max", "vehicles_total", "real_hour_vehicles"])
    w.writerows(rows)
print(f"\nwritten: {OUT}")
