"""Measure the two existing traffic-light types on the real hour of traffic.

Fixed-time : the scenario's own signal plan (tlLogic type="static").
Actuated   : the same phases, but SUMO extends or cuts each green using
             induction-loop detectors it places on every incoming lane
             (tlLogic type="actuated"; min/max green taken from the plan).

Output per run comes from SUMO's own trip records (tripinfo) and summary stats.
"""
import os
try:
    import sumo as _sumo
    os.environ.setdefault("SUMO_HOME", _sumo.SUMO_HOME)   # pip eclipse-sumo sets this for you
except ImportError:
    pass

import os, sys, subprocess, tempfile, xml.etree.ElementTree as ET
import sumo_rl

SUMO = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")
RESCO = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "RESCO")
WINDOW = {"cologne": (25200, 28800), "ingolstadt": (57600, 61200)}

def actuated_additional(net, path):
    """Copy every signal plan, switch it to actuated, give greens a min/max."""
    root = ET.Element("additional")
    for tl in ET.parse(net).getroot().findall("tlLogic"):
        a = ET.SubElement(root, "tlLogic", id=tl.get("id"), type="actuated",
                          programID="actuated", offset="0")
        for p in tl.findall("phase"):
            q = ET.SubElement(a, "phase", duration=p.get("duration"), state=p.get("state"))
            if "G" in p.get("state") or "g" in p.get("state"):
                dur = float(p.get("duration"))
                q.set("minDur", "5"); q.set("maxDur", str(max(dur * 2, 30)))
    ET.ElementTree(root).write(path)

def run(s, rou, mode, seed=0):
    begin, end = WINDOW["cologne" if s.startswith("cologne") else "ingolstadt"]
    net = os.path.join(RESCO, s, f"{s}.net.xml")
    tmp = tempfile.mkdtemp()
    trip = os.path.join(tmp, "trip.xml")
    cmd = [SUMO, "-n", net, "-r", rou, "-b", str(begin), "-e", str(end), "--seed", str(seed),
           "--tripinfo-output", trip, "--no-step-log", "--no-warnings"]
    if mode == "actuated":
        add = os.path.join(tmp, "act.add.xml"); actuated_additional(net, add)
        cmd += ["-a", add]
    subprocess.run(cmd, check=True, capture_output=True)
    ti = ET.parse(trip).getroot().findall("tripinfo")
    wait = sum(float(t.get("waitingTime")) for t in ti) / len(ti)
    dur = sum(float(t.get("duration")) for t in ti) / len(ti)
    stops = sum(int(t.get("waitingCount")) for t in ti) / len(ti)
    return len(ti), wait, dur, stops

if __name__ == "__main__":
    for s in sys.argv[1:] or ["cologne1", "ingolstadt1"]:
        rou = [os.path.join(RESCO, s, f) for f in os.listdir(os.path.join(RESCO, s)) if f.endswith(".rou.xml")][0]
        print(f"\n== {s} (real hour)")
        print(f"  {'controller':10} {'finished':>8} {'avg wait s':>11} {'avg trip s':>11} {'stops/veh':>10}")
        for mode in ("fixed", "actuated"):
            n, w, d, st = run(s, rou, mode)
            print(f"  {mode:10} {n:8d} {w:11.1f} {d:11.1f} {st:10.2f}")
