"""Inspect RESCO scenarios bundled with sumo-rl: demand, junctions, signal plans.

Run:  python inspect_scenarios.py
Needs: pip install eclipse-sumo sumo-rl
"""
import os
try:
    import sumo as _sumo
    os.environ.setdefault("SUMO_HOME", _sumo.SUMO_HOME)   # pip eclipse-sumo sets this for you
except ImportError:
    pass

import os, glob, xml.etree.ElementTree as ET
import sumo_rl

RESCO = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "RESCO")
SCEN = ["cologne1", "ingolstadt1", "cologne3", "ingolstadt7"]

def window(cfg):
    t = ET.parse(cfg).getroot().find("time")
    return float(t.find("begin").get("value")), float(t.find("end").get("value"))

for s in SCEN:
    d = os.path.join(RESCO, s)
    begin, end = window(os.path.join(d, f"{s}.sumocfg"))
    rou = glob.glob(os.path.join(d, "*.rou.xml"))[0]
    net = os.path.join(d, f"{s}.net.xml")

    # demand: trips / vehicles / flows departing inside the window
    r = ET.parse(rou).getroot()
    total = in_win = 0
    vclass = {}
    types = {vt.get("id"): vt.get("vClass", "passenger") for vt in r.iter("vType")}
    for tag in ("trip", "vehicle"):
        for v in r.iter(tag):
            dep = float(v.get("depart"))
            total += 1
            if begin <= dep < end:
                in_win += 1
                c = types.get(v.get("type"), "passenger")
                vclass[c] = vclass.get(c, 0) + 1
    flows = sum(1 for _ in r.iter("flow"))

    # network: traffic lights, phases, controlled lanes
    n = ET.parse(net).getroot()
    tls = n.findall("tlLogic")
    tl_types = sorted({t.get("type") for t in tls})
    phases = [len(t.findall("phase")) for t in tls]
    green = [sum(1 for p in t.findall("phase") if "G" in p.get("state") or "g" in p.get("state")) for t in tls]
    lanes = {c.get("from") + "_" + c.get("fromLane") for c in n.iter("connection") if c.get("tl")}

    print(f"\n== {s}")
    print(f"  window           : {begin:.0f}-{end:.0f} s  ({(end-begin)/3600:.1f} h, "
          f"{int(begin//3600):02d}:00-{int(end//3600):02d}:00)")
    print(f"  vehicles in file : {total}   departing in window: {in_win}   flows: {flows}")
    print(f"  vehicle classes  : {vclass}")
    print(f"  traffic lights   : {len(tls)}  type(s): {tl_types}")
    print(f"  phases per light : {phases}   green phases: {green}")
    print(f"  signal-controlled incoming lanes: {len(lanes)}")
