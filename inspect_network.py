"""List the junctions and edges (roads) of each RESCO scenario's road network.

The .rou.xml files in splits/ only hold vehicle trips. The road network itself
(junctions, edges, lanes, traffic lights) is the <scenario>.net.xml that ships
inside sumo-rl.

Run:  python inspect_network.py                 (cologne1, ingolstadt1, cologne3)
      python inspect_network.py cologne3 --all  (also list every edge)
Writes network_<scenario>_edges.csv and network_<scenario>_junctions.csv.
"""
import os
try:
    import sumo as _sumo
    os.environ.setdefault("SUMO_HOME", _sumo.SUMO_HOME)   # pip eclipse-sumo sets this for you
except ImportError:
    pass

import os, sys, csv, xml.etree.ElementTree as ET
import sumo_rl

RESCO = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "RESCO")
OUT = os.path.dirname(os.path.abspath(__file__))


def inspect(s, show_all=False):
    net = ET.parse(os.path.join(RESCO, s, f"{s}.net.xml")).getroot()

    # real roads only; "internal" edges are the short paths drawn inside a junction
    edges = [e for e in net.findall("edge") if e.get("function") != "internal"]
    junctions = [j for j in net.findall("junction") if j.get("type") != "internal"]
    tls = {t.get("id"): t for t in net.findall("tlLogic")}

    # which incoming lanes / edges each traffic light controls (a light can cover
    # a cluster of several junctions, so go by its connections, not a junction id)
    controlled, edges_in, edges_out = {}, {}, {}
    for c in net.iter("connection"):
        if c.get("tl"):
            controlled.setdefault(c.get("tl"), set()).add(c.get("from") + "_" + c.get("fromLane"))
            edges_in.setdefault(c.get("tl"), set()).add(c.get("from"))
            edges_out.setdefault(c.get("tl"), set()).add(c.get("to"))

    types = {}
    for j in junctions:
        types[j.get("type")] = types.get(j.get("type"), 0) + 1
    length_km = sum(float(e.find("lane").get("length")) for e in edges) / 1000

    print(f"\n== {s}")
    print(f"  junctions : {len(junctions)}  by type: {types}")
    print(f"  edges     : {len(edges)}  ({sum(len(e.findall('lane')) for e in edges)} lanes, "
          f"{length_km:.1f} km of road)")
    print(f"  traffic lights (= RL agents): {len(tls)}")

    for tid, tl in tls.items():
        incoming = sorted(edges_in.get(tid, []))
        outgoing = sorted(edges_out.get(tid, []))
        phases = tl.findall("phase")
        green = [p for p in phases if "y" not in p.get("state") and ("G" in p.get("state") or "g" in p.get("state"))]
        print(f"\n  traffic light '{tid}'")
        print(f"    incoming edges : {incoming}")
        print(f"    outgoing edges : {outgoing}")
        print(f"    controlled incoming lanes: {len(controlled.get(tid, []))}")
        print(f"    phases: {len(phases)} ({len(green)} green)")
        for p in phases:
            print(f"      {p.get('duration'):>5}s  {p.get('state')}")

    with open(os.path.join(OUT, f"network_{s}_edges.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["edge_id", "from_junction", "to_junction", "lanes", "length_m", "speed_limit_kmh", "name"])
        for e in edges:
            lane = e.find("lane")
            w.writerow([e.get("id"), e.get("from"), e.get("to"), len(e.findall("lane")),
                        f"{float(lane.get('length')):.1f}", f"{float(lane.get('speed')) * 3.6:.0f}",
                        e.get("name", "")])
    with open(os.path.join(OUT, f"network_{s}_junctions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["junction_id", "type", "x", "y", "incoming_lanes"])
        for j in junctions:
            w.writerow([j.get("id"), j.get("type"), j.get("x"), j.get("y"),
                        len(j.get("incLanes", "").split())])

    if show_all:
        print(f"\n  all edges (id: from -> to, lanes):")
        for e in edges:
            print(f"    {e.get('id')}: {e.get('from')} -> {e.get('to')}, {len(e.findall('lane'))}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for s in args or ["cologne1", "ingolstadt1", "cologne3"]:
        inspect(s, show_all="--all" in sys.argv)
