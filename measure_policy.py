"""Can the policy run on a roadside unit?  Size and decision time, FP32 vs INT8.

For every trained configuration in results/ (one seed per configuration; all
seeds of a configuration have the same size) this reports:
  params        number of weights
  KB            size of the saved network in bytes / 1024 (what must fit in storage)
  kFLOPs        arithmetic per decision
  us/decision   median time for ONE decision on ONE CPU core (batch of 1)
for the FP32 network and its INT8-quantised copy.

Before any training it measures untrained networks of the usual sizes instead,
so you can plan. To report real roadside numbers, copy results/ and this
script to the device (e.g. a Raspberry Pi; needs Python + torch) and run it there.

Run:  python measure_policy.py            -> prints the table, writes results/policy_hardware.csv
"""
import common as C

import csv, glob, io, os, platform, statistics, time
import torch as th
import torch.nn as nn

th.set_num_threads(1)


def saved_kb(module):
    buf = io.BytesIO()
    th.save(module.state_dict(), buf)
    return buf.tell() / 1024


def us_per_decision(module, inputs, n=3000):
    x = th.rand(1, inputs)
    with th.no_grad():
        for _ in range(300):
            module(x)
        times = []
        for _ in range(n):
            t = time.perf_counter()
            module(x).argmax(1)
            times.append(time.perf_counter() - t)
    return statistics.median(times) * 1e6


def describe(name, net):
    linears = [m for m in net if isinstance(m, nn.Linear)]
    inputs = linears[0].in_features
    row = dict(config=name, inputs=inputs, outputs=linears[-1].out_features,
               hidden="x".join(str(l.out_features) for l in linears[:-1]),
               params=sum(p.numel() for p in net.parameters()),
               kflops=round(2 * sum(l.in_features * l.out_features for l in linears) / 1e3, 1))
    q = C.quantize_int8(net)
    row.update(fp32_kb=round(saved_kb(net), 1), int8_kb=round(saved_kb(q), 1),
               fp32_us=round(us_per_decision(net, inputs), 1), int8_us=round(us_per_decision(q, inputs), 1))
    return row


def main():
    rows = []
    models = sorted(glob.glob(os.path.join(C.RESULTS, "*", "*", "seed*", "best_model.zip")))
    seen = set()
    for f in models:
        scenario, tag = f.split(os.sep)[-4], f.split(os.sep)[-3]
        if (scenario, tag) in seen:
            continue
        seen.add((scenario, tag))
        algo = C.parse_tag(tag)[0]
        rows.append(describe(f"{scenario} {tag}", C.export_policy(C.ALGOS[algo].load(f, device="cpu"))))
    if not rows:
        print("No trained models yet: measuring untrained networks (cologne1: 21 inputs, 4 phases).\n")
        for hidden in ([16, 16], [32, 32], [64, 64], [256, 256]):
            layers, prev = [], 21
            for w in hidden:
                layers += [nn.Linear(prev, w), nn.ReLU()]
                prev = w
            rows.append(describe(f"untrained {'x'.join(map(str, hidden))}", nn.Sequential(*layers, nn.Linear(prev, 4)).eval()))

    print(f"CPU: {platform.processor() or platform.machine()}, 1 thread\n")
    print(f"{'configuration':36} {'params':>7} {'kFLOPs':>7} {'FP32 KB':>8} {'INT8 KB':>8} {'FP32 us':>8} {'INT8 us':>8}")
    for r in rows:
        print(f"{r['config']:36} {r['params']:7d} {r['kflops']:7.1f} {r['fp32_kb']:8.1f} {r['int8_kb']:8.1f}"
              f" {r['fp32_us']:8.1f} {r['int8_us']:8.1f}")
    os.makedirs(C.RESULTS, exist_ok=True)
    out = os.path.join(C.RESULTS, "policy_hardware.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) + ["cpu"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, cpu=platform.processor() or platform.machine()))
    print(f"\nwritten: {out}")
    print("Note: one decision is needed every 5 s, so any time below ~1 s per decision is fast enough;")
    print("the useful comparison is memory and energy on the target device, not raw speed here.")


if __name__ == "__main__":
    main()
