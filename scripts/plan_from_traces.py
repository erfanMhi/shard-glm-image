#!/usr/bin/env python3
"""plan_from_traces.py <run_dir_with_stage_traces> [cap=16] [floor=6]: layers per stage proportional to each box's measured
speed (median compute ms per layer from STAGE_TRACE), so all stages take about the same time per chunk. Prints the plan."""
import json, glob, os, sys, statistics as st
R = sys.argv[1]; cap = int(sys.argv[2]) if len(sys.argv) > 2 else 16; floor = int(sys.argv[3]) if len(sys.argv) > 3 else 6; NL = 78
stages = []
for d in sorted(glob.glob(os.path.join(R, "stage*_*"))):
    f = os.path.join(d, "stage_trace.jsonl")
    if not os.path.exists(f): continue
    rows = [json.loads(l) for l in open(f) if l.strip() and json.loads(l).get("sp", 0) > 0]
    comp = st.median((r["t_comp1"] - r["t_recv1"]) * 1e3 for r in rows)
    stages.append((os.path.basename(d), comp))
ring = json.load(open(os.path.join(R, "manifest.json")))["ring"]
assert len(ring) == len(stages), (len(ring), len(stages))
per_layer = [comp / (rg["layers"][1] - rg["layers"][0] + 1) for (name, comp), rg in zip(stages, ring)]
speed = [1 / x for x in per_layer]; tot = sum(speed)
raw = [NL * s / tot for s in speed]
plan = [min(cap, max(floor, round(x))) for x in raw]
for _ in range(200):                          # fix rounding within the cap/floor bounds
    diff = NL - sum(plan)
    if diff == 0: break
    if diff > 0:
        cands = [k for k in range(len(plan)) if plan[k] < cap]
        if not cands: break
        i = max(cands, key=lambda k: raw[k] - plan[k]); plan[i] += 1
    else:
        cands = [k for k in range(len(plan)) if plan[k] > floor]
        if not cands: break
        i = min(cands, key=lambda k: raw[k] - plan[k]); plan[i] -= 1
pred = [pl * x for pl, x in zip(plan, per_layer)]
for (name, comp), rg, x, pl, pr in zip(stages, ring, per_layer, plan, pred):
    print(f"{name:<22} {rg['geo']:<20} layers {rg['layers'][0]:>2}-{rg['layers'][1]:<2} compute {comp:5.1f} ms = {x:4.2f} ms/layer -> plan {pl:>2} layers ({pr:5.1f} ms)", file=sys.stderr)
print(f"# max stage {max(comp for _, comp in stages):.1f} ms now -> {max(pred):.1f} ms with the plan", file=sys.stderr)
print(",".join(map(str, plan)))
