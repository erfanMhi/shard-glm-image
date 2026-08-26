#!/usr/bin/env python3
"""summarize_run.py <run_dir>: one Markdown summary of a ring run: boxes + profiles, RTT mesh, every coordinator result
(mode, depth, K, tok/s, accept, stale, receipt match), and the trace analysis if present. Writes RESULTS.md in run_dir."""
import json, glob, os, sys, statistics as st
R = sys.argv[1]; out = []
def add(s=""): out.append(s)
# boxes
ids = [l.split() for l in open(os.path.join(R, "ring_ids.txt.kept"))] if os.path.exists(os.path.join(R, "ring_ids.txt.kept")) else []
add("## Boxes"); add(); add("| id | state | offer | $/h | host | GPU | power cap | copy GB/s | frame pickle p50 ms | disk MB/s | link (vast) |"); add("|---|---|---|---|---|---|---|---|---|---|---|")
for row in ids:
    iid, state, offer, price, host = row[:5]
    p = os.path.join(R, f"profile_{iid}.json"); g = t = d = {}
    if os.path.exists(p):
        pj = json.load(open(p)); g = (pj.get("gpu") or [{}])[0]; t = pj.get("torch") or {}; d = pj.get("disk") or {}
    add(f"| {iid} | {state} | {offer} | {price} | {host} | {g.get('name','?').replace('NVIDIA RTX PRO 6000 Blackwell ','')} | {g.get('power_limit_W','?')} W | {t.get('mem_bw_GBps_read_plus_write','?')} | {(t.get('pickle_roundtrip_ms') or {}).get('p50','?')} | {d.get('read_MBps_direct','?')} | |")
# mesh
m = glob.glob(os.path.join(R, "mesh*/mesh_rtt.json"))
if m:
    mj = json.load(open(sorted(m)[0])); short = [x.split(",")[0][:6] for x in mj["geo"]]
    add(); add("## RTT mesh (ms, row to column, TCP connect)"); add(); add("| | " + " | ".join(short) + " |"); add("|---|" + "---|" * len(short))
    for i, row in enumerate(mj["rtt"]): add(f"| {short[i]} | " + " | ".join(f"{x:.1f}" for x in row) + " |")
# results
add(); add("## Runs"); add(); add("| mode | depth | K | run | tok/s | mean accept | valid | stale | receipt match | wall s |"); add("|---|---|---|---|---|---|---|---|---|---|")
rows = []
for mf in sorted(glob.glob(os.path.join(R, "*", "manifest.json"))):
    d = json.load(open(mf)); mode = os.path.basename(os.path.dirname(mf)); a = d.get("args", {})
    for r in d.get("results", []):
        s = r.get("summary", ""); acc = valid = stale = ""
        import re
        mm = re.search(r"mean accept ([0-9.]+)", s); acc = mm.group(1) if mm else ""
        mm = re.search(r"(\d+) valid", s); valid = mm.group(1) if mm else ""
        mm = re.search(r"\+(\d+) stale", s); stale = mm.group(1) if mm else ""
        rows.append((mode, a.get("depth"), d.get("K"), r.get("run"), r.get("tok_s"), acc, valid, stale, r.get("matches_receipt"), r.get("wall_s")))
for row in rows: add("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
# per-mode medians
add(); add("## Medians per configuration"); add(); add("| mode | depth | K | runs | median tok/s | min | max |"); add("|---|---|---|---|---|---|---|")
grp = {}
for row in rows:
    if row[4] is None: continue
    grp.setdefault((row[0].split("_")[0] if not row[0].startswith("sweep") else "cg", row[1], row[2]), []).append(row[4])
for k, v in sorted(grp.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0, kv[0][2] or 0)):
    add(f"| {k[0]} | {k[1]} | {k[2]} | {len(v)} | {st.median(v):.2f} | {min(v):.2f} | {max(v):.2f} |")
# traces
for tdir in ("cgtrace2", "pipetrace", "cgtrace"):
    an = os.path.join(R, tdir, "ANALYSIS.txt")
    if os.path.exists(an): add(); add(f"## Trace analysis ({tdir})"); add(); add("```"); add(open(an).read().rstrip()); add("```")
open(os.path.join(R, "RESULTS.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
