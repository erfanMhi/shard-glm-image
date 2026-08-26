#!/usr/bin/env python3
"""analyze_trace.py <run_dir>: per-stage service time and per-chunk traversal from the STAGE_TRACE / COORD_TRACE files
pulled by launch_ring.py --trace. Prints medians so the round-time model's 38 ms/stage and 376 ms traversal can be checked."""
import json, glob, os, sys, statistics as st
R = sys.argv[1]
def med(v): return f"{st.median(v):7.1f}" if v else "      -"
print(f"{'stage':<22} {'n':>4} {'recv ms':>8} {'compute ms':>11} {'send ms':>8} {'service ms':>11} {'gap ms':>8}   (recv = header seen -> tensor decoded; service = recv..send done; gap = idle between messages)")
for d in sorted(glob.glob(os.path.join(R, "stage*_*"))):
    f = os.path.join(d, "stage_trace.jsonl")
    if not os.path.exists(f): continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    rows = [r for r in rows if r.get("sp", 0) > 0]            # skip the prefill message
    recv = [(r["t_recv1"] - r["t_recv0"]) * 1e3 for r in rows]
    comp = [(r["t_comp1"] - r["t_recv1"]) * 1e3 for r in rows]
    send = [(r["t_send1"] - r["t_comp1"]) * 1e3 for r in rows]
    serv = [(r["t_send1"] - r["t_recv0"]) * 1e3 for r in rows]
    gaps = [(b["t_recv0"] - a["t_send1"]) * 1e3 for a, b in zip(rows, rows[1:])]
    print(f"{os.path.basename(d):<22} {len(rows):>4} {med(recv)} {med(comp):>11} {med(send)} {med(serv):>11} {med(gaps)}")
for f in glob.glob(os.path.join(R, "coord_trace_*.jsonl")):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    ch = [r for r in rows if "t_send" in r and "t_recv" in r]
    trav = [(r["t_recv"] - r["t_send"]) * 1e3 for r in ch]
    draft = [(r["t_draft1"] - r["t_draft0"]) * 1e3 for r in ch if "t_draft0" in r]
    sends = sorted(r["t_send"] for r in ch); period = [(b - a) * 1e3 for a, b in zip(sends, sends[1:])]
    verdicts = {}
    for r in ch: verdicts[r.get("verdict")] = verdicts.get(r.get("verdict"), 0) + 1
    print(f"\ncoordinator ({os.path.basename(f)}): {len(ch)} chunks | traversal median {med(trav)} ms (p10 {sorted(trav)[len(trav)//10]:.0f}, p90 {sorted(trav)[-max(1,len(trav)//10)]:.0f}) | draft per chunk median {med(draft)} ms | send period median {med(period)} ms | verdicts {verdicts}")
    summ = [r for r in rows if "summary" in r or "tok_s" in r]
    if summ: print("summary:", summ[-1])
