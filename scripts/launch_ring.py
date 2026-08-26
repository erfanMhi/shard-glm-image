#!/usr/bin/env python3
"""launch_ring.py: drive the GLM-5.2 NVFP4 ring on rented vast.ai boxes, from the Mac.

Adapted from shard's phase0/launch_swarm.py (same vast conventions: ports["29600/tcp"][0]["HostPort"],
public_ipaddr, ssh via ports["22/tcp"]). The boxes run the shard-glm image, so there is no bootstrap step;
the venv is verified instead. Steps:

  1. wait until every id is running with 29600 mapped; refuse instances not labeled erfan-*
  2. verify /root/vmoe (import vllm) on every box
  3. all-pairs TCP RTT mesh (launch_swarm.mesh_rtt; cached in OUT/mesh_rtt.json)
  4. ring order = shard.topology.optimal_loop with the coordinator as depot
  5. 78 layers -> contiguous blocks in ring order (13 per stage with 6 stages)
  6. parallel node_fetch: stages --layers, coordinator --coord --draft REPO
  7. stages: --ring for ring-class modes (tail --next = coordinator ip:mapped 29600), plain chain for
     relay-class modes (tail has no --next); restarted only when the class changes
  8. the coordinator command, --runs times (one process per run), stdout/dump/trace pulled into OUT

Mode -> coordinator command (head = first stage's public_ip:mapped port, P = prompt, N = --max-new):
  plain    kv.py    coord --stage head --prompt P --max-new N                          relay-back, 1 token/round trip
  relay6   draft.py coord --stage head --prompt P --max-new N --K 6                    GLM-4-9B draft, relay-back
  direct6  draft.py coord --stage head --ret-port 29600 --prompt P --max-new N --K 6   ring direct return
  pipe     pipe.py  coord --stage head --ret-port 29600 --depth 6 --K 2 ...            async pipelined verify
  cg       cg.py    coord --stage head --ret-port 29600 --depth 6 --K 2 --compile --dump   the receipt's engine
  cgeager  cg.py    same without --compile (the receipt's lossless reference)
  cgplain  cg.py    coord --stage head --ret-port 29600 --plain --dump                 1-token greedy over the ring
Only cg/cgeager/cgplain have --dump (token ids -> sha256 check against the receipt); the others are read from stdout.

Example:
  python launch_ring.py --ids 1,2,3,4,5,6,7 --coord-id 7 --mode cg --runs 3 --trace --out runs/cg
"""
import os, sys, json, time, shlex, hashlib, subprocess, argparse, concurrent.futures as cf

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD = os.environ.get("SHARD_REPO", os.path.join(os.path.dirname(HERE), "shard"))   # shard-glm-image/shard (no .git)
sys.path.insert(0, SHARD); sys.path.insert(0, os.path.join(SHARD, "phase0"))
import launch_swarm as LS                     # mesh_rtt / solve_order / assign_layers / vast helpers

VASTAI = os.path.expanduser("~/.vastcli/bin/vastai")
KEY = os.path.expanduser("~/.ssh/id_ed25519")
LS.KEY = KEY                                   # their helpers read this module global for ssh -i
STAGE_PORT = 29600
NLAYERS = 78
PY = "/root/vmoe/bin/python"
RECEIPT_SHA = "d9e61275084cb2bf74b44aaaceec65c6f9882bd66bf2a3de6fc81104d73a7d66"
REAP = ("nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; "
        f"fuser -k {STAGE_PORT}/tcp 2>/dev/null; sleep 2; ")   # never pkill -f glm_swarm (self-kills the ssh shell)

MODES = {
    "plain":   dict(cls="relay", cmd="glm_swarm_nvfp4_kv.py coord --stage {head} --prompt {prompt} --max-new {n}", dump=False, K=None),
    "relay6":  dict(cls="relay", cmd="glm_swarm_nvfp4_draft.py coord --stage {head} --prompt {prompt} --max-new {n} --K {K}", dump=False, K=6),
    "direct6": dict(cls="ring",  cmd="glm_swarm_nvfp4_draft.py coord --stage {head} --ret-port {port} --prompt {prompt} --max-new {n} --K {K}", dump=False, K=6),
    "pipe":    dict(cls="ring",  cmd="glm_swarm_nvfp4_pipe.py coord --stage {head} --ret-port {port} --depth {depth} --K {K} --prompt {prompt} --max-new {n}", dump=False, K=2),
    "cg":      dict(cls="ring",  cmd="glm_swarm_nvfp4_cg.py coord --stage {head} --ret-port {port} --depth {depth} --K {K} --compile --prompt {prompt} --max-new {n}", dump=True, K=2),
    "cgeager": dict(cls="ring",  cmd="glm_swarm_nvfp4_cg.py coord --stage {head} --ret-port {port} --depth {depth} --K {K} --prompt {prompt} --max-new {n}", dump=True, K=2),
    "cgplain": dict(cls="ring",  cmd="glm_swarm_nvfp4_cg.py coord --stage {head} --ret-port {port} --plain --prompt {prompt} --max-new {n}", dump=True, K=None),
}

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---------- vast ----------
def vast_json(args, timeout=120):
    r = subprocess.run([VASTAI] + args + ["--raw"], capture_output=True, text=True, timeout=timeout)
    out = r.stdout
    for cut in ("[", "{"):
        k = out.find(cut)
        if k >= 0:
            try: return json.loads(out[k:])
            except Exception: pass
    raise RuntimeError(f"vastai {' '.join(args)} returned no JSON: {(out + r.stderr)[-300:]}")

def instances():
    d = vast_json(["show", "instances"])
    return d if isinstance(d, list) else d.get("instances", [])

def mapped_port(inst, cport=STAGE_PORT):
    p = (inst.get("ports") or {}).get(f"{cport}/tcp")
    return int(p[0]["HostPort"]) if p else None

def wait_running(ids, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        insts = {i["id"]: i for i in instances() if i["id"] in ids}
        missing = [i for i in ids if i not in insts]
        if missing: raise RuntimeError(f"instance ids not on this account: {missing}")
        ok = all(i.get("actual_status") == "running" and mapped_port(i) and (i.get("ports") or {}).get("22/tcp") and i.get("public_ipaddr")
                 for i in insts.values())
        if ok: return [insts[i] for i in ids]
        log("waiting:", {i: (insts[i].get("actual_status"), mapped_port(insts[i])) for i in ids})
        time.sleep(15)
    raise RuntimeError(f"nodes not all running with :{STAGE_PORT} mapped after {timeout}s")

# ---------- ssh ----------
def ssh_cmd(inst):
    return ["ssh", "-i", KEY, "-p", str(inst["ports"]["22/tcp"][0]["HostPort"])] + LS.SSHO + [f"root@{inst['public_ipaddr']}"]

def rssh(inst, cmd, timeout=120):
    r = subprocess.run(ssh_cmd(inst) + [cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr

def scp_from(inst, remote_paths, dest):
    os.makedirs(dest, exist_ok=True)
    port = str(inst["ports"]["22/tcp"][0]["HostPort"])
    for rp in remote_paths:
        subprocess.run(["scp", "-i", KEY, "-P", port] + LS.SSHO + [f"root@{inst['public_ipaddr']}:{rp}", dest],
                       capture_output=True, text=True, timeout=600)

def ep(inst):
    return f"{inst['public_ipaddr']}:{mapped_port(inst)}"

def pmap(fn, items):
    with cf.ThreadPoolExecutor(max_workers=max(1, len(items))) as ex:
        return list(ex.map(fn, items))

# ---------- steps ----------
def verify_env(nodes):
    def one(n):
        rc, out, err = rssh(n, f"{PY} -c 'import torch,vllm,transformers;print(\"OK\",torch.__version__,vllm.__version__,transformers.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))'; "
                               f"nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; test -f /root/glm_swarm_nvfp4_kv.py && echo SCRIPTS_OK", 180)
        return n["id"], out.strip().replace("\n", " | "), err[-200:]
    for nid, out, err in pmap(one, nodes):
        log(f"  {nid}: {out}" + ("" if "OK" in out and "SCRIPTS_OK" in out else f"  !! {err}"))
        if "OK" not in out or "SCRIPTS_OK" not in out: raise RuntimeError(f"box {nid} failed the image check")

def mesh(nodes, out_dir, remesh):
    path = os.path.join(out_dir, "mesh_rtt.json")
    if os.path.exists(path) and not remesh:
        d = json.load(open(path))
        if d.get("ids") == [n["id"] for n in nodes]:
            log("mesh: reusing", path); return d["rtt"]
    log("mesh: measuring all-pairs TCP connect RTT (kills any listener on :29600 first)")
    rtt = LS.mesh_rtt(nodes)                  # NxN ms, node->node, uses public_ip:mapped 29600
    json.dump({"ids": [n["id"] for n in nodes], "geo": [n.get("geolocation") for n in nodes], "rtt": rtt}, open(path, "w"), indent=1)
    return rtt

def stage_launch_cmd(blk, nxt, ring, trace):
    args = f"stage --layers {' '.join(map(str, blk))} --port {STAGE_PORT}" + (f" --next {nxt}" if nxt else "") + (" --ring" if ring else "")
    env = "STAGE_TRACE=/root/stage_trace.jsonl " if trace else ""
    return (REAP + "rm -f /root/stage.log /root/stage_trace.jsonl; cd /root && "
            f"echo {shlex.quote(('ring' if ring else 'relay') + ' ' + args)} > /root/stage.class && "
            f"setsid env {env}{PY} glm_swarm_nvfp4_kv.py {args} > /root/stage.log 2>&1 < /dev/null & echo launched")

def stage_state(inst):
    rc, out, err = rssh(inst, "cat /root/stage.class 2>/dev/null; echo '|'; grep -c WARM /root/stage.log 2>/dev/null; echo '|'; "
                              f"fuser {STAGE_PORT}/tcp 2>/dev/null | wc -w", 60)
    parts = out.split("|")
    cls = parts[0].strip() if parts else ""
    warm = parts[1].strip() not in ("", "0") if len(parts) > 1 else False
    listening = parts[2].strip() not in ("", "0") if len(parts) > 2 else False
    return cls, warm and listening

def ensure_stages(chain, coord, ring, trace, timeout):
    """chain = [(inst, layers)] in ring order. Launch/relaunch each stage; wait for WARM."""
    eps = [ep(inst) for inst, _ in chain]
    todo = []
    for i, (inst, blk) in enumerate(chain):
        nxt = eps[i + 1] if i + 1 < len(chain) else (ep(coord) if ring else None)
        want = ("ring" if ring else "relay") + " " + f"stage --layers {' '.join(map(str, blk))} --port {STAGE_PORT}" + (f" --next {nxt}" if nxt else "") + (" --ring" if ring else "")
        cls, alive = stage_state(inst)
        if alive and cls == want:
            log(f"  stage{i} {inst['id']} layers {blk[0]}-{blk[-1]}: already up ({cls.split()[0]})"); continue
        todo.append((i, inst, blk, nxt))
    def launch(item):
        i, inst, blk, nxt = item
        rssh(inst, stage_launch_cmd(blk, nxt, ring, trace), 90)
        return i
    if todo:
        for i, inst, blk, nxt in todo:
            log(f"  stage{i} {inst['id']} ({inst.get('geolocation')}): layers {blk[0]}-{blk[-1]} -> {nxt or '(tail, relay-back)'}")
        pmap(launch, todo)
    def warm(item):
        i, inst, blk, nxt = item
        t0 = time.time()
        while time.time() - t0 < timeout:
            rc, out, err = rssh(inst, "grep -c WARM /root/stage.log 2>/dev/null; grep -ciE 'exit status|Traceback' /root/stage.log 2>/dev/null", 60)
            nums = [x for x in out.split() if x.isdigit()]
            if nums and nums[0] != "0": return i, True, time.time() - t0
            if len(nums) > 1 and nums[1] != "0":
                rc, tail, _ = rssh(inst, "tail -15 /root/stage.log", 30); return i, False, tail
            time.sleep(15)
        return i, False, "timeout"
    for i, ok, info in pmap(warm, todo):
        log(f"  stage{i}: {'WARM in %.0fs' % info if ok else 'FAILED ' + str(info)[-600:]}")
        if not ok: raise RuntimeError(f"stage{i} did not warm")

def coord_cmd(mode, head, prompt, n, K, depth, dump):
    m = MODES[mode]
    cmd = m["cmd"].format(head=head, port=STAGE_PORT, prompt=shlex.quote(prompt), n=n, K=K, depth=depth)
    if m["dump"] and dump: cmd += f" --dump {dump}"
    return cmd

def parse_result(stdout):
    r = {}
    for ln in stdout.splitlines():
        if ln.startswith("GENERATED") or ln.startswith("PLAIN GREEDY"):
            r["summary"] = ln.strip()
            try: r["tok_s"] = float(ln.split("=")[1].split("tok/s")[0])
            except Exception: pass
        elif ln.startswith("  time split"): r["time_split"] = ln.strip()
        elif ln.startswith("decoded:"): r["decoded"] = ln[len("decoded:"):].strip()[:300]
    return r

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True, help="comma-separated vast instance ids (stages + coordinator)")
    ap.add_argument("--coord-id", type=int, required=True)
    ap.add_argument("--mode", choices=list(MODES), default="cg")
    ap.add_argument("--prompt", default="def quicksort(arr):")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--runs", type=int, default=3, help="coordinator processes to run (0 = only bring the ring up)")
    ap.add_argument("--out", default="runs/" + time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--trace", action="store_true", help="STAGE_TRACE on stages, COORD_TRACE on the coordinator")
    ap.add_argument("--draft-repo", default="zai-org/GLM-4-9B-0414", help="GLM-4-9B draft HF id (verify with glm_draft_compat.py first)")
    ap.add_argument("--K", type=int, default=None, help="draft length (default: 6 for relay6/direct6, 2 for pipe/cg)")
    ap.add_argument("--depth", type=int, default=6, help="chunks in flight (pipe/cg)")
    ap.add_argument("--order", default="", help="explicit ring order as comma-separated stage ids (skips the topology solve)")
    ap.add_argument("--remesh", action="store_true", help="re-measure the RTT mesh even if OUT/mesh_rtt.json exists")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--stage-timeout", type=int, default=2400, help="seconds to wait for a stage to print WARM")
    ap.add_argument("--allow-any-label", action="store_true", help="operate on instances whose label is not erfan-*")
    ap.add_argument("--auto-coord", action="store_true", help="after the RTT mesh, pick the coordinator that minimizes the loop cost (overrides --coord-id)")
    ap.add_argument("--mesh-only", action="store_true", help="measure the mesh, print the ring choice, write manifest, and exit")
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",") if x.strip()]
    if a.coord_id not in ids: raise SystemExit("--coord-id must be one of --ids")
    K = a.K if a.K is not None else (MODES[a.mode]["K"] or 2)
    os.makedirs(a.out, exist_ok=True)
    manifest = {"args": vars(a), "K": K, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    log("[1] instances")
    nodes = wait_running(ids)
    for n in nodes:
        lbl = n.get("label") or ""
        log(f"    {n['id']} label={lbl!r} {n.get('geolocation')} {n.get('gpu_name')} ip={n['public_ipaddr']} :{mapped_port(n)} ssh:{n['ports']['22/tcp'][0]['HostPort']}")
        if not lbl.startswith("erfan-") and not a.allow_any_label:
            raise SystemExit(f"instance {n['id']} label {lbl!r} is not erfan-*; refusing (pass --allow-any-label to override)")
    coord = next(n for n in nodes if n["id"] == a.coord_id)
    manifest["nodes"] = [{k: n.get(k) for k in ("id", "label", "geolocation", "gpu_name", "public_ipaddr", "cuda_max_good", "inet_up", "inet_down", "dph_total")} | {"stage_port": mapped_port(n)} for n in nodes]

    log("[2] image check (vmoe venv + scripts)")
    verify_env(nodes)

    if a.order:
        order_ids = [int(x) for x in a.order.split(",")]
        order = [ids.index(i) for i in order_ids]; cost = None
        log(f"[3-4] explicit ring order: {order_ids}")
    else:
        log("[3] RTT mesh")
        rtt = mesh(nodes, a.out, a.remesh)
        for n, row in zip(nodes, rtt): log("    " + f"{n['id']:>9} " + " ".join(f"{x:6.1f}" for x in row))
        log("[4] topology (shard.topology.optimal_loop, coordinator as depot)")
        if a.auto_coord:
            cands = []
            for ci in range(len(nodes)):
                o, c = LS.solve_order(rtt, ci); cands.append((c, ci, o))
                log(f"    depot {nodes[ci]['id']} ({nodes[ci].get('geolocation')}): loop {c:.1f} ms")
            cands.sort(); a.coord_id = nodes[cands[0][1]]["id"]; coord = nodes[cands[0][1]]
            manifest["auto_coord"] = [{"id": nodes[ci]["id"], "geo": nodes[ci].get("geolocation"), "loop_ms": round(c, 1)} for c, ci, o in cands]
            log(f"    -> coordinator {a.coord_id} ({coord.get('geolocation')})")
        order, cost = LS.solve_order(rtt, ids.index(a.coord_id))
        log(f"    loop cost {cost:.1f} ms; stage order: {[nodes[i]['id'] for i in order]} ({[nodes[i].get('geolocation') for i in order]})")
    chain = LS.assign_layers(order, nodes, NLAYERS)
    manifest["ring"] = [{"id": inst["id"], "geo": inst.get("geolocation"), "layers": [blk[0], blk[-1]], "ep": ep(inst)} for inst, blk in chain]
    manifest["loop_cost_ms"] = cost
    for i, (inst, blk) in enumerate(chain): log(f"    stage{i}: {inst['id']} ({inst.get('geolocation')}) layers {blk[0]}-{blk[-1]}")
    log(f"    coord: {coord['id']} ({coord.get('geolocation')}) ret {ep(coord)}")
    manifest["coord_id"] = coord["id"]; manifest["order_ids"] = [inst["id"] for inst, _ in chain]
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    if a.mesh_only:
        log(f"[mesh-only] coord {coord['id']} order {manifest['order_ids']} -> {a.out}/manifest.json"); return

    if not a.skip_fetch:
        log("[6] node_fetch (parallel; idempotent); pushing the current root/node_fetch.py first")
        nf = os.path.join(os.path.dirname(HERE), "root", "node_fetch.py")
        def push(n):
            port = str(n["ports"]["22/tcp"][0]["HostPort"])
            subprocess.run(["scp", "-i", KEY, "-P", port] + LS.SSHO + [nf, f"root@{n['public_ipaddr']}:/root/node_fetch.py"], capture_output=True, timeout=120)
            return n["id"]
        pmap(push, nodes)
        tok = os.environ.get("HF_TOKEN")
        envp = ("HF_XET_HIGH_PERFORMANCE=1 " + (f"HF_TOKEN={shlex.quote(tok)} " if tok else ""))   # hub 1.19 downloads via xet; HF_HUB_ENABLE_HF_TRANSFER is deprecated
        def dl(item):
            inst, blk = item
            args = f"--coord --draft {a.draft_repo}" if blk is None else f"--layers {' '.join(map(str, blk))}"
            rc, out, err = rssh(inst, f"cd /root && {envp}{PY} node_fetch.py {args} 2>&1 | tail -6", 4 * 3600)
            return inst["id"], "NODE_FETCH_DONE" in out, out.strip().replace("\n", " | ")[-300:]
        t0 = time.time()
        for nid, ok, tail in pmap(dl, list(chain) + [(coord, None)]):
            log(f"    {nid}: {'OK' if ok else 'FAIL'} {tail}")
            if not ok: raise RuntimeError(f"fetch failed on {nid}")
        manifest["fetch_s"] = round(time.time() - t0)

    ring = MODES[a.mode]["cls"] == "ring"
    log(f"[7] stages ({'ring direct-return' if ring else 'relay-back chain'}; trace={a.trace})")
    ensure_stages(chain, coord, ring, a.trace, a.stage_timeout)

    head = ep(chain[0][0])
    results = []
    for i in range(a.runs):
        dump = f"/root/run_{a.mode}_{i}.json" if MODES[a.mode]["dump"] else None
        trace = f"/root/coord_trace_{a.mode}_{i}.jsonl" if a.trace else None
        cmd = coord_cmd(a.mode, head, a.prompt, a.max_new, K, a.depth, dump)
        envp = f"COORD_TRACE={trace} " if trace else ""
        full = (REAP + f"rm -f {dump or ''} {trace or ''}; cd /root && {envp}{PY} {cmd} 2>&1")
        log(f"[8] run {i}: {cmd}")
        t0 = time.time()
        rc, out, err = rssh(coord, full, 3600)
        dt = time.time() - t0
        open(os.path.join(a.out, f"{a.mode}_run{i}.stdout"), "w").write(out + err)
        r = parse_result(out) | {"run": i, "mode": a.mode, "rc": rc, "wall_s": round(dt), "t_start": t0, "t_end": t0 + dt, "cmd": cmd}
        if dump or trace:
            scp_from(coord, [p for p in (dump, trace) if p], a.out)
        if dump and os.path.exists(os.path.join(a.out, os.path.basename(dump))):
            d = json.load(open(os.path.join(a.out, os.path.basename(dump))))
            sha = hashlib.sha256(json.dumps(d["output_token_ids"]).encode()).hexdigest()
            r["output_sha256"] = sha; r["matches_receipt"] = sha == RECEIPT_SHA; r["n_ids"] = len(d["output_token_ids"])
        results.append(r)
        log(f"    -> {r.get('summary', 'NO SUMMARY LINE (see stdout file)')}" + (f" | sha {r['output_sha256'][:12]} receipt_match={r['matches_receipt']}" if "output_sha256" in r else ""))
        if "time_split" in r: log(f"       {r['time_split']}")
        json.dump(manifest | {"results": results}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)

    log("[9] pull stage logs" + (" + traces" if a.trace else ""))
    for i, (inst, blk) in enumerate(chain):
        scp_from(inst, ["/root/stage.log", "/root/profile.json"] + (["/root/stage_trace.jsonl"] if a.trace else []), os.path.join(a.out, f"stage{i}_{inst['id']}"))
    scp_from(coord, ["/root/profile.json"], os.path.join(a.out, f"coord_{coord['id']}"))
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(manifest | {"results": results}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    if results:
        ts = [r["tok_s"] for r in results if "tok_s" in r]
        log(f"[done] {a.mode}: tok/s per run {ts}" + (f", median {sorted(ts)[len(ts)//2]:.2f}" if ts else "") + f" -> {a.out}/manifest.json")
    log("stages stay up. Teardown is manual: vastai destroy instance <id> for every erfan-glm-* id, after the artifacts are copied.")

if __name__ == "__main__":
    main()
