#!/usr/bin/env bash
# ring_cycle.sh: the whole Phase B, detached. rent 7 -> wait ssh -> profile -> mesh + auto coordinator -> fetch ->
# ladder (plain, relay6 | direct6, pipe, cg x3 each) -> cgeager x1 -> cg --trace x1 -> sweep -> pull -> destroy all.
# Destroys every erfan-glm-* box on any failure and at the hard cap. Markers: RENTED, READY, PROFILED, MESH, RUN <mode>, SWEEP, PULLED, DESTROYED, FAILED.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); IMG=$(dirname "$HERE"); V=$HOME/.vastcli/bin/vastai
STAMP=$(date +%Y%m%d-%H%M); RUNS=${RUNS:-$IMG/runs/ring-$STAMP}; mkdir -p "$RUNS"; export RING_IDS=$RUNS/ring_ids.txt
CAP_S=${CAP_S:-10800}; T0=$(date +%s); N=${N:-7}; RUNS_PER=${RUNS_PER:-3}; PROMPT=${PROMPT:-"def quicksort(arr):"}; MAXNEW=${MAXNEW:-96}
log() { echo "[$(date '+%H:%M:%S') +$(( ($(date +%s)-T0)/60 ))m] $*"; }
teardown() { bash "$HERE/ring_down.sh" 2>&1 | sed 's/^/    /'; log "DESTROYED"; }
fail() { log "FAILED $*"; teardown; exit 1; }
trap 'fail signal' INT TERM
budget() { [ $(( $(date +%s) - T0 )) -ge $CAP_S ] && fail "hard cap ${CAP_S}s"; }
LR() { budget; python3 "$HERE/launch_ring.py" --ids "$IDS" --coord-id "$COORD" --prompt "$PROMPT" --max-new "$MAXNEW" "$@" 2>&1 | tee -a "$RUNS/launch.log" | grep -E "^\S+ (\[|    ->|       time|    stage|    loop|    ->)" ; return ${PIPESTATUS[0]}; }

log "rent $N boxes"; bash "$HERE/ring_up.sh" "$N" 2>&1 | tee "$RUNS/ring_up.log" | sed 's/^/    /'
[ "$(wc -l < "$RING_IDS" | tr -d ' ')" -ge "$N" ] || fail "rented fewer than $N"
IDS=$(awk '{print $1}' "$RING_IDS" | paste -sd, -); log "RENTED $IDS"

log "wait for ssh on all (cap 25 min each, parallel)"
for id in $(awk '{print $1}' "$RING_IDS"); do ( bash "$IMG/../vast/wait_ssh.sh" "$id" 1500 > "$RUNS/wait_$id.txt" 2>"$RUNS/wait_$id.log" ) & done; wait
READY=$(cat "$RUNS"/wait_*.txt | grep -c READY); log "READY $READY/$N"; cat "$RUNS"/wait_*.txt | sed 's/^/    /'
[ "$READY" -ge "$N" ] || fail "only $READY boxes reachable"
COORD=$(awk 'NR==1{print $1}' "$RING_IDS")   # placeholder; --auto-coord picks the real one

log "profile every box (parallel)"
while read -r id st off pr; do R=$(grep -h READY "$RUNS/wait_$id.txt"); set -- $R; ip=$2; p22=$3
  ( ssh -i ~/.ssh/id_ed25519 -p "$p22" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes root@"$ip" 'bash /root/profile_box.sh /root/profile.json >/dev/null 2>&1; python3 - /root/profile.json' <<'PY' > "$RUNS/profile_$id.txt" 2>&1
import json, sys
p = json.load(open(sys.argv[1])); g = (p.get("gpu") or [{}])[0]; t = p.get("torch") or {}
print(g.get("name"), g.get("power_limit_W"), "W |", t.get("mem_bw_GBps_read_plus_write"), "GB/s |", (t.get("pickle_roundtrip_ms") or {}).get("p50"), "ms frame |",
      (p.get("disk") or {}).get("read_MBps_direct"), "MB/s disk |", (p.get("sysctl") or {}).get("net.ipv4.tcp_slow_start_after_idle", "?").strip(), "ssai |", p.get("public_ip"))
PY
    scp -i ~/.ssh/id_ed25519 -P "$p22" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -q root@"$ip":/root/profile.json "$RUNS/profile_$id.json" 2>/dev/null ) &
done < "$RING_IDS"; wait; for f in "$RUNS"/profile_*.txt; do echo "    $(basename $f .txt): $(tail -1 $f)"; done; log "PROFILED"

log "mesh + auto coordinator + fetch + ring up (cg class), no runs yet"
LR --auto-coord --mesh-only --out "$RUNS/mesh" || fail "mesh"
COORD=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["coord_id"])' "$RUNS/mesh/manifest.json")
ORDER=$(python3 -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["order_ids"])))' "$RUNS/mesh/manifest.json")
log "MESH coord=$COORD order=$ORDER"

# ladder: relay class first (plain, relay6), then ring class (direct6, pipe, cg). Fetch happens in the first call.
for MODE in plain relay6 direct6 pipe cg; do
  log "RUN $MODE x$RUNS_PER"; cp -n "$RUNS/mesh/mesh_rtt.json" "$RUNS/$MODE/" 2>/dev/null; mkdir -p "$RUNS/$MODE"; cp "$RUNS/mesh/mesh_rtt.json" "$RUNS/$MODE/"
  LR --mode "$MODE" --runs "$RUNS_PER" --order "$ORDER" $([ "$MODE" != plain ] && echo --skip-fetch) --out "$RUNS/$MODE" || fail "mode $MODE"
done
log "RUN cgeager x1 (receipt's lossless reference) + cg --trace x1 (instrumented)"
mkdir -p "$RUNS/cgeager" "$RUNS/cgtrace"; cp "$RUNS/mesh/mesh_rtt.json" "$RUNS/cgeager/"; cp "$RUNS/mesh/mesh_rtt.json" "$RUNS/cgtrace/"
LR --mode cgeager --runs 1 --order "$ORDER" --skip-fetch --out "$RUNS/cgeager" || log "cgeager failed (continuing)"
LR --mode cg --runs 1 --order "$ORDER" --skip-fetch --trace --out "$RUNS/cgtrace" || log "cg --trace failed (continuing)"
log "SWEEP depth x K (cg, 1 run each)"
for cfg in "8 2" "10 2" "12 2" "14 2" "10 3" "10 4" "8 4"; do set -- $cfg; d=$1; k=$2
  mkdir -p "$RUNS/sweep_d${d}_k${k}"; cp "$RUNS/mesh/mesh_rtt.json" "$RUNS/sweep_d${d}_k${k}/"
  LR --mode cg --runs 1 --order "$ORDER" --skip-fetch --depth "$d" --K "$k" --out "$RUNS/sweep_d${d}_k${k}" || log "sweep d=$d K=$k failed (continuing)"
done
python3 - "$RUNS" <<'PY'
import json, glob, os, sys
R = sys.argv[1]; rows = []
for m in sorted(glob.glob(os.path.join(R, "*", "manifest.json"))):
    d = json.load(open(m)); mode = os.path.basename(os.path.dirname(m))
    for r in d.get("results", []): rows.append((mode, r.get("run"), r.get("tok_s"), r.get("matches_receipt"), d.get("args", {}).get("depth"), d.get("K")))
print("mode run tok/s receipt_match depth K"); [print(*r) for r in rows]
open(os.path.join(R, "SUMMARY.txt"), "w").write("\n".join(" ".join(map(str, r)) for r in rows) + "\n")
PY
log "PULLED -> $RUNS"; trap - INT TERM; teardown; exit 0
