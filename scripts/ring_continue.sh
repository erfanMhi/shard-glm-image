#!/usr/bin/env bash
# ring_continue.sh <runs_dir> <ids,comma> <coord_id>: Phase B from the mesh step on, for boxes that are already rented and
# fetched (used after a box was replaced by hand). Same ladder/sweep as ring_cycle.sh; tears everything down at the end
# or on failure. Set NO_TEARDOWN=1 to keep the ring up.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); RUNS=$1; IDS=$2; COORD=$3
CAP_S=${CAP_S:-9000}; T0=$(date +%s); RUNS_PER=${RUNS_PER:-3}; PROMPT=${PROMPT:-"def quicksort(arr):"}; MAXNEW=${MAXNEW:-96}
log() { echo "[$(date '+%H:%M:%S') +$(( ($(date +%s)-T0)/60 ))m] $*"; }
teardown() { [ -n "${NO_TEARDOWN:-}" ] && { log "NO_TEARDOWN set; ring left up"; return; }; bash "$HERE/ring_down.sh" 2>&1 | sed 's/^/    /'; log "DESTROYED"; }
fail() { log "FAILED $*"; if [ -n "${FAIL_TEARDOWN:-}" ]; then teardown; else log "ring LEFT UP for inspection (FAIL_TEARDOWN=1 to change); run scripts/ring_down.sh when done"; fi; exit 1; }
trap 'fail signal' INT TERM
budget() { [ $(( $(date +%s) - T0 )) -ge $CAP_S ] && { log "FAILED hard cap ${CAP_S}s"; teardown; exit 1; }; }
LR() { budget; python3 "$HERE/launch_ring.py" --ids "$IDS" --coord-id "$COORD" --prompt "$PROMPT" --max-new "$MAXNEW" "$@" 2>&1 | tee -a "$RUNS/launch2.log" | grep -E "^\S+ (\[|    ->|       time|    stage|    loop|    coord|    [0-9]+: )" ; return ${PIPESTATUS[0]}; }

log "mesh (remeasured with the replacement box; coordinator fixed to $COORD)"
mkdir -p "$RUNS/mesh2"; LR --mesh-only --remesh --out "$RUNS/mesh2" || fail "mesh"
ORDER=${ORDER_OVERRIDE:-$(python3 -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["order_ids"])))' "$RUNS/mesh2/manifest.json")}   # ORDER_OVERRIDE keeps the blocks the boxes already hold
log "MESH coord=$COORD order=$ORDER"
for MODE in cg pipe direct6 relay6 plain; do
  log "RUN $MODE x$RUNS_PER"; mkdir -p "$RUNS/$MODE"; cp "$RUNS/mesh2/mesh_rtt.json" "$RUNS/$MODE/"
  LR --mode "$MODE" --runs "$RUNS_PER" --order "$ORDER" $([ "$MODE" != cg ] && echo --skip-fetch) --out "$RUNS/$MODE" || fail "mode $MODE"
done
log "RUN cgeager x1 + cg --trace x1"
for M in cgeager cgtrace; do mkdir -p "$RUNS/$M"; cp "$RUNS/mesh2/mesh_rtt.json" "$RUNS/$M/"; done
LR --mode cgeager --runs 1 --order "$ORDER" --skip-fetch --out "$RUNS/cgeager" || log "cgeager failed (continuing)"
LR --mode cg --runs 1 --order "$ORDER" --skip-fetch --trace --out "$RUNS/cgtrace" || log "cg --trace failed (continuing)"
log "SWEEP depth x K (cg, 1 run each)"
for cfg in "8 2" "10 2" "12 2" "14 2" "10 3" "10 4" "8 4"; do set -- $cfg; d=$1; k=$2
  mkdir -p "$RUNS/sweep_d${d}_k${k}"; cp "$RUNS/mesh2/mesh_rtt.json" "$RUNS/sweep_d${d}_k${k}/"
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
