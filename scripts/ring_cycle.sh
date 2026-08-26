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

SPARE=${SPARE:-1}; log "rent $((N+SPARE)) boxes ($N needed, $SPARE spare against slow image pulls)"; bash "$HERE/ring_up.sh" "$((N+SPARE))" 2>&1 | tee "$RUNS/ring_up.log" | sed 's/^/    /'
[ "$(wc -l < "$RING_IDS" | tr -d ' ')" -ge "$N" ] || fail "rented fewer than $N"
log "RENTED $(awk '{print $1}' "$RING_IDS" | paste -sd, -)"

wait_boxes() {  # $1 = cap seconds; start a waiter for every id in RING_IDS that is not READY yet, return when N are READY or all waiters ended
  for id in $(awk '{print $1}' "$RING_IDS"); do grep -q READY "$RUNS/wait_$id.txt" 2>/dev/null && continue
    ( bash "$IMG/../vast/wait_ssh.sh" "$id" "$1" > "$RUNS/wait_$id.txt" 2>"$RUNS/wait_$id.log" ) & done
  while :; do n=$(cat "$RUNS"/wait_*.txt 2>/dev/null | grep -c READY); [ "$n" -ge "$N" ] && break; [ "$(jobs -r | wc -l | tr -d ' ')" -eq 0 ] && break; sleep 10; done
}
log "wait for ssh (cap 15 min, parallel); keep the first $N that come up; boxes that never boot are replaced from other hosts"
wait_boxes 900
for round in 1 2; do
  R=$(cat "$RUNS"/wait_*.txt 2>/dev/null | grep -c READY); [ "$R" -ge "$N" ] && break
  BAD=""; for id in $(awk '{print $1}' "$RING_IDS"); do grep -q READY "$RUNS/wait_$id.txt" 2>/dev/null || BAD="$BAD $id"; done
  for id in $BAD; do h=$(awk -v i="$id" '$1==i{print $5}' "$RING_IDS"); EXCLUDE_HOSTS="${EXCLUDE_HOSTS:-}${EXCLUDE_HOSTS:+,}$h"; rm -f "$RUNS/wait_$id.txt"; log "release unbooted box $id (host $h)"; bash "$HERE/ring_down.sh" "$id" | sed 's/^/    /'; done
  grep -v -E "^($(echo $BAD | tr ' ' '|')) " "$RING_IDS" > "$RING_IDS.tmp" && mv "$RING_IDS.tmp" "$RING_IDS"
  need=$(( N - R )); log "top-up round $round: $R ready, renting $need more (excluding hosts $EXCLUDE_HOSTS)"
  APPEND=1 EXCLUDE_HOSTS="$EXCLUDE_HOSTS" bash "$HERE/ring_up.sh" "$need" 2>&1 | tee -a "$RUNS/ring_up.log" | sed 's/^/    /'
  wait_boxes 900
done
wait 2>/dev/null
KEEP=$(grep -l READY "$RUNS"/wait_*.txt | head -n "$N" | sed 's/.*wait_\([0-9]*\)\.txt/\1/' | paste -sd, -)
READY=$(echo "$KEEP" | tr ',' '\n' | grep -c .); log "READY $READY/$N kept: $KEEP"; cat "$RUNS"/wait_*.txt | sed 's/^/    /'
[ "$READY" -ge "$N" ] || fail "only $READY boxes reachable after top-ups"
for id in $(awk '{print $1}' "$RING_IDS"); do case ",$KEEP," in *,$id,*) ;; *) log "release extra box $id"; bash "$HERE/ring_down.sh" "$id" | sed 's/^/    /';; esac; done
grep -E "^($(echo "$KEEP" | tr ',' '|')) " "$RING_IDS" > "$RING_IDS.kept"; RING_IDS="$RING_IDS.kept"; IDS=$KEEP
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
