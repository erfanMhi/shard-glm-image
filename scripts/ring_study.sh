#!/usr/bin/env bash
# ring_study.sh: the 100-prompt study. rent 7 (+spare) -> wait/top-up -> profile -> mesh + auto coordinator -> fetch -> sanity (receipt
# sha) -> multi-prompt validation -> depth sweep (K=2: 6,8,10,12,14) -> eager reference (correctness per prompt) -> K sweep (D=8: 3,4)
# -> jitter (20 prompts x 3 reps at D=6 and D=12) -> traced calibration -> rebalanced plan (D=6, D=12) -> traced D=12 -> stats -> teardown.
# Failures leave the ring up (FAIL_TEARDOWN=1 to change); the hard cap always tears down.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); IMG=$(dirname "$HERE"); V=$HOME/.vastcli/bin/vastai
STAMP=$(date +%Y%m%d-%H%M); RUNS=${RUNS:-$IMG/runs/study-$STAMP}; mkdir -p "$RUNS"; export RING_IDS=$RUNS/ring_ids.txt
CAP_S=${CAP_S:-14400}; T0=$(date +%s); N=${N:-7}; SPARE=${SPARE:-1}; MAXNEW=${MAXNEW:-96}
PROMPTS=${PROMPTS:-$IMG/prompts/study100.jsonl}
log() { echo "[$(date '+%H:%M:%S') +$(( ($(date +%s)-T0)/60 ))m] $*"; }
command -v caffeinate >/dev/null && { caffeinate -is -w $$ & }   # no idle/system sleep while the study runs (a laptop lid must still stay open)
# deadman: an independent process that destroys every erfan-glm-* box DEADMAN_S after start, even if this script exited on a failure
# with the ring left up, is stuck inside one launch_ring step (budget() only runs between steps), or was killed. Cancelled by teardown.
DEADMAN_S=${DEADMAN_S:-$((CAP_S + 1800))}
( while [ $(( $(date +%s) - T0 )) -lt "$DEADMAN_S" ]; do sleep 60; done
  echo "[$(date '+%H:%M:%S')] deadman: ${DEADMAN_S}s since start, destroying every erfan-glm-* box"; bash "$HERE/ring_down.sh" ) >> "$RUNS/deadman.log" 2>&1 &
DEADMAN=$!; log "deadman pid $DEADMAN fires at +$((DEADMAN_S/60)) min (log: $RUNS/deadman.log)"
teardown() { bash "$HERE/ring_down.sh" 2>&1 | sed 's/^/    /'; log "DESTROYED"; kill "$DEADMAN" 2>/dev/null; }
stats() { log "STATS"; ls "$RUNS"/*/results_*.jsonl >/dev/null 2>&1 && python3 "$HERE/study_stats.py" $(ls "$RUNS"/*/results_*.jsonl) --baseline d6_k2 --out "$RUNS/REPORT.md" --csv "$RUNS/summary.csv" 2>&1 | head -60
  for t in trace_d6 plan_trace_d12; do [ -d "$RUNS/$t" ] && python3 "$HERE/analyze_trace.py" "$RUNS/$t" > "$RUNS/$t/ANALYSIS.txt" 2>&1; done; return 0; }
fail() { log "FAILED $*"; if [ -n "${FAIL_TEARDOWN:-}" ]; then teardown; else log "ring LEFT UP for inspection (deadman still destroys it at +$((DEADMAN_S/60)) min); run scripts/ring_down.sh when done"; fi; exit 1; }
trap 'fail signal' INT TERM
budget() { [ $(( $(date +%s) - T0 )) -ge $CAP_S ] && { log "FAILED hard cap ${CAP_S}s"; teardown; stats; exit 1; }; }   # bill stops first, then the report
LR() { budget; python3 "$HERE/launch_ring.py" --ids "$IDS" --coord-id "$COORD" --max-new "$MAXNEW" "$@" 2>&1 | tee -a "$RUNS/launch.log" | grep -E "^\S+ (\[|    ->|    !!|    stage|    loop|    coord|    pushed|    explicit|    [0-9]+: )" ; return ${PIPESTATUS[0]}; }
FAILS=0   # consecutive STUDY steps whose launch_ring raised (box gone, stage did not warm, push failed): two in a row = the ring is broken
STUDY() { # STUDY <tag> <mode> <depth> <K> <prompts> <repeat> [extra launch_ring args]
  local tag=$1 mode=$2 d=$3 k=$4 pf=$5 rep=$6; shift 6; mkdir -p "$RUNS/$tag"; cp "$RUNS/mesh/mesh_rtt.json" "$RUNS/$tag/" 2>/dev/null
  log "STUDY $tag: $mode depth $d K $k prompts $(basename $pf) x$rep $*"
  if LR --mode "$mode" --runs 1 --depth "$d" --K "$k" --prompts-file "$pf" --repeat "$rep" --order "$ORDER" --wait-timeout 180 --out "$RUNS/$tag" "$@"; then FAILS=0; return 0; fi
  FAILS=$((FAILS+1)); [ "$FAILS" -ge 2 ] && fail "two consecutive study steps failed ($tag); the ring is probably broken (a box offline?)"; return 1
}
# prompt subsets
python3 - "$PROMPTS" "$RUNS" <<'PY'
import json, sys, collections
P = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]; R = sys.argv[2]
bycat = collections.defaultdict(list)
for p in P: bycat[p["cat"]].append(p)
sub20 = [p for c in sorted(bycat) for p in bycat[c][:5]]
# validate set: the receipt prompt first (clean ring, as in the single-prompt runs) AND again last (after 8 other prompts, incl. ones that
# hit EOS early) under a different id; both must reproduce the receipt sha, which tests the per-prompt reset of draft cache + stages
sub10 = [P[0]] + [p for c in sorted(bycat) for p in bycat[c][1:3] if p["id"] != P[0]["id"]][:8] + [dict(P[0], id=P[0]["id"] + "-again")]
open(f"{R}/prompts20.jsonl", "w").write("".join(json.dumps(p) + "\n" for p in sub20))
open(f"{R}/prompts10.jsonl", "w").write("".join(json.dumps(p) + "\n" for p in sub10))
print(f"prompts: {len(P)} total, subsets 20 and {len(sub10)}; categories {dict((c, len(v)) for c, v in bycat.items())}")
PY

log "rent $((N+SPARE)) boxes"; bash "$HERE/ring_up.sh" "$((N+SPARE))" 2>&1 | tee "$RUNS/ring_up.log" | sed 's/^/    /'
[ "$(wc -l < "$RING_IDS" | tr -d ' ')" -ge "$N" ] || fail "rented fewer than $N"
log "RENTED $(awk '{print $1}' "$RING_IDS" | paste -sd, -)"
wait_boxes() { for id in $(awk '{print $1}' "$RING_IDS"); do grep -q READY "$RUNS/wait_$id.txt" 2>/dev/null && continue
    ( bash "$IMG/../vast/wait_ssh.sh" "$id" "$1" > "$RUNS/wait_$id.txt" 2>"$RUNS/wait_$id.log" ) & done
  while :; do n=$(cat "$RUNS"/wait_*.txt 2>/dev/null | grep -c READY); [ "$n" -ge "$N" ] && break; [ "$(jobs -r | wc -l | tr -d ' ')" -eq 0 ] && break; sleep 10; done; }
log "wait for ssh (cap 15 min); keep the first $N; replace boxes that never boot"
wait_boxes 900
for round in 1 2; do
  R=$(cat "$RUNS"/wait_*.txt 2>/dev/null | grep -c READY); [ "$R" -ge "$N" ] && break
  BAD=""; for id in $(awk '{print $1}' "$RING_IDS"); do grep -q READY "$RUNS/wait_$id.txt" 2>/dev/null || BAD="$BAD $id"; done
  for id in $BAD; do h=$(awk -v i="$id" '$1==i{print $5}' "$RING_IDS"); EXCLUDE_HOSTS="${EXCLUDE_HOSTS:-}${EXCLUDE_HOSTS:+,}$h"; rm -f "$RUNS/wait_$id.txt"; log "release unbooted box $id (host $h)"; bash "$HERE/ring_down.sh" "$id" | sed 's/^/    /'; done
  grep -v -E "^($(echo $BAD | tr ' ' '|')) " "$RING_IDS" > "$RING_IDS.tmp" && mv "$RING_IDS.tmp" "$RING_IDS"
  need=$(( N - R )); log "top-up round $round: $R ready, renting $need more"; APPEND=1 EXCLUDE_HOSTS="$EXCLUDE_HOSTS" bash "$HERE/ring_up.sh" "$need" 2>&1 | tee -a "$RUNS/ring_up.log" | sed 's/^/    /'; wait_boxes 900
done
kill $(jobs -p) 2>/dev/null; wait 2>/dev/null
KEEP=$(grep -l READY "$RUNS"/wait_*.txt | head -n "$N" | sed 's/.*wait_\([0-9]*\)\.txt/\1/' | paste -sd, -)
READY=$(echo "$KEEP" | tr ',' '\n' | grep -c .); log "READY $READY/$N kept: $KEEP"; cat "$RUNS"/wait_*.txt | sed 's/^/    /'
[ "$READY" -ge "$N" ] || fail "only $READY boxes reachable after top-ups"
for id in $(awk '{print $1}' "$RING_IDS"); do case ",$KEEP," in *,$id,*) ;; *) log "release extra box $id"; bash "$HERE/ring_down.sh" "$id" | sed 's/^/    /';; esac; done
grep -E "^($(echo "$KEEP" | tr ',' '|')) " "$RING_IDS" > "$RING_IDS.kept"; RING_IDS="$RING_IDS.kept"; IDS=$KEEP; COORD=$(awk 'NR==1{print $1}' "$RING_IDS")

log "profile every box (parallel)"
while read -r id st off pr host; do R=$(grep -h READY "$RUNS/wait_$id.txt"); set -- $R; ip=$2; p22=$3
  ( ssh -i ~/.ssh/id_ed25519 -p "$p22" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes root@"$ip" 'bash /root/profile_box.sh /root/profile.json >/dev/null 2>&1; echo profiled' > "$RUNS/profile_$id.txt" 2>&1
    scp -i ~/.ssh/id_ed25519 -P "$p22" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -q root@"$ip":/root/profile.json "$RUNS/profile_$id.json" 2>/dev/null ) &
done < "$RING_IDS"; wait; log "PROFILED"

log "mesh + auto coordinator"; LR --auto-coord --mesh-only --out "$RUNS/mesh" || fail "mesh"
COORD=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["coord_id"])' "$RUNS/mesh/manifest.json")
ORDER=$(python3 -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["order_ids"])))' "$RUNS/mesh/manifest.json")
log "MESH coord=$COORD order=$ORDER"

log "SANITY: receipt prompt, cg K=2 D=6 x2 (fetch happens here)"; mkdir -p "$RUNS/sanity"; cp "$RUNS/mesh/mesh_rtt.json" "$RUNS/sanity/"
LR --mode cg --runs 2 --order "$ORDER" --prompt "def quicksort(arr):" --out "$RUNS/sanity" || fail "sanity run"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); ok=[r.get("matches_receipt") for r in d["results"]]; print("receipt_match:", ok); sys.exit(0 if all(ok) else 1)' "$RUNS/sanity/manifest.json" || fail "output does not match the receipt"

log "VALIDATE multi-prompt path on 10 prompts (code-000 first AND code-000-again last must both reproduce the receipt sha)"
STUDY validate cgmulti 6 2 "$RUNS/prompts10.jsonl" 1 --skip-fetch || fail "validate"
python3 - "$RUNS/validate" <<'PY' || fail "multi-prompt path does not reproduce the receipt on code-000 / code-000-again"
import json, glob, sys
fs = glob.glob(sys.argv[1] + "/results_*.jsonl")
rows = [json.loads(l) for l in open(fs[0])] if fs else []
r0 = {r["id"]: r for r in rows if r.get("id") in ("code-000", "code-000-again")}
for k, r in sorted(r0.items()): print(f"{k}: {r.get('tok_s')} tok/s, ntok {r.get('ntok')}, sha {str(r.get('output_sha', ''))[:12]}, error {r.get('error')}")
print("n =", len(rows), "ok =", sum(1 for r in rows if "tok_s" in r), "errors =", sum(1 for r in rows if "error" in r))
sys.exit(0 if len(r0) == 2 and all(str(r.get("output_sha", "")).startswith("d9e61275084cb2bf") for r in r0.values()) else 1)
PY

for d in 6 8 10 12 14; do STUDY "d${d}_k2" cgmulti $d 2 "$PROMPTS" 1 --skip-fetch || log "d${d}_k2 failed (continuing)"; done
STUDY d6_k2_eager cgmulti_eager 6 2 "$PROMPTS" 1 --skip-fetch || log "eager reference failed (continuing)"
for k in 3 4; do STUDY "d8_k${k}" cgmulti 8 $k "$PROMPTS" 1 --skip-fetch || log "d8_k${k} failed (continuing)"; done
STUDY jitter_d6 cgmulti 6 2 "$RUNS/prompts20.jsonl" 3 --skip-fetch || log "jitter d6 failed"
STUDY jitter_d12 cgmulti 12 2 "$RUNS/prompts20.jsonl" 3 --skip-fetch || log "jitter d12 failed"
STUDY trace_d6 cgmulti 6 2 "$RUNS/prompts10.jsonl" 1 --skip-fetch --trace || log "trace d6 failed"
PLAN=$(python3 "$HERE/plan_from_traces.py" "$RUNS/trace_d6" 2> "$RUNS/plan.txt"); cat "$RUNS/plan.txt" | sed 's/^/    /'; log "PLAN $PLAN"
if [ -n "$PLAN" ]; then
  STUDY plan_d6_k2 cgmulti 6 2 "$PROMPTS" 1 --plan "$PLAN" || log "plan d6 failed (continuing)"          # fetch step runs: extra shards for the new blocks
  STUDY plan_d12_k2 cgmulti 12 2 "$PROMPTS" 1 --plan "$PLAN" --skip-fetch || log "plan d12 failed (continuing)"
  STUDY plan_trace_d12 cgmulti 12 2 "$RUNS/prompts10.jsonl" 1 --plan "$PLAN" --skip-fetch --trace || log "plan trace failed"
fi
log "STUDY_DONE -> $RUNS"; trap - INT TERM; teardown; stats; exit 0   # bill stops first, then the report
