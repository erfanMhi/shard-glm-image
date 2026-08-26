#!/usr/bin/env bash
# smoke_test.sh   run ON one rented box (ssh -p <port> root@<ip> 'bash /root/smoke_test.sh 2>&1 | tee /root/smoke.log')
# Proves the image end to end on one GPU before renting seven: versions, profile, a one-layer weight fetch,
# draft tokenizer compatibility, the draft download + bench, the NVFP4 MoE kernel on real layer-6 weights,
# and a live one-layer stage probed over loopback with STAGE_TRACE on. About 25 GB of downloads.
#
# Env knobs: HF_TOKEN (recommended), DRAFT_CANDIDATES (space-separated HF ids), SKIP_DRAFT=1, SKIP_STAGE=1.
# Kill rule: GPU processes are reaped via nvidia-smi pid list + fuser on the port. Never `pkill -f glm_swarm`
# (it matches the ssh shell that launched it and kills the session).
set -uo pipefail
cd /root
PY=/root/vmoe/bin/python
export GLM_DIR=${GLM_DIR:-/root/glm52nvfp4}
DRAFT_CANDIDATES=${DRAFT_CANDIDATES:-"zai-org/GLM-4-9B-0414 THUDM/glm-4-9b-chat THUDM/glm-4-9b"}
PORT=29600
SUMMARY=/root/smoke_summary.txt; : > "$SUMMARY"
note() { echo "[smoke] $*"; echo "$*" >> "$SUMMARY"; }
step() { echo; echo "=================== $* ($(date -u +%H:%M:%S)) ==================="; }
tstart() { T0=$(date +%s); }
tend() { echo "$(( $(date +%s) - T0 ))s"; }
reap_gpu() {  # kill every GPU compute process + whatever holds the stage port; self-safe (no pattern match)
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
  fuser -k ${PORT}/tcp 2>/dev/null; sleep 2
}

step "1 versions"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PY" -c "import sys,torch,vllm,transformers,huggingface_hub; print('py',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,'vllm',vllm.__version__,'tf',transformers.__version__,'hub',huggingface_hub.__version__,'gpu',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" \
  && note "versions OK" || { note "versions FAIL"; exit 1; }
cmp -s /root/orig/glm_swarm_nvfp4_cg.py /opt/shard/research/glm_swarm_nvfp4_cg.py && note "orig cg.py identical to /opt/shard (sha $(sha256sum /root/orig/glm_swarm_nvfp4_cg.py | cut -c1-12))"

step "2 profile_box.sh"
tstart; bash /root/profile_box.sh /root/profile.json && note "profile OK ($(tend)) -> /root/profile.json" || note "profile FAIL"

step "3 node_fetch --tokenizer-only, then --layers 6 (timed)"
tstart; "$PY" node_fetch.py --tokenizer-only && note "meta fetch OK ($(tend))" || { note "meta fetch FAIL"; exit 1; }
tstart; "$PY" node_fetch.py --layers 6 | tee /root/fetch_layer6.log | grep -E "shards:|total:|NODE_FETCH_DONE|retry" ; note "layer-6 fetch $(tend): $(grep '^shards:' /root/fetch_layer6.log)"
ls -la "$GLM_DIR" | head -20

step "4 draft tokenizer compatibility"
DRAFT=""
for cand in $DRAFT_CANDIDATES; do
  echo "--- $cand"
  v=$("$PY" glm_draft_compat.py "$cand" 2>&1 | tee /root/compat_$(echo "$cand" | tr '/' '_').log | grep -E "^VERDICT|^vocab" | tr '\n' ' ')
  echo "$v"; note "compat $cand: $v"
  if [ -z "$DRAFT" ] && echo "$v" | grep -q COMPATIBLE; then DRAFT=$cand; fi
done
[ -n "$DRAFT" ] && note "draft pick: $DRAFT" || note "draft pick: NONE compatible (Phase B blocker)"

if [ -z "${SKIP_DRAFT:-}" ] && [ -n "$DRAFT" ]; then
  step "5 node_fetch --draft $DRAFT + glm_draft_bench.py"
  tstart; "$PY" node_fetch.py --draft "$DRAFT" | grep -E "draft:|total:|NODE_FETCH_DONE|retry"; note "draft fetch $(tend)"
  "$PY" glm_draft_bench.py --n 128 2>&1 | grep -E "draft loaded|decode:|out-of-vocab|MAXLEN" | tee /root/draft_bench_eager.log
  "$PY" glm_draft_bench.py --compile --n 128 2>&1 | grep -E "draft loaded|decode:|out-of-vocab|MAXLEN" | tee /root/draft_bench_compile.log
  note "draft bench: $(grep -h 'decode:' /root/draft_bench_eager.log /root/draft_bench_compile.log | tr '\n' ' ')"
fi

step "6 NVFP4 MoE kernel on real layer-6 weights (glm_nvfp4_moe.py) + fused_experts bench"
tstart; "$PY" glm_nvfp4_moe.py 2>&1 | grep -vE "^INFO|WARNING" | tail -5 | tee /root/nvfp4_moe.log; note "nvfp4 moe ($(tend)): $(grep VERDICT /root/nvfp4_moe.log | cut -c1-90)"
"$PY" /opt/shard/research/bench_fused_moe.py 2>&1 | grep -E "fused_experts|proj" | tee /root/bench_fused_moe.log; note "fused_experts: $(grep -h 'fused_experts' /root/bench_fused_moe.log | tr '\n' ' ')"

if [ -z "${SKIP_STAGE:-}" ]; then
  step "7 one-layer stage with STAGE_TRACE + loopback probe (20 round trips of [1,3,6144] bf16)"
  reap_gpu; rm -f /root/smoke_stage.log /root/smoke_stage_trace.jsonl
  STAGE_TRACE=/root/smoke_stage_trace.jsonl setsid "$PY" glm_swarm_nvfp4_kv.py stage --layers 6 --port $PORT > /root/smoke_stage.log 2>&1 < /dev/null &
  tstart
  for i in $(seq 1 120); do
    grep -q "WARM" /root/smoke_stage.log 2>/dev/null && break
    grep -qE "Traceback|exit status" /root/smoke_stage.log 2>/dev/null && { tail -20 /root/smoke_stage.log; note "stage FAIL (traceback)"; break; }
    sleep 5
  done
  if grep -q WARM /root/smoke_stage.log; then
    note "stage WARM in $(tend): $(grep -E 'loaded|moe_backend' /root/smoke_stage.log | tr '\n' ' ')"
    "$PY" - <<'PY' 2>&1 | grep -vE "^INFO|WARNING" | tee /root/loopback_probe.log
import socket, time, statistics, sys, torch
sys.path.insert(0, "/root")
from glm_swarm_nvfp4_kv import send_msg, recv_msg, H
s = socket.create_connection(("127.0.0.1", 29600), timeout=120); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
x = torch.randn(1, 3, H, dtype=torch.bfloat16, device="cuda") * 0.1
for _ in range(3): send_msg(s, 0, x); recv_msg(s)                     # warm (sp=0 resets the cache each time)
ts = []
for _ in range(20):
    t = time.perf_counter(); send_msg(s, 0, x); sp, y = recv_msg(s); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
print(f"LOOPBACK 20 round trips [1,3,{H}] bf16 through a 1-layer stage: mean {statistics.mean(ts):.2f} ms  p50 {statistics.median(ts):.2f}  min {min(ts):.2f}  max {max(ts):.2f}  | reply {tuple(y.shape)} finite={torch.isfinite(y).all().item()}")
s.close()
PY
    note "loopback: $(grep LOOPBACK /root/loopback_probe.log | cut -c1-140)"
    sleep 1; note "stage trace lines: $(wc -l < /root/smoke_stage_trace.jsonl 2>/dev/null || echo 0)"; tail -2 /root/smoke_stage_trace.jsonl 2>/dev/null
  fi
  reap_gpu; note "stage reaped (nvidia-smi pid list + fuser :$PORT)"
fi

step "SUMMARY (/root/smoke_summary.txt)"
cat "$SUMMARY"
echo "SMOKE_DONE"
