#!/usr/bin/env bash
# profile_box.sh [out.json]   one-shot profile of this box -> /root/profile.json (GPU, memory bandwidth,
# torch.save/load cost of one [1,3,6144] bf16 frame, disk, TCP sysctls, MTU, CPU, versions, public IP).
# Runs on the box, inside the image. Needs ~8 GB free GPU memory and 4 GB free disk for a moment.
set -uo pipefail
OUT=${1:-/root/profile.json}
PY=/root/vmoe/bin/python
TMP=$(mktemp -d)
trap 'rm -rf "$TMP" /root/.ddtest' EXIT

echo "[profile] gpu"
nvidia-smi --query-gpu=name,uuid,power.limit,power.max_limit,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max,memory.total,driver_version \
  --format=csv,noheader > "$TMP/gpu.csv" 2>"$TMP/gpu.err" || echo "nvidia-smi failed: $(cat "$TMP/gpu.err")"

echo "[profile] torch microbench (4 GB copy x10, 200x save/load of [1,3,6144] bf16)"
"$PY" - > "$TMP/torch.json" 2>"$TMP/torch.err" <<'PY'
import io, json, time, statistics, sys
import torch
r = {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda}
try:
    import vllm; r["vllm"] = vllm.__version__
except Exception as e: r["vllm"] = f"import failed: {e}"
try:
    import transformers; r["transformers"] = transformers.__version__
except Exception as e: r["transformers"] = f"import failed: {e}"
try:
    import huggingface_hub; r["huggingface_hub"] = huggingface_hub.__version__
except Exception: pass
if not torch.cuda.is_available():
    r["error"] = "cuda not available"; print(json.dumps(r)); sys.exit(0)
dev = "cuda"
r["gpu_name"] = torch.cuda.get_device_name(0); r["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
free, total = torch.cuda.mem_get_info(); r["gpu_mem_free_GB"] = round(free / 1e9, 1); r["gpu_mem_total_GB"] = round(total / 1e9, 1)
# --- device memory bandwidth: 4 GB tensor copied 10x (bytes moved = read + write)
n = (4 * 1024**3) // 2
a = torch.empty(n, dtype=torch.bfloat16, device=dev); b = torch.empty_like(a)
b.copy_(a); torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(10): b.copy_(a)
torch.cuda.synchronize(); dt = time.perf_counter() - t
r["copy_4GB_x10_s"] = round(dt, 4); r["mem_bw_GBps_read_plus_write"] = round(10 * 2 * 4 * 1024**3 / dt / 1e9, 1)
del a, b; torch.cuda.empty_cache()
# --- pickle cost of one stage message (what send_msg/recv_msg do): GPU->cpu, torch.save, torch.load, ->GPU
x = torch.randn(1, 3, 6144, dtype=torch.bfloat16, device=dev)
save_ms, load_ms, rt_ms = [], [], []
for _ in range(200):
    t0 = time.perf_counter(); bio = io.BytesIO(); torch.save((7, x.cpu()), bio); frame = bio.getvalue(); t1 = time.perf_counter()
    sp, y = torch.load(io.BytesIO(frame), weights_only=False); y = y.to(dev); torch.cuda.synchronize(); t2 = time.perf_counter()
    save_ms.append((t1 - t0) * 1e3); load_ms.append((t2 - t1) * 1e3); rt_ms.append((t2 - t0) * 1e3)
q = lambda v: {"mean": round(statistics.mean(v), 3), "p50": round(statistics.median(v), 3), "p90": round(sorted(v)[int(0.9 * len(v))], 3), "min": round(min(v), 3)}
r["frame_bytes_1x3x6144_bf16"] = len(frame)
r["pickle_save_ms"] = q(save_ms); r["pickle_load_ms"] = q(load_ms); r["pickle_roundtrip_ms"] = q(rt_ms)
print(json.dumps(r))
PY
[ -s "$TMP/torch.json" ] || echo "torch microbench failed: $(tail -3 "$TMP/torch.err")"

echo "[profile] disk (4 GB write then read, O_DIRECT)"
dd if=/dev/zero of=/root/.ddtest bs=1M count=4096 oflag=direct 2> "$TMP/ddw.txt" || true
dd if=/root/.ddtest of=/dev/null bs=16M iflag=direct 2> "$TMP/ddr.txt" || true
rm -f /root/.ddtest
df -h /root | tail -1 > "$TMP/df.txt"

echo "[profile] network + cpu"
for k in net.ipv4.tcp_slow_start_after_idle net.ipv4.tcp_congestion_control net.ipv4.tcp_rmem net.ipv4.tcp_wmem net.core.rmem_max net.core.wmem_max net.ipv4.tcp_window_scaling; do
  printf '%s=%s\n' "$k" "$(sysctl -n "$k" 2>/dev/null | tr '\t' ' ')"
done > "$TMP/sysctl.txt"
DEFIF=$(ip route show default 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -1)
{ for d in /sys/class/net/*; do printf '%s=%s\n' "$(basename "$d")" "$(cat "$d/mtu" 2>/dev/null)"; done; } > "$TMP/mtu.txt"
nproc > "$TMP/nproc.txt"
grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //' > "$TMP/cpu.txt"
curl -s --max-time 10 ifconfig.me > "$TMP/ip.txt" || true
env | grep -E '^(VAST_|PUBLIC_IPADDR|CONTAINER_ID)' | sort > "$TMP/vastenv.txt" || true

echo "[profile] assemble -> $OUT"
"$PY" - "$TMP" "$OUT" "$DEFIF" <<'PY'
import json, re, sys, time, socket
tmp, out, defif = sys.argv[1], sys.argv[2], sys.argv[3]
def rd(n):
    try: return open(f"{tmp}/{n}").read().strip()
    except Exception: return ""
p = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hostname": socket.gethostname()}
keys = ["name", "uuid", "power_limit_W", "power_max_limit_W", "clocks_max_sm_MHz", "clocks_max_mem_MHz", "pcie_gen_max", "pcie_width_max", "memory_total_MiB", "driver_version"]
g = rd("gpu.csv").splitlines()
p["gpu"] = [dict(zip(keys, [c.strip() for c in ln.split(",")])) for ln in g if ln.strip()]
try: p["torch"] = json.loads(rd("torch.json"))
except Exception: p["torch"] = {"error": rd("torch.err")[-400:]}
def dd(n):
    m = re.search(r"(\d+) bytes .*copied, ([\d.]+) s", rd(n))
    return round(int(m.group(1)) / float(m.group(2)) / 1e6, 1) if m else None
p["disk"] = {"write_MBps_direct": dd("ddw.txt"), "read_MBps_direct": dd("ddr.txt"), "df_root": rd("df.txt")}
p["sysctl"] = dict(ln.split("=", 1) for ln in rd("sysctl.txt").splitlines() if "=" in ln)
p["mtu"] = dict(ln.split("=", 1) for ln in rd("mtu.txt").splitlines() if "=" in ln)
p["default_iface"] = defif
p["cpu"] = {"nproc": int(rd("nproc.txt") or 0), "model": rd("cpu.txt")}
p["public_ip"] = rd("ip.txt")
p["vast_env"] = dict(ln.split("=", 1) for ln in rd("vastenv.txt").splitlines() if "=" in ln)
json.dump(p, open(out, "w"), indent=1)
t = p["torch"]
print(f"gpu {p['gpu'][0]['name'] if p['gpu'] else '?'} | mem bw {t.get('mem_bw_GBps_read_plus_write')} GB/s | pickle rt p50 {t.get('pickle_roundtrip_ms', {}).get('p50')} ms "
      f"({t.get('frame_bytes_1x3x6144_bf16')} B) | disk read {p['disk']['read_MBps_direct']} MB/s | cc {p['sysctl'].get('net.ipv4.tcp_congestion_control')} "
      f"ssai {p['sysctl'].get('net.ipv4.tcp_slow_start_after_idle')} | mtu {p['mtu'].get(defif)} | ip {p['public_ip']}")
PY
