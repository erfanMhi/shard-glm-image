# NOTES (Phase A, 2026-08-26, prepared offline; nothing built or rented yet)

## Base image and wheels

- `vastai/base-image:cuda-13.2.1-auto` exists on Docker Hub (checked via the v2 tags API on 2026-08-26): amd64 + arm64, last pushed 2026-08-26, 7.4 GB compressed. Same tag `phase0/launch_swarm.py` used, so hosts have its layers cached. Neighbors if it ever disappears: `cuda-13.3.1-auto`, `cuda-13.1.2-auto`, `cuda-13.0.3-auto`, and the explicit `cuda-13.2.1-cudnn-devel-ubuntu24.04[-py3xx]` family.
- Python: the base is Ubuntu 24.04, whose system `python3` is 3.12 (vast's README: "tags without a Python suffix use the Ubuntu default"). In a Docker `RUN` the shell is non-login, so `python3 -m venv /root/vmoe` uses `/usr/bin/python3` (3.12), not the conda `/venv/main`. The Dockerfile prints `sys.version` at the end of the pip layer so the build log settles it.
- `vllm==0.23.0` on PyPI: `cp38-abi3-manylinux_2_28_x86_64` (274 MB, 2026-06-13; a `-2` rebuild 2026-06-15), `requires_python <3.15,>=3.10`, `Requires-Dist: torch==2.11.0, torchaudio==2.11.0, torchvision==0.26.0, flashinfer-python==0.6.12, transformers>=4.56 (!=5.0..5.5.0)`. All satisfied by the pinned list.
- `torch==2.11.0` on PyPI: `cp310` to `cp314` (+`t` builds) `manylinux_2_28_x86_64`, 531 MB each. The requirements pin the PyPI wheel (no `--index-url .../cu130`) together with `nvidia-cuda-runtime==13.0.96`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, so the default PyPI build of 2.11.0 is the CUDA 13.0 one; the image check prints `torch.version.cuda` to confirm.
- `transformers==5.12.1` exists (`requires_python >=3.10`). It's the first line with `glm_moe_dsa`, which `glm_swarm_nvfp4_kv.py` imports directly.
- `huggingface_hub==1.19.0` is in the pin list. `HF_HUB_ENABLE_HF_TRANSFER=1` is set as requested, but hub 1.x moved bulk transfer to `hf_xet` (also pinned, 1.5.1); if the fetch log says the variable is ignored, that's expected and the 8-way `ThreadPoolExecutor` in `node_fetch.py` still gives shard-level parallelism.
- Build-size risk: the venv layer is roughly 10 to 12 GB uncompressed (torch 531 MB wheel plus nvidia libs plus vllm plus flashinfer cubins). `ubuntu-latest` has about 14 GB free before the cleanup step and about 30 GB after. If the build fails on disk, the fallbacks are a larger runner (`ubuntu-latest-4-cores` or bigger) or building once on a rented box with `docker build` and pushing from there.

## Patched files (originals byte-identical under /root/orig/)

### node_fetch.py (rewritten, same CLI plus new flags)

- `ThreadPoolExecutor(8)` over the shard list; each file keeps the 8-try resume + backoff `fetch()`.
- `HF_TOKEN` env forwarded as `token=` to `hf_hub_download` / `snapshot_download` (the tokenizer repo `zai-org/GLM-5.2` has only 3 tries in the original and a bare warning on failure; without a token the coordinator can come up without `tokenizer.json` and die at `AutoTokenizer`).
- `--draft <repo>`: `snapshot_download` to `/root/glm4_9b_draft` (override `--draft-dir`), safetensors/json/py/txt/model only.
- `--tokenizer-only`: index + config + tokenizer, no shards (what `glm_draft_compat.py` and the loopback probe need to import `glm_swarm_nvfp4_kv`).
- Prints `shards: X GB in Ys = Z MB/s`, `draft: ...`, `total: ...`, then `NODE_FETCH_DONE` (the launcher greps it).
- `GLM_DIR` env overrides `/root/glm52nvfp4` (matches the driver's own env lookup).

### glm_swarm_nvfp4_kv.py (STAGE_TRACE)

Unset: identical control flow; the only additions on the hot path are `time.time()` calls and one module global. Set: one JSON line per message, `torch.cuda.synchronize()` before `t_comp1` so compute end is real, not launch end. Field meaning: `t_wait0` = entered `recv_msg` (start of idle wait), `t_recv0` = the 8-byte length header arrived (first bytes of the frame), `t_recv1` = payload unpickled and on the GPU, `t_comp1` = block forward done, `t_send1` = `send_msg` returned (ring: after the forward send; relay-back: after the reply to upstream, so it includes the downstream round trip; tail without `--next`: after the reply to the caller). Recv-wait = `t_recv0 - t_wait0`; wire+pickle in = `t_recv1 - t_recv0`; compute = `t_comp1 - t_recv1`; pickle+send out = `t_send1 - t_comp1`.

```diff
--- root/orig/glm_swarm_nvfp4_kv.py
+++ root/glm_swarm_nvfp4_kv.py
@@ -46,8 +46,11 @@
     return bytes(buf)
 def send_msg(sock, start_pos, hidden):
     bio = io.BytesIO(); torch.save((int(start_pos), hidden.cpu()), bio); _sendall(sock, bio.getvalue())
+_T_HDR = 0.0   # STAGE_TRACE: wall time the 8-byte length header of the last recv_msg arrived
 def recv_msg(sock):
-    sp, t = torch.load(io.BytesIO(_recvn(sock, struct.unpack("!Q", _recvn(sock, 8))[0])), weights_only=False)
+    global _T_HDR
+    n = struct.unpack("!Q", _recvn(sock, 8))[0]; _T_HDR = time.time()
+    sp, t = torch.load(io.BytesIO(_recvn(sock, n)), weights_only=False)
     return sp, t.to(dev)
 
 # ====================== NVFP4 execution + VLLM_CUTLASS (stage role) ======================
@@ -203,14 +206,19 @@
     srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
     srv.bind(("0.0.0.0", port)); srv.listen(4)
     fwd = None
+    trace = open(os.environ["STAGE_TRACE"], "a") if os.environ.get("STAGE_TRACE") else None   # one JSON line per message
     while True:
         conn, _ = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
         try:
             while True:
+                t_wait0 = time.time()
                 sp, h = recv_msg(conn)
+                t_recv1 = time.time(); t_recv0 = _T_HDR
                 if sp == 0:
                     for L in layers: L.reset()
                 h = run_block(layers, sp, h, vcfg)
+                if trace: torch.cuda.synchronize()          # only when tracing: make t_comp1 the real compute end
+                t_comp1 = time.time()
                 if nxt:
                     if fwd is None:
                         host, p = nxt.rsplit(":", 1); fwd = socket.create_connection((host, int(p)))
@@ -221,6 +229,9 @@
                         send_msg(fwd, sp, h); _, back = recv_msg(fwd); send_msg(conn, sp, back)  # relay-back
                 else:
                     send_msg(conn, sp, h)
+                if trace:
+                    trace.write(json.dumps({"sp": sp, "ntok": int(h.shape[1]), "t_wait0": t_wait0, "t_recv0": t_recv0,
+                                            "t_recv1": t_recv1, "t_comp1": t_comp1, "t_send1": time.time()}) + "\n"); trace.flush()
         except (ConnectionError, EOFError):
             print("conn closed", flush=True); fwd = None
             for L in layers: L.reset()
```

### glm_swarm_nvfp4_cg.py (COORD_TRACE)

Unset: identical (the `dt_draft`/`dt_recv` accumulations are the same values, computed from a saved timestamp instead of a second `time.time()`). Set: one line per chunk when its result is consumed (`send_pos, K, t_draft0, t_draft1, t_send, t_wait0, t_recv, verdict`), verdict `full` | `diverge_<n>` (n drafts accepted before the miss) | `stale` (discarded after a rewind), then one `{"summary": true, ...}` line with `t0`, `t_end`, `ntok`, `tok_s`, `valid`, `stale`, `accepted`, `dt_draft`, `dt_recv`. The `--plain` path is not traced (no chunks; per-token timing is in the stage traces). Timing of a chunk in flight = `t_recv - t_send`; draft time = `t_draft1 - t_draft0`; coordinator idle = `t_recv - t_wait0`.

```diff
--- root/orig/glm_swarm_nvfp4_cg.py
+++ root/glm_swarm_nvfp4_cg.py
@@ -10,7 +10,7 @@
 
   coord: python glm_swarm_nvfp4_cg.py coord --stage head:port --ret-port 29600 --depth 6 --K 2 [--compile]
 """
-import socket, time, json, argparse, torch
+import os, socket, time, json, argparse, torch
 import transformers.models.glm4.modeling_glm4 as G
 import transformers.cache_utils as CU
 import glm_swarm_nvfp4_kv as KV
@@ -101,6 +101,8 @@
             for w in range(8): dstep(cur, L + w)
     out = [cur]; pos = L; inflight = []; discard = 0; send_pos = pos; tail_tok = cur
     valid = 0; accepted = 0; wasted = 0; dt_draft = 0.0; dt_recv = 0.0
+    trace = open(os.environ["COORD_TRACE"], "a") if os.environ.get("COORD_TRACE") else None   # one JSON line per chunk
+    tr_q = []                                     # trace records of in-flight chunks, parallel to `inflight`
     def draft_k():
         nonlocal tail_tok
         ds = []; t = tail_tok; p = send_pos
@@ -112,18 +114,24 @@
         done = False
         while not done:
             while len(inflight) < depth and not done:
-                _td = time.time(); ds = draft_k(); dt_draft += time.time() - _td
+                _td = time.time(); ds = draft_k(); _td1 = time.time(); dt_draft += _td1 - _td
                 send_chunk(send_pos, [tail_tok] + ds)
+                if trace: tr_q.append({"send_pos": send_pos, "K": K, "t_draft0": _td, "t_draft1": _td1, "t_send": time.time()})
                 inflight.append((send_pos, ds)); tail_tok = ds[-1]; send_pos += K
-            _tr = time.time(); r = recv_logits(); dt_recv += time.time() - _tr
+            _tr = time.time(); r = recv_logits(); _tr1 = time.time(); dt_recv += _tr1 - _tr
             sp, ds = inflight.pop(0)
+            rec = tr_q.pop(0) if trace else None
+            if rec: rec["t_wait0"] = _tr; rec["t_recv"] = _tr1
             if discard > 0:
-                discard -= 1; wasted += 1; continue
+                discard -= 1; wasted += 1
+                if rec: rec["verdict"] = "stale"; trace.write(json.dumps(rec) + "\n")
+                continue
             n = 0
             for j in range(K):
                 if ds[j] == r[j]: n += 1
                 else: break
             valid += 1; accepted += n
+            if rec: rec["verdict"] = "full" if n == K else f"diverge_{n}"; trace.write(json.dumps(rec) + "\n")
             if n == K:
                 out.extend(ds); pos += K; cur = ds[-1]
             else:
@@ -131,6 +139,10 @@
                 discard = len(inflight); tail_tok = cur; send_pos = pos   # rewind: _WRITE_POS (set per dstep) moves the write; patch masks the stale tail
             if len(out) >= max_new or cur in eos: done = True
     dt = time.time() - t0; ntok = len(out)
+    if trace:
+        trace.write(json.dumps({"summary": True, "t0": t0, "t_end": t0 + dt, "ntok": ntok, "tok_s": ntok / dt, "depth": depth, "K": K,
+                                "compile": compile, "valid": valid, "stale": wasted, "accepted": accepted,
+                                "dt_draft": dt_draft, "dt_recv": dt_recv}) + "\n"); trace.close()
     if cur in eos and out and out[-1] in eos: out = out[:-1]
     print(f"\nGENERATED {ntok} tokens in {dt:.1f}s = {ntok/dt:.2f} tok/s | depth {depth} K {K} compile={compile} | "
           f"{valid} valid (+{wasted} stale) | mean accept {accepted/max(valid,1):.2f} | "
```

## Ladder: which script is which rung (from the docstrings and argparsers at fcf7280)

| rung | receipt-era tok/s | driver and coordinator command (cwd /root, `/root/vmoe/bin/python`) | stage class | dump |
|---|---|---|---|---|
| plain KV | 1.87 | `glm_swarm_nvfp4_kv.py coord --stage HEAD --prompt P --max-new 96` | relay (no `--ring`, tail has no `--next`) | no |
| K=6 relay | 1.99 | `glm_swarm_nvfp4_draft.py coord --stage HEAD --prompt P --max-new 96 --K 6` | relay | no |
| K=6 direct | 2.94 | `glm_swarm_nvfp4_draft.py coord --stage HEAD --ret-port 29600 --prompt P --max-new 96 --K 6` | ring (`--ring`, tail `--next` = coord ip:mapped 29600) | no |
| pipe K=2 D=6 | 16.6 | `glm_swarm_nvfp4_pipe.py coord --stage HEAD --ret-port 29600 --depth 6 --K 2 --prompt P --max-new 96` | ring | no |
| cg K=2 D=6 | 30.03 | `glm_swarm_nvfp4_cg.py coord --stage HEAD --ret-port 29600 --depth 6 --K 2 --compile --prompt P --max-new 96 --dump run.json` | ring | yes |
| cg eager (receipt's reference) | n/a | same without `--compile` | ring | yes |
| cg plain (1-token greedy over the ring, with dump) | n/a | `glm_swarm_nvfp4_cg.py coord --stage HEAD --ret-port 29600 --plain --prompt P --max-new 96 --dump run.json` | ring | yes |

`P` = `"def quicksort(arr):"`. HEAD = first stage's `public_ip:mapped 29600`. The stage command never changes across rungs except for `--ring` and the tail's `--next`.

## Things in shard's scripts Phase B must know

1. **`glm_swarm_nvfp4_spec.py` is not the K=6 rung.** It drafts with GLM-5.2's native MTP head (fp8 layer 78 from `/root/glm52_mtp/mtp_layer78.safetensors`, a file no script in the repo produces) and is relay-back only. The relay/direct K=6 rungs are `glm_swarm_nvfp4_draft.py` (GLM-4-9B, `DynamicCache`), whose `--ret-port` toggles ring direct-return. Its own docstring names the draft: "GLM-4-9B-0414, same base vocab as GLM-5.2", so the default `--draft-repo` is `zai-org/GLM-4-9B-0414`; `glm_draft_compat.py` must still print COMPATIBLE before renting seven boxes (smoke step 4 does this for three candidates).
2. **Only `cg.py` has `--dump`.** `kv.py`, `draft.py`, `pipe.py` print `GENERATED ... tok/s` and a `decoded:` line truncated to 400 to 600 chars. The launcher parses stdout for those rungs; token-id sha checks are possible only for `cg`, `cgeager`, `cgplain`. The receipt's `reference` was "eager decode, same engine, CUDA graph OFF", i.e. `cgeager`, and `reference_source` in cg's dump literally says "plain greedy KV decode (glm_swarm_nvfp4_kv.py)", so the plain-KV output was compared by eye, not by hash.
3. **The receipt has 101 token ids** = 5 prompt tokens + 96 generated (`output_token_ids` includes the prompt). `output_sha256 = sha256(json.dumps(ids))` (`phase0/proof_receipt.py::_sha`, default separators).
4. **Draft vocab gap.** GLM-4-9B has 151552 tokens, GLM-5.2 154880; every draft-based driver clamps out-of-vocab target ids to 0 before feeding the draft. A different draft repo with a different vocab silently sends garbage proposals (acceptance collapses, output stays correct).
5. **Stage restarts are not needed between runs.** When the coordinator exits, the head sees `peer closed`, drops its forward socket, and the close cascades down the chain; every stage ends back in `accept()` with caches reset. Only the ring/relay class change requires a relaunch (the launcher checks `/root/stage.class`). The coordinator binds the return port 29600 with `SO_REUSEADDR`, so back-to-back runs work; the launcher still runs `fuser -k 29600/tcp` on the coordinator before each run.
6. **The mesh RTT probe and iperf3 need port 29600 free**, so they must run before the stages come up (or add `-p 29601:29601` at create). `launch_swarm.mesh_rtt` kills whatever listens on 29600 on every node. The launcher caches the mesh in `OUT/mesh_rtt.json` and reuses it for the next rung unless `--remesh`.
7. **Never `pkill -f glm_swarm`** (matches the ssh shell and kills it; documented in `launch_swarm.py` and `safe_kill.sh`). The launcher and smoke test reap GPU processes via `nvidia-smi --query-compute-apps=pid` plus `fuser -k 29600/tcp`.
8. **`glm_swarm_nvfp4_kv.py` loads the GLM-5.2 config at import** (`GlmMoeDsaConfig.from_pretrained(GLM_DIR)` and the safetensors index), so any process that imports it (the coordinator, `glm_draft_compat.py`, the loopback probe) needs `config.json` and `model.safetensors.index.json` in `GLM_DIR` first. `node_fetch.py --tokenizer-only` provides exactly that.
9. **The tokenizer comes from `zai-org/GLM-5.2`, not the NVFP4 repo**, with only 3 tries and a warning on failure. Set `HF_TOKEN` for the coordinator fetch; the launcher forwards it.
10. **`--ring` on the tail makes its `--next` the coordinator's public ip:mapped 29600.** In relay modes the tail has no `--next`, and the coordinator reads the reply from its connection to the head. The stage's `run_block` runs `s` tokens at absolute positions and crops the KV cache to `start_pos` first, which is how rejected drafts roll back for free.
11. **Per-layer config reload.** `Layer.__init__` calls `GlmMoeDsaConfig.from_pretrained(DIR)` once per layer (13 times per stage). Harmless, but it's disk I/O during load, and load time for a 13-layer stage is dominated by `safe_open(...).get_tensor` over 70 to 90 GB of shards plus `process_weights_after_loading` for 13 x 257 experts.
12. **`.gitignore` in shard lists `phase0/launch_ring.py`, `cg_run.sh`, `cgd_run.sh`, `cg_sweep.sh`, `make_receipt.py`** as personal ops scripts that were never committed. Our `scripts/launch_ring.py` is a fresh reconstruction from `launch_swarm.py` plus the drivers' argparsers, not their file.
13. `bench_fused_moe.py` is not `glm_*` so it's at `/opt/shard/research/bench_fused_moe.py` in the image; `glm_nvfp4_moe.py` hardcodes `/root/glm52nvfp4` and layer 6 (matches `GLM_DIR` default).
14. Vast's `show instances --raw` (CLI 1.5.1 on this Mac) returns `ports` as `{"22/tcp": [{"HostIp": ..., "HostPort": "59062"}], ...}` with string ports; the launcher `int()`s them. Instances without `--direct` show empty `ports` and only `ssh_host/ssh_port` (proxy), which the ring cannot use.
