"""GLM-5.2 NVFP4 swarm — PIPELINED spec-decode with a CUDA-GRAPHED 9B draft (plan B).

Same overlap-by-1 pipeline as glm_swarm_nvfp4_pipe.py, but the draft uses StaticCache (cudagraph-able)
instead of DynamicCache. The blocker was rollback: StaticCache leaves rejected drafts in the buffer
and HF's causal mask keys off max-written length, so a re-draft attends the stale tail (g collapses).
FIX: monkeypatch GLM-4's create_causal_mask to build the mask from position_ids (key_pos <= query pos)
— position_ids is derived from cache_position, so rewinding cache_position on divergence gives both the
right mask AND the right RoPE, and it's cudagraph-safe (only the position varies; arange(MAXLEN) const).
No dcache.crop, no attention_mask kwarg. --compile wraps the draft in torch.compile(reduce-overhead).

  coord: python glm_swarm_nvfp4_cg.py coord --stage head:port --ret-port 29600 --depth 6 --K 2 [--compile]
"""
import os, socket, time, json, argparse, torch
import transformers.models.glm4.modeling_glm4 as G
import transformers.cache_utils as CU
import glm_swarm_nvfp4_kv as KV
from glm_swarm_nvfp4_kv import dev, cfg, eps, send_msg, recv_msg
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

DRAFT = "/root/glm4_9b_draft"
_MAXLEN = 4096
_WRITE_POS = None    # static-address tensor that controls the StaticCache write slot (set per dstep)

def _patched_static_update(self, key_states, value_states, *args, **kwargs):
    # stock StaticLayer.update writes at its own monotonic cumulative_length, IGNORING cache_position,
    # so rollback can't move the write. Write at _WRITE_POS instead -> rollback works AND it's cudagraph-safe
    # (the graph reads the fixed-address _WRITE_POS at replay). Our patched causal mask handles the stale tail.
    if not self.is_initialized: self.lazy_initialization(key_states, value_states)
    cp = torch.arange(key_states.shape[-2], device=self.device) + _WRITE_POS
    self.keys.index_copy_(2, cp, key_states); self.values.index_copy_(2, cp, value_states)
    return self.keys, self.values
CU.StaticLayer.update = _patched_static_update   # set per-run before the cache; arange(_MAXLEN) must be a compile-time constant

def cg_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids=None, **kw):
    dtype = inputs_embeds.dtype; d = inputs_embeds.device
    qpos = position_ids.reshape(-1)                                  # [q_len]; query at qpos[i]
    kp = torch.arange(_MAXLEN, device=d)
    allow = kp.unsqueeze(0) <= qpos.unsqueeze(1)                     # key j attended iff j <= query position
    neg = torch.finfo(dtype).min
    return torch.where(allow, torch.zeros((), dtype=dtype, device=d), torch.full((), neg, dtype=dtype, device=d))[None, None]
G.create_causal_mask = cg_causal_mask

def coord(stage_ep, prompt, max_new, K, ret_port, depth, compile=False, dump=None, plain=False, prompts_file=None, results=None, repeat=1, configs=None):
    # configs="6:2,8:2,12:2": with --prompts-file, every prompt runs under every (depth, K) back to back in a rotating order, so a slow
    # minute on the WAN hits all configurations alike instead of aliasing with one of them (interleaved design)
    cfgs = [(depth, K)] if not configs else [tuple(int(x) for x in c.split(":")) for c in configs.split(",")]
    global _MAXLEN, _WRITE_POS
    _WRITE_POS = torch.zeros((), dtype=torch.long, device=dev)
    tok = AutoTokenizer.from_pretrained(KV.DIR, trust_remote_code=True)
    embed_w = KV.raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
    lm_head_w = KV.raw("lm_head.weight").to(torch.bfloat16).to(dev)
    norm_w = KV.raw("model.norm.weight").float().to(dev)
    host, p = stage_ep.rsplit(":", 1)
    fwd = socket.create_connection((host, int(p)), timeout=300); fwd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); fwd.settimeout(300)
    ret_srv = socket.socket(); ret_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ret_srv.bind(("0.0.0.0", ret_port)); ret_srv.listen(1); ret_conn = [None]
    print(f"coord(CG depth={depth} K={K} compile={compile}) -> head {stage_ep}; tail returns on :{ret_port}", flush=True)
    eos = cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]

    outstanding = [0]                             # messages sent to the ring whose response has not been read yet (ring = FIFO, 1 reply per message)
    def send_chunk(start, toks):
        send_msg(fwd, start, torch.nn.functional.embedding(torch.tensor([toks], device=dev), embed_w)); outstanding[0] += 1
    def recv_logits():
        if ret_conn[0] is None:
            ret_conn[0], _ = ret_srv.accept(); ret_conn[0].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); ret_conn[0].settimeout(300)
        _, hb = recv_msg(ret_conn[0]); outstanding[0] -= 1
        x = hb[0].float(); xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_w
        return (xn.to(torch.bfloat16) @ lm_head_w.t()).float().argmax(-1).tolist()


    # --prompts-file: run every prompt (JSONL: {"id","cat","prompt"}) in ONE process, draft loaded and compiled once; each prompt
    # resets the stages (start_pos 0) and re-prefills the draft exactly as the single-prompt path does. --results appends one
    # JSON line per generation. Without --prompts-file the behaviour is the original single-prompt run.
    items = [{"id": "single", "cat": "", "prompt": prompt}]
    if prompts_file:
        items = [json.loads(l) for l in open(prompts_file) if l.strip()]
    ids = tok(items[0]["prompt"], return_tensors="pt").input_ids[0].tolist(); L = len(ids)
    if plain:                                     # REFERENCE: pure 1-token greedy over the ring — no draft, no spec, no cudagraph
        t0 = time.time(); send_chunk(0, ids); r = recv_logits(); cur = r[-1]; out = [cur]; pos = L
        with torch.no_grad():
            while len(out) < max_new and cur not in eos:
                send_chunk(pos, [cur]); r = recv_logits(); cur = r[0]; out.append(cur); pos += 1
        dt = time.time() - t0
        if cur in eos and out and out[-1] in eos: out = out[:-1]
        print(f"\nPLAIN GREEDY {len(out)} tokens in {dt:.1f}s = {len(out)/dt:.2f} tok/s (no draft/spec/cudagraph)", flush=True)
        print("decoded:", repr(tok.decode(ids + out, skip_special_tokens=True)[:600]), flush=True)
        if dump:
            json.dump({"prompt": prompt, "output_text": tok.decode(ids + out, skip_special_tokens=True),
                       "output_token_ids": ids + out, "tok_s_warm": round(len(out) / dt, 2)}, open(dump, "w"))
            print(f"dumped reference -> {dump}", flush=True)
        return len(out) / dt
    print("loading draft GLM-4-9B...", flush=True)
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    print(f"draft loaded ({torch.cuda.memory_allocated()/1e9:.1f} GB)", flush=True)
    DVOCAB = draft.config.vocab_size
    Lmax = max(len(tok(it["prompt"], return_tensors="pt").input_ids[0]) for it in items)
    _MAXLEN = max(2048, Lmax + 4 * max_new + max(d_ * k_ for d_, k_ in cfgs) + 256)
    dcache = StaticCache(config=draft.config, max_cache_len=_MAXLEN, device=dev, dtype=torch.bfloat16)
    step = torch.compile(draft, mode="reduce-overhead", fullgraph=False) if compile else draft
    _inp = torch.zeros((1, 1), dtype=torch.long, device=dev); _cp = torch.zeros((1,), dtype=torch.long, device=dev)
    _pid = torch.zeros((1, 1), dtype=torch.long, device=dev)
    def dstep(t, position):                       # pass position_ids EXPLICITLY (= the rewound pos) so RoPE + the
        _inp[0, 0] = t if t < DVOCAB else 0       # patched mask both use it, not the cache's max-written seq length
        _cp[0] = position; _pid[0, 0] = position; _WRITE_POS.fill_(position)   # _WRITE_POS drives the cache write slot
        if compile: torch.compiler.cudagraph_mark_step_begin()   # else cudagraph-trees reuses buffers across calls -> corrupt drafts
        return int(step(input_ids=_inp, position_ids=_pid, past_key_values=dcache, cache_position=_cp, use_cache=True).logits[0, -1].argmax())

    def run_one(prompt, pid="single", depth=depth, K=K):
        ids = tok(prompt, return_tensors="pt").input_ids[0].tolist(); L = len(ids)
        with torch.no_grad():
            send_chunk(0, ids); r = recv_logits(); cur = r[-1]          # start_pos 0 resets every stage's KV cache
            _WRITE_POS.fill_(0)                                       # prefill writes [0..L-1]
            draft(input_ids=torch.tensor([[min(t, DVOCAB - 1) for t in ids]], device=dev), past_key_values=dcache,
                  cache_position=torch.arange(L, device=dev), position_ids=torch.arange(L, device=dev)[None], use_cache=True)
            if compile:
                for w in range(8): dstep(cur, L + w)
        out = [cur]; pos = L; inflight = []; discard = 0; send_pos = pos; tail_tok = cur
        valid = 0; accepted = 0; wasted = 0; dt_draft = 0.0; dt_recv = 0.0; t_first = None; n_full = 0
        trace = open(os.environ["COORD_TRACE"], "a") if os.environ.get("COORD_TRACE") else None   # one JSON line per chunk
        tr_q = []                                     # trace records of in-flight chunks, parallel to `inflight`
        def draft_k():
            nonlocal tail_tok
            ds = []; t = tail_tok; p = send_pos
            for _ in range(K):
                t = dstep(t, p); ds.append(t); p += 1
            return ds
        t0 = time.time()
        with torch.no_grad():
            done = False
            while not done:
                while len(inflight) < depth and not done:
                    _td = time.time(); ds = draft_k(); _td1 = time.time(); dt_draft += _td1 - _td
                    send_chunk(send_pos, [tail_tok] + ds)
                    if trace: tr_q.append({"id": pid, "send_pos": send_pos, "K": K, "t_draft0": _td, "t_draft1": _td1, "t_send": time.time()})
                    inflight.append((send_pos, ds)); tail_tok = ds[-1]; send_pos += K
                _tr = time.time(); r = recv_logits(); _tr1 = time.time(); dt_recv += _tr1 - _tr
                if t_first is None: t_first = _tr1          # first verified chunk back: the pipe is filled from here on
                sp, ds = inflight.pop(0)
                rec = tr_q.pop(0) if trace else None
                if rec: rec["t_wait0"] = _tr; rec["t_recv"] = _tr1
                if discard > 0:
                    discard -= 1; wasted += 1
                    if rec: rec["verdict"] = "stale"; trace.write(json.dumps(rec) + "\n")
                    continue
                n = 0
                for j in range(K):
                    if ds[j] == r[j]: n += 1
                    else: break
                valid += 1; accepted += n
                if rec: rec["verdict"] = "full" if n == K else f"diverge_{n}"; trace.write(json.dumps(rec) + "\n")
                if n == K:
                    n_full += 1; out.extend(ds); pos += K; cur = ds[-1]
                else:
                    out.extend(ds[:n] + [r[n]]); cur = r[n]; pos += n + 1
                    discard = len(inflight); tail_tok = cur; send_pos = pos   # rewind: _WRITE_POS (set per dstep) moves the write; patch masks the stale tail
                if len(out) >= max_new or cur in eos: done = True
        dt = time.time() - t0; ntok = len(out)
        if prompts_file:                              # multi-prompt: consume the responses of the chunks still in flight (depth-1 of them);
            with torch.no_grad():                     # otherwise the next prompt reads them as its prefill result. Not timed (dt is final).
                while outstanding[0] > 0: recv_logits()   # (single-prompt path unchanged: the process exits and the sockets drop them)
        if trace:
            trace.write(json.dumps({"summary": True, "id": pid, "t0": t0, "t_end": t0 + dt, "ntok": ntok, "tok_s": ntok / dt, "depth": depth, "K": K,
                                    "compile": compile, "valid": valid, "stale": wasted, "accepted": accepted,
                                    "dt_draft": dt_draft, "dt_recv": dt_recv}) + "\n"); trace.close()
        if cur in eos and out and out[-1] in eos: out = out[:-1]
        print(f"\nGENERATED {ntok} tokens in {dt:.1f}s = {ntok/dt:.2f} tok/s | depth {depth} K {K} compile={compile} | "
              f"{valid} valid (+{wasted} stale) | mean accept {accepted/max(valid,1):.2f} | "
              f"{(accepted+valid)/max(valid,1):.2f} tok/valid-traversal" + (f" | id {pid}" if pid != "single" else ""), flush=True)
        print(f"  time split: draft {dt_draft:.1f}s ({dt_draft/dt:.0%}) | recv-wait {dt_recv:.1f}s ({dt_recv/dt:.0%})", flush=True)
        print("decoded:", repr(tok.decode(ids + out, skip_special_tokens=True)[:600]), flush=True)
        return dict(ids=ids, out=out, ntok=ntok, dt=dt, valid=valid, wasted=wasted, accepted=accepted, dt_draft=dt_draft, dt_recv=dt_recv,
                    t0=t0, t_first=t_first if t_first is not None else t0 + dt, n_full=n_full, eos_hit=bool(cur in eos), depth=depth, K=K)

    last = None
    for rep in range(max(1, repeat)):
      for i_it, it in enumerate(items):
        j = (i_it + rep) % len(cfgs); rot = cfgs[j:] + cfgs[:j]     # every configuration takes every slot of the rotation over the pass
        for (d_, k_) in rot:
            pid = it.get("id", "single")
            try:
                R = run_one(it["prompt"], pid, d_, k_); last = R
            except Exception as e:                    # record the failure per prompt (partial results stay valid). Go on ONLY when the ring is
                print(f"\nPROMPT FAILED id {pid}: {type(e).__name__}: {e} (outstanding {outstanding[0]})", flush=True)   # provably clean:
                if results:                           # a socket error/timeout (OSError) or unread responses would misalign every later prompt.
                    with open(results, "a") as f:
                        f.write(json.dumps({"id": it.get("id"), "cat": it.get("cat", ""), "rep": rep, "depth": d_, "K": k_, "compile": compile,
                                            "error": f"{type(e).__name__}: {e}"[:300], "outstanding": outstanding[0]}) + "\n")
                if isinstance(e, OSError) or outstanding[0] or not prompts_file: raise
                continue
            if results:
                import hashlib
                with open(results, "a") as f:
                    f.write(json.dumps({"id": it.get("id"), "cat": it.get("cat", ""), "rep": rep, "depth": d_, "K": k_, "compile": compile,
                                        "ntok": R["ntok"], "seconds": round(R["dt"], 4), "tok_s": round(R["ntok"] / R["dt"], 3),
                                        "t0": round(R["t0"], 3), "t_first_s": round(R["t_first"] - R["t0"], 4),
                                        "tok_s_ss": round(max(0, R["ntok"] - 1 - k_) / max(1e-6, R["dt"] - (R["t_first"] - R["t0"])), 3),   # after the first reply: fill and cold windows excluded
                                        "n_div": R["valid"] - R["n_full"], "eos_hit": R["eos_hit"], "prompt_len": len(R["ids"]),
                                        "valid": R["valid"], "stale": R["wasted"], "accepted": R["accepted"],
                                        "mean_accept": round(R["accepted"] / max(R["valid"], 1), 4), "draft_s": round(R["dt_draft"], 3), "wait_s": round(R["dt_recv"], 3),
                                        "output_sha": hashlib.sha256(json.dumps(R["ids"] + R["out"]).encode()).hexdigest(),
                                        # the stop rule is chunk-aligned (EOS inside a chunk is passed; out can be max_new+K-1 long), so two
                                        # correct greedy runs can differ in length: compare on the ids up to the first EOS, capped at max_new
                                        "eos_at": next((i for i, t in enumerate(R["out"]) if t in eos), None),
                                        "output_sha_eos": hashlib.sha256(json.dumps(R["ids"] + (R["out"][:next((i for i, t in enumerate(R["out"]) if t in eos), len(R["out"]))])[:max_new]).encode()).hexdigest(),
                                        "out_ids": R["out"]}) + "\n")
    if dump and last is not None:
        json.dump({"prompt": items[-1]["prompt"], "output_text": tok.decode(last["ids"] + last["out"], skip_special_tokens=True),
                   "output_token_ids": last["ids"] + last["out"], "tok_s_warm": round(last["ntok"] / last["dt"], 2),
                   "reference_source": "plain greedy KV decode (glm_swarm_nvfp4_kv.py)"}, open(dump, "w"))
        print(f"dumped run -> {dump}", flush=True)
    return last["ntok"] / last["dt"] if last else 0.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="role", required=True)
    p = sub.add_parser("coord"); p.add_argument("--stage", required=True)
    p.add_argument("--prompt", default="def quicksort(arr):"); p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--K", type=int, default=2); p.add_argument("--ret-port", type=int, default=29600)
    p.add_argument("--depth", type=int, default=6); p.add_argument("--compile", action="store_true")
    p.add_argument("--dump", default=None); p.add_argument("--plain", action="store_true")
    p.add_argument("--prompts-file", default=None, help="JSONL of {id, cat, prompt}: run them all in this process")
    p.add_argument("--results", default=None, help="append one JSON line per generation here")
    p.add_argument("--repeat", type=int, default=1, help="passes over the prompt set")
    p.add_argument("--configs", default=None, help="interleave these depth:K configurations per prompt, e.g. 6:2,8:2,12:2 (needs --prompts-file)")
    a = ap.parse_args(); coord(a.stage, a.prompt, a.max_new, a.K, a.ret_port, a.depth, a.compile, a.dump, a.plain, a.prompts_file, a.results, a.repeat, a.configs)
