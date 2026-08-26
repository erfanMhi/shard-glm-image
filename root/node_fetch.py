"""Per-node selective fetch for the GLM-5.2-NVFP4 swarm: download ONLY the safetensors files
that hold this node's assigned layers (~4 layers ~20GB), not the full 410GB. The coordinator
additionally grabs embed/norm/lm_head + tokenizer.

  stage node:  python node_fetch.py --layers 6 7 8 9
  coord node:  python node_fetch.py --coord [--layers 0 1 2 3 4 5] [--draft zai-org/GLM-4-9B-0414]
  meta only:   python node_fetch.py --tokenizer-only          (index + config + tokenizer)

Shards download in parallel (8 workers); each file keeps the resume + backoff retry. HF_TOKEN
is honored when set (the tokenizer repo zai-org/GLM-5.2 may need it; anonymous pulls throttle).
GLM_DIR overrides the target directory (default /root/glm52nvfp4).
"""
import os, sys, json, time, argparse, subprocess, concurrent.futures as cf
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN") or None
MIN_MBPS = float(os.environ.get("FETCH_MIN_MBPS", "3"))    # per-file floor (plus 90 s grace); 8 parallel files on a ~1 Gbps box run at ~12-15 MB/s each, so 20 was too tight (2026-08-26)
_SIZES = {}

def repo_sizes(repo):
    """file -> bytes for the repo (one listing per repo)."""
    if repo not in _SIZES:
        try:
            _SIZES[repo] = {e.path: (getattr(e, "size", None) or 0) for e in HfApi(token=TOKEN).list_repo_tree(repo, recursive=True)}
        except Exception as e:
            print(f"  warn: could not list {repo}: {str(e)[:80]}", flush=True); _SIZES[repo] = {}
    return _SIZES[repo]

_CHILD = ("import sys; from huggingface_hub import hf_hub_download; "
          "hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3], token=(sys.argv[4] or None))")

def fetch(repo, f, tries=8, local_dir=None):
    """Download one file in a child process with resume + backoff, killed if it runs past size/MIN_MBPS + 90 s.
    A stall seen 2026-08-26: snapshot_download sat at 0 MB/s for 30 min on one 4.9 GB file (xet path). From the third
    attempt on, HF_HUB_DISABLE_XET=1 forces the plain HTTP path; resume picks up the .incomplete blob."""
    local_dir = local_dir or D
    size = repo_sizes(repo).get(f, 0)
    budget = 90 + size / (MIN_MBPS * 1e6)
    for i in range(tries):
        env = dict(os.environ); env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        if i >= 2: env["HF_HUB_DISABLE_XET"] = "1"
        p = subprocess.Popen([sys.executable, "-c", _CHILD, repo, f, local_dir, TOKEN or ""], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        t0 = time.time(); why = ""
        while p.poll() is None:
            time.sleep(3)
            if time.time() - t0 > budget:
                p.kill(); p.wait(); why = f"over budget {budget:.0f}s for {size/1e9:.2f} GB"; break
        if p.returncode == 0 and os.path.exists(os.path.join(local_dir, f)):
            return os.path.join(local_dir, f)
        if not why: why = (p.stderr.read().decode(errors="replace").strip().splitlines() or ["?"])[-1][:100]
        wait = min(60, 5 * (i + 1))
        print(f"  retry {f} ({i+1}/{tries}) after {wait}s: {why}{' [xet off]' if i + 1 >= 2 else ''}", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"failed to fetch {f} after {tries} tries")

def _dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total

def report(label, nbytes, secs):
    gb = nbytes / 1e9
    print(f"{label}: {gb:.2f} GB in {secs:.0f}s = {nbytes / 1e6 / max(secs, 1e-9):.0f} MB/s", flush=True)

REPO = "Mapika/GLM-5.2-NVFP4"; TOK_REPO = "zai-org/GLM-5.2"
D = os.environ.get("GLM_DIR", "/root/glm52nvfp4")

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, nargs="*", default=[])
ap.add_argument("--coord", action="store_true")
ap.add_argument("--draft", default=None, help="HF repo id of the GLM-4-9B draft (coord only)")
ap.add_argument("--draft-dir", default="/root/glm4_9b_draft")
ap.add_argument("--tokenizer-only", action="store_true", help="index + config + tokenizer, no shards")
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

os.makedirs(D, exist_ok=True)
T0 = time.time()
# index + config first
for f in ["model.safetensors.index.json", "config.json"]:
    fetch(REPO, f)
idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]

def fetch_tokenizer():
    # tokenizer (from the base GLM repo; nvfp4 repo ships none)
    for f in ["tokenizer.json", "tokenizer_config.json"]:
        try:
            fetch(TOK_REPO, f, tries=3)
        except Exception as e:
            print("tokenizer fetch warn:", e, flush=True)

if a.tokenizer_only:
    fetch_tokenizer()
    report("meta fetch", _dir_bytes(D), time.time() - T0)
    print("NODE_FETCH_DONE", flush=True)
    sys.exit(0)

want = set()
for L in a.layers:
    for w, f in idx.items():
        if w.startswith(f"model.layers.{L}."):
            want.add(f)
if a.coord:
    for w in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]:
        want.add(idx[w])

print(f"node fetch: layers {a.layers} coord={a.coord} -> {len(want)} files ({a.workers} parallel)", flush=True)
for f in sorted(want):
    print("  ", f, flush=True)
t_sh = time.time()
with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
    for f in ex.map(lambda f: fetch(REPO, f), sorted(want)):
        print("  done", os.path.basename(f), flush=True)
got = sum(os.path.getsize(f"{D}/{f}") for f in want if os.path.exists(f"{D}/{f}"))
report("shards", got, time.time() - t_sh)

if a.coord:
    fetch_tokenizer()

if a.draft:
    if not a.coord:
        print("warn: --draft is a coordinator asset; fetching anyway", flush=True)
    print(f"draft fetch: {a.draft} -> {a.draft_dir}", flush=True)
    t_dr = time.time(); os.makedirs(a.draft_dir, exist_ok=True)
    exts = (".safetensors", ".json", ".py", ".txt", ".model", ".tiktoken")
    dfiles = sorted(f for f in repo_sizes(a.draft) if f.endswith(exts) and not f.startswith("."))
    if not dfiles: raise RuntimeError(f"no files listed for {a.draft}")
    print(f"  {len(dfiles)} files, {sum(repo_sizes(a.draft)[f] for f in dfiles)/1e9:.1f} GB", flush=True)
    with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for f in ex.map(lambda f: fetch(a.draft, f, local_dir=a.draft_dir), dfiles):
            print("  done", os.path.basename(f), flush=True)
    report("draft", _dir_bytes(a.draft_dir), time.time() - t_dr)

report("total", _dir_bytes(D) + (_dir_bytes(a.draft_dir) if a.draft else 0), time.time() - T0)
print("NODE_FETCH_DONE", flush=True)
