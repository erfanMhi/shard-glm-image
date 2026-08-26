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
import os, sys, json, time, argparse, concurrent.futures as cf
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

TOKEN = os.environ.get("HF_TOKEN") or None

def fetch(repo, f, tries=8, local_dir=None):
    """Download with resume + backoff — no HF token means throttling/429 under fleet load."""
    for i in range(tries):
        try:
            return hf_hub_download(repo, f, local_dir=local_dir or D, token=TOKEN)
        except Exception as e:
            wait = min(60, 5 * (i + 1))
            print(f"  retry {f} ({i+1}/{tries}) after {wait}s: {str(e)[:80]}", flush=True)
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
    t_dr = time.time()
    for i in range(4):
        try:
            snapshot_download(a.draft, local_dir=a.draft_dir, token=TOKEN,
                              allow_patterns=["*.safetensors", "*.json", "*.py", "*.txt", "*.model", "*.tiktoken"],
                              max_workers=a.workers)
            break
        except Exception as e:
            if i == 3: raise
            wait = min(60, 10 * (i + 1))
            print(f"  draft retry ({i+1}/4) after {wait}s: {str(e)[:80]}", flush=True)
            time.sleep(wait)
    report("draft", _dir_bytes(a.draft_dir), time.time() - t_dr)

report("total", _dir_bytes(D) + (_dir_bytes(a.draft_dir) if a.draft else 0), time.time() - T0)
print("NODE_FETCH_DONE", flush=True)
