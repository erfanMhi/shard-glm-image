#!/usr/bin/env bash
# ring_down.sh [ids...]: destroy every instance labeled erfan-glm-* (or only the given ids, each label-verified). Idempotent.
set -uo pipefail
V=${VASTAI:-$HOME/.vastcli/bin/vastai}
for attempt in 1 2 3; do   # retried: a transient vast API error must not leave boxes billing (the deadman relies on this script)
$V show instances --raw 2>/dev/null | python3 -c '
import json, sys
want = set(int(x) for x in sys.argv[1:])
raw = sys.stdin.read(); k = raw.find("[")              # tolerate a non-JSON prefix line (deprecation notice) like launch_ring.vast_json
insts = json.loads(raw[k:]) if k >= 0 else []
for i in insts:
    lbl = str(i.get("label") or "")
    if lbl.startswith("erfan-glm-") and (not want or i["id"] in want): print(i["id"], lbl, i.get("actual_status"))
    elif want and i["id"] in want: print("REFUSE", i["id"], lbl, file=sys.stderr)
' "$@" | while read -r ID LBL ST; do $V destroy instance "$ID" -y >/dev/null 2>&1 && echo "destroyed $ID ($LBL, was $ST)"; done
left=$($V show instances --raw 2>/dev/null | python3 -c 'import json,sys; raw=sys.stdin.read(); k=raw.find("["); print([i["id"] for i in (json.loads(raw[k:]) if k>=0 else []) if str(i.get("label") or "").startswith("erfan-glm-") and (not sys.argv[1:] or str(i["id"]) in sys.argv[1:])])' "$@")
echo "erfan-glm-* remaining (attempt $attempt): $left"
[ "$left" = "[]" ] && break; sleep 15
done
