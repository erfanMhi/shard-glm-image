#!/usr/bin/env bash
# ring_down.sh [ids...]: destroy every instance labeled erfan-glm-* (or only the given ids, each label-verified). Idempotent.
set -uo pipefail
V=${VASTAI:-$HOME/.vastcli/bin/vastai}
$V show instances --raw 2>/dev/null | python3 -c '
import json, sys
want = set(int(x) for x in sys.argv[1:])
for i in json.load(sys.stdin):
    lbl = str(i.get("label") or "")
    if lbl.startswith("erfan-glm-") and (not want or i["id"] in want): print(i["id"], lbl, i.get("actual_status"))
    elif want and i["id"] in want: print("REFUSE", i["id"], lbl, file=sys.stderr)
' "$@" | while read -r ID LBL ST; do $V destroy instance "$ID" -y >/dev/null 2>&1 && echo "destroyed $ID ($LBL, was $ST)"; done
left=$($V show instances --raw 2>/dev/null | python3 -c 'import json,sys; print([i["id"] for i in json.load(sys.stdin) if str(i.get("label") or "").startswith("erfan-glm-")])')
echo "erfan-glm-* remaining: $left"
