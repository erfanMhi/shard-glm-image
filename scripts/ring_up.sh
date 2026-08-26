#!/usr/bin/env bash
# ring_up.sh [N=7] [states...]: rent N single RTX PRO 6000 boxes, one per US state (cheapest verified offer per state
# with reliability>=0.92, cuda>=13, disk>=150 GB, inet_down>=500 Mbps), labels erfan-glm-ring-<state>, ssh key attached.
# Writes ring_ids.txt (id state offer price) next to this script's runs dir. Preferred states first, then fallbacks.
set -uo pipefail
V=${VASTAI:-$HOME/.vastcli/bin/vastai}; HERE=$(cd "$(dirname "$0")" && pwd); N=${1:-7}; shift || true
PREF=(${@:-California Nevada Texas Kentucky Virginia New_Jersey Massachusetts Maryland Florida Indiana North_Carolina New_York Utah})
OUT=${RING_IDS:-$HERE/../runs/ring_ids.txt}; mkdir -p "$(dirname "$OUT")"; : > "$OUT"
$V search offers 'gpu_name in ["RTX PRO 6000 WS","RTX PRO 6000 S"] geolocation in [US] rentable=true num_gpus=1 reliability2>=0.92 cuda_max_good>=13.0 disk_space>=150 inet_down>=500' --raw -o 'dph_total+' > /tmp/ring_offers.json 2>/dev/null
python3 - "$N" "${PREF[@]}" > /tmp/ring_pick.txt <<'PY'
import json, sys
n = int(sys.argv[1]); pref = [s.replace("_", " ") for s in sys.argv[2:]]
offers = json.load(open("/tmp/ring_offers.json"))
best = {}
for o in sorted(offers, key=lambda x: x["dph_total"]):
    st = (o.get("geolocation") or "").split(",")[0].strip()
    if st and st not in best: best[st] = o
picks = [best[s] for s in pref if s in best][:n]
for o in picks: print(o["id"], o["geolocation"].split(",")[0].strip().replace(" ", "_"), round(o["dph_total"], 3))
PY
cnt=$(wc -l < /tmp/ring_pick.txt | tr -d ' '); [ "$cnt" -ge "$N" ] || { echo "only $cnt states available (need $N):"; cat /tmp/ring_pick.txt; exit 1; }
tot=0
while read -r OFFER STATE PRICE; do
  ID=$(bash "$HERE/rent.sh" "$OFFER" "ring-$(echo "$STATE" | tr 'A-Z' 'a-z')" 150 2>&1 | tail -1)
  [[ "$ID" =~ ^[0-9]+$ ]] && { echo "$ID $STATE $OFFER $PRICE" | tee -a "$OUT"; tot=$(python3 -c "print(round($tot+$PRICE,3))"); } || echo "RENT FAILED for $STATE offer $OFFER: $ID"
done < /tmp/ring_pick.txt
echo "rented $(wc -l < "$OUT" | tr -d ' ') boxes, \$$tot/h -> $OUT"
