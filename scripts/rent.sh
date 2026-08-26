#!/usr/bin/env bash
# rent.sh <OFFER_ID> <label-suffix> [disk_gb]: create one on-demand vast instance from the shard-glm image and attach
# the local ssh key. The team vast account rejects account-level ssh keys ("SSH keys can only be created in
# personal context"), so the key is attached per instance. Label is always erfan-glm-<suffix>.
set -euo pipefail
V=${VASTAI:-$HOME/.vastcli/bin/vastai}
OFFER=$1; SUFFIX=$2; DISK=${3:-150}
IMAGE=${IMAGE:-ghcr.io/erfanmhi/shard-glm:cu13-vllm0.23}
KEY=${SSH_PUB:-$HOME/.ssh/id_ed25519.pub}
PUB=$(cat "$KEY")
# onstart writes our key with the modes sshd wants, independent of how vast injects attached keys
ONSTART="mkdir -p /root/.ssh; grep -qF '$PUB' /root/.ssh/authorized_keys 2>/dev/null || echo '$PUB' >> /root/.ssh/authorized_keys; chown -R root:root /root/.ssh; chmod 700 /root /root/.ssh; chmod 600 /root/.ssh/authorized_keys"
out=$($V create instance "$OFFER" --image "$IMAGE" --disk "$DISK" --ssh --direct --env '-p 29600:29600 -p 29601:29601' --onstart-cmd "$ONSTART" --label "erfan-glm-$SUFFIX" --raw)
echo "$out"
ID=$(echo "$out" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("new_contract") or d.get("id"))')
[ -n "$ID" ] && [ "$ID" != "None" ] || { echo "no instance id in create output" >&2; exit 1; }
$V attach ssh "$ID" "$(cat "$KEY")" >/dev/null && echo "instance $ID: label erfan-glm-$SUFFIX, ssh key attached"
echo "$ID"
