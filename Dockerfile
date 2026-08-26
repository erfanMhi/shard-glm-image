# shard-glm: leyten/shard GLM-5.2 NVFP4 WAN ring, pinned runtime (vllm 0.23.0 / torch 2.11.0 cu13 / transformers 5.12.1).
#
# Base tag: cuda-13.2.1-auto still exists on Docker Hub (checked 2026-08-26: amd64+arm64, last pushed
# 2026-08-26, 7.4 GB compressed). It is the tag phase0/launch_swarm.py used, so hosts have its layers
# cached. Ubuntu 24.04 -> system python3 is 3.12; /venv/main (conda) is left alone.
FROM vastai/base-image:cuda-13.2.1-auto

LABEL org.opencontainers.image.source=https://github.com/erfanMhi/shard-glm-image
LABEL org.opencontainers.image.description="GLM-5.2 NVFP4 pipeline-parallel WAN ring (leyten/shard fcf7280) with pinned vmoe venv"

# iperf3 + jq (edge throughput), psmisc (fuser: the self-safe way to reap a stage), iproute2 (MTU / default route in profile_box.sh)
RUN apt-get update && apt-get install -y --no-install-recommends iperf3 jq psmisc iproute2 \
    && rm -rf /var/lib/apt/lists/*

# The pinned vmoe environment (phase0/requirements_vmoe.txt at shard fcf7280). ~12 min on a box, once here.
COPY requirements_vmoe.txt /root/
RUN python3 -m venv /root/vmoe \
    && /root/vmoe/bin/pip install --no-cache-dir -U pip \
    && /root/vmoe/bin/pip install --no-cache-dir -r /root/requirements_vmoe.txt \
    && rm -rf /root/.cache/pip \
    && /root/vmoe/bin/python -c "import sys, torch, vllm, transformers; print('py', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__, 'tf', transformers.__version__)"

# Full shard checkout (no .git) for bench_fused_moe.py, shard/topology.py, docs, receipts.
COPY shard/ /opt/shard/

# Drivers run from /root (they import glm_swarm_nvfp4_kv from cwd). root/ holds research/glm_*.py +
# phase0/node_fetch.py; three are patched (STAGE_TRACE / COORD_TRACE / parallel fetch), the
# byte-identical originals sit under /root/orig/ for diffing.
COPY root/ /root/
COPY scripts/*.sh /root/
RUN chmod +x /root/*.sh

ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    GLM_DIR=/root/glm52nvfp4

# No ENTRYPOINT/CMD override: the vast base image's entrypoint (sshd, portal, provisioning) must keep running.
