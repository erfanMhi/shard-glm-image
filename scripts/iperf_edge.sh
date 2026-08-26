#!/usr/bin/env bash
# iperf_edge.sh <server_ip> <port> [seconds] [streams]   client side: throughput of one ring edge (this box -> server)
# iperf_edge.sh --server <port> [seconds]                 server side: one-shot listener, exits after one test
#
# Only 22 and 29600 are mapped on a plain `--env '-p 29600:29600'` box, so the server must use 29600 and the
# stage must not be running yet (or add `-p 29601:29601` at create time and use 29601). Prints one JSON line:
# sender/receiver Mbit/s, retransmits, mean RTT reported by iperf3 (microseconds), plus a human summary.
set -uo pipefail
if [ "${1:-}" = "--server" ]; then
  PORT=${2:?port}; SECS=${3:-60}
  echo "[iperf] one-shot server on :$PORT (timeout ${SECS}s idle)"
  exec timeout "$SECS" iperf3 -s -p "$PORT" -1
fi
IP=${1:?server_ip}; PORT=${2:?port}; SECS=${3:-10}; STREAMS=${4:-4}
iperf3 -c "$IP" -p "$PORT" -t "$SECS" -P "$STREAMS" -J > /tmp/iperf_edge.json 2>/tmp/iperf_edge.err
if [ ! -s /tmp/iperf_edge.json ] || jq -e '.error' /tmp/iperf_edge.json >/dev/null 2>&1; then
  echo "iperf failed: $(jq -r '.error // empty' /tmp/iperf_edge.json 2>/dev/null) $(cat /tmp/iperf_edge.err)"; exit 1
fi
jq -c --arg ip "$IP" --arg port "$PORT" '{
  server: $ip, port: $port, seconds: .start.test_start.duration, streams: .start.test_start.num_streams,
  sent_Mbps: (.end.sum_sent.bits_per_second / 1e6 | floor),
  recv_Mbps: (.end.sum_received.bits_per_second / 1e6 | floor),
  retransmits: .end.sum_sent.retransmits,
  mean_rtt_us: ([.end.streams[].sender.mean_rtt] | add / length | floor),
  max_rtt_us: ([.end.streams[].sender.max_rtt] | max),
  cwnd_max_bytes: ([.end.streams[].sender.max_snd_cwnd] | max)
}' /tmp/iperf_edge.json | tee /tmp/iperf_edge.summary.json
jq -r '"edge -> \(.server):\(.port)  \(.recv_Mbps) Mbit/s received (\(.sent_Mbps) sent), \(.retransmits) retransmits, mean rtt \(.mean_rtt_us / 1000) ms"' /tmp/iperf_edge.summary.json
