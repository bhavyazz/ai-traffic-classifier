"""
Feature extraction from PCAP files for encrypted-traffic classification.

Uses only packet metadata (lengths, timestamps, directions).
Never decrypts payloads.

`nfstream` is listed in requirements for flow-level experiments; this module uses
Scapy end-to-end so bursts, direction ratios, and timing/size statistics stay
consistent and fully documented for reproducibility.

Flow definition: bidirectional 5-tuple (TCP/UDP), canonicalized so both directions
merge into one flow. Direction (client->server) is inferred from the first TCP SYN
or, for UDP, the first packet's source.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scapy.all import IP, TCP, UDP, PcapReader, rdpcap
from scapy.packet import Packet


# -----------------------------------------------------------------------------
# TODO: Default paths — set to your dataset or pass CLI args (see __main__)
# -----------------------------------------------------------------------------
DEFAULT_RAW_PCAP_DIR = "data/raw"
DEFAULT_OUTPUT_CSV = "data/processed/flows_features.csv"

# Column order for ML. Import this in training scripts.
# flow_duration = last_packet_time - first_packet_time (seconds).
# bytes_per_second = total_bytes / flow_duration; packets_per_second = total_packets / flow_duration.
FEATURE_COLUMNS = (
    "total_packets",
    "total_bytes",
    "flow_duration",
    "mean_packet_size",
    "std_packet_size",
    "min_packet_size",
    "max_packet_size",
    "mean_interarrival_time",
    "std_interarrival_time",
    "direction_ratio",
    "burst_count",
    "burst_avg_size",
    "bytes_per_second",
    "packets_per_second",
)

LABELS = ("chatgpt", "claude", "copilot", "non_ai")

# Filename prefix aliases → canonical training label (e.g. nonai_001.pcap)
LABEL_ALIASES = {"nonai": "non_ai"}


def _canonical_flow_key(pkt: Packet) -> Optional[Tuple[Any, ...]]:
    """Return a hashable bidirectional flow key, or None if not IP/TCP/UDP."""
    if not pkt.haslayer(IP):
        return None
    ip = pkt[IP]
    proto = int(ip.proto)
    if proto not in (6, 17):  # TCP, UDP
        return None

    src, dst = ip.src, ip.dst
    sport, dport = 0, 0
    if pkt.haslayer(TCP):
        sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
    elif pkt.haslayer(UDP):
        sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
    else:
        return None

    ep_a = (src, sport)
    ep_b = (dst, dport)
    if ep_a <= ep_b:
        key = (proto, ep_a[0], ep_a[1], ep_b[0], ep_b[1])
    else:
        key = (proto, ep_b[0], ep_b[1], ep_a[0], ep_a[1])
    return key


def _ip_total_len(pkt: Packet) -> int:
    """IP total length field (bytes of IP datagram) — stable for stats across tools."""
    if pkt.haslayer(IP):
        return int(pkt[IP].len)
    return int(len(pkt))


def _identify_client_for_flow(
    packets: List[Packet],
) -> Tuple[Optional[str], Optional[int]]:
    """
    Return (client_ip, client_port) using first TCP SYN, else first packet src.
    """
    for pkt in packets:
        if pkt.haslayer(TCP) and pkt.haslayer(IP):
            tcp = pkt[TCP]
            flags = int(tcp.flags)
            # SYN=0x02, ACK=0x10
            if flags & 0x02 and not (flags & 0x10):
                return pkt[IP].src, int(tcp.sport)
        if pkt.haslayer(UDP) and pkt.haslayer(IP):
            return pkt[IP].src, int(pkt[UDP].sport)
    if packets and packets[0].haslayer(IP):
        p0 = packets[0]
        if p0.haslayer(TCP):
            return p0[IP].src, int(p0[TCP].sport)
        if p0.haslayer(UDP):
            return p0[IP].src, int(p0[UDP].sport)
    return None, None


def _burst_stats(timestamps: List[float], burst_gap_s: float = 0.1) -> Tuple[int, float]:
    """
    Segment flow by inter-arrival gaps > burst_gap_s.
    Returns (burst_count, average packets per burst).
    """
    if not timestamps:
        return 0, 0.0
    if len(timestamps) == 1:
        return 1, 1.0
    ts = sorted(timestamps)
    burst_sizes: List[int] = []
    cur = 1
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] > burst_gap_s:
            burst_sizes.append(cur)
            cur = 1
        else:
            cur += 1
    burst_sizes.append(cur)
    return len(burst_sizes), float(np.mean(burst_sizes))


def flows_from_packets(
    packets: List[Packet],
    *,
    burst_gap_s: float = 0.1,
) -> pd.DataFrame:
    """
    Group packets into flows and compute all paper features (no decryption).
    """
    flow_pkts: Dict[Tuple[Any, ...], List[Packet]] = defaultdict(list)
    for pkt in packets:
        key = _canonical_flow_key(pkt)
        if key is None:
            continue
        flow_pkts[key].append(pkt)

    rows: List[Dict[str, Any]] = []
    for key, pkts in flow_pkts.items():
        pkts.sort(key=lambda p: float(p.time))
        times = [float(p.time) for p in pkts]
        lens = [_ip_total_len(p) for p in pkts]

        client_ip, client_port = _identify_client_for_flow(pkts)
        c2s_bytes = 0
        s2c_bytes = 0

        for p in pkts:
            if not p.haslayer(IP):
                continue
            ip = p[IP]
            is_c2s = False
            if client_ip is not None and client_port is not None:
                if p.haslayer(TCP):
                    is_c2s = ip.src == client_ip and int(p[TCP].sport) == client_port
                elif p.haslayer(UDP):
                    is_c2s = ip.src == client_ip and int(p[UDP].sport) == client_port
            blen = _ip_total_len(p)
            if is_c2s:
                c2s_bytes += blen
            else:
                s2c_bytes += blen

        total_packets = len(pkts)
        total_bytes = sum(lens)
        t0, t1 = times[0], times[-1]
        flow_duration = max(t1 - t0, 1e-9)

        ia = np.diff(times) if len(times) > 1 else np.array([])
        mean_ia = float(np.mean(ia)) if ia.size else 0.0
        std_ia = float(np.std(ia)) if ia.size else 0.0

        denom = c2s_bytes + s2c_bytes
        direction_ratio = float(c2s_bytes / denom) if denom > 0 else 0.5

        burst_count, burst_avg_size = _burst_stats(times, burst_gap_s=burst_gap_s)

        row = {
            "flow_key": str(key),
            "total_packets": total_packets,
            "total_bytes": int(total_bytes),
            "flow_duration": float(flow_duration),
            "mean_packet_size": float(np.mean(lens)) if lens else 0.0,
            "std_packet_size": float(np.std(lens)) if lens else 0.0,
            "min_packet_size": int(np.min(lens)) if lens else 0,
            "max_packet_size": int(np.max(lens)) if lens else 0,
            "mean_interarrival_time": mean_ia,
            "std_interarrival_time": std_ia,
            "direction_ratio": direction_ratio,
            "burst_count": int(burst_count),
            "burst_avg_size": float(burst_avg_size),
            "bytes_per_second": float(total_bytes / flow_duration),
            "packets_per_second": float(total_packets / flow_duration),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def label_from_pcap_path(path: str) -> Optional[str]:
    """
    Infer label from filename: chatgpt_001.pcap -> chatgpt.
    Returns None if prefix is not a known class.
    """
    base = os.path.basename(path)
    m = re.match(r"^([a-zA-Z0-9_]+)_", base)
    if not m:
        return None
    lab = m.group(1).lower()
    lab = LABEL_ALIASES.get(lab, lab)
    if lab in LABELS:
        return lab
    return None


def extract_from_pcap(
    pcap_path: str,
    *,
    label: Optional[str] = None,
    burst_gap_s: float = 0.1,
    max_packets: Optional[int] = None,
) -> pd.DataFrame:
    """Read one pcap and return feature DataFrame with optional label column."""
    packets: List[Packet] = []
    count = 0
    # PcapReader streams large files; rdpcap for small
    try:
        reader = PcapReader(pcap_path)
        try:
            for pkt in reader:
                packets.append(pkt)
                count += 1
                if max_packets is not None and count >= max_packets:
                    break
        finally:
            reader.close()
    except Exception:
        packets = list(rdpcap(pcap_path))
        if max_packets is not None:
            packets = packets[:max_packets]

    df = flows_from_packets(packets, burst_gap_s=burst_gap_s)
    df["source_pcap"] = os.path.basename(pcap_path)
    inferred = label_from_pcap_path(pcap_path)
    final_lab = label or inferred
    if final_lab is not None:
        df["label"] = final_lab
    return df


def extract_directory(
    raw_dir: str,
    *,
    glob_pat: str = "*.pcap",
    burst_gap_s: float = 0.1,
) -> pd.DataFrame:
    """
    TODO: Point raw_dir at your captured pcaps (e.g. data/raw).
    Files should be named like chatgpt_001.pcap for automatic labels.
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, glob_pat)))
    if not paths:
        return pd.DataFrame()
    parts: List[pd.DataFrame] = []
    for p in paths:
        lab = label_from_pcap_path(p)
        part = extract_from_pcap(p, label=lab, burst_gap_s=burst_gap_s)
        if "label" not in part.columns and lab:
            part["label"] = lab
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract flow features from PCAP (no decryption).")
    p.add_argument(
        "--pcap",
        type=str,
        default=None,
        help="Single pcap file. TODO: set path or use --pcap-dir.",
    )
    p.add_argument(
        "--pcap-dir",
        type=str,
        default=None,
        help="Directory of pcaps (label from filename prefix).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/processed/flows_features.csv",
        help="Output CSV path.",
    )
    p.add_argument("--label", type=str, default=None, help="Force label for single --pcap.")
    p.add_argument(
        "--burst-gap",
        type=float,
        default=0.1,
        help="Inter-arrival gap (seconds) that ends a burst.",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV if present.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    burst = float(args.burst_gap)
    out_path = args.output or DEFAULT_OUTPUT_CSV

    if args.pcap:
        df = extract_from_pcap(args.pcap, label=args.label, burst_gap_s=burst)
    elif args.pcap_dir:
        df = extract_directory(args.pcap_dir, burst_gap_s=burst)
    else:
        # TODO: Replace DEFAULT_RAW_PCAP_DIR with your pcap collection folder.
        print(
            f"No --pcap or --pcap-dir given; using default directory: {DEFAULT_RAW_PCAP_DIR}",
            flush=True,
        )
        df = extract_directory(DEFAULT_RAW_PCAP_DIR, burst_gap_s=burst)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if args.append and os.path.isfile(out_path):
        old = pd.read_csv(out_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} flow rows to {out_path}", flush=True)
