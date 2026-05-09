"""
Live capture helper: sniff network traffic for a fixed duration and write a labeled pcap.

Requires OS permission to capture (often admin/root). Uses only Scapy; no decryption.

Example:
    python capture/capture.py --label chatgpt --duration 60
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

from scapy.all import sniff, wrpcap

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR_DEFAULT = ROOT / "data" / "raw"

VALID_LABELS = ("chatgpt", "claude", "copilot", "non_ai")


def next_index(label: str, outdir: Path) -> int:
    """Find next NNN for label_NNN.pcap."""
    pattern = str(outdir / f"{label}_*.pcap")
    exists = glob.glob(pattern)
    best = 0
    for p in exists:
        m = re.match(rf".*{re.escape(label)}_(\d+)\.pcap$", p.replace("\\", "/"))
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture labeled pcap for the dataset.")
    p.add_argument(
        "--label",
        type=str,
        required=True,
        choices=VALID_LABELS,
        help="Traffic class label (filename prefix).",
    )
    p.add_argument("--duration", type=float, default=60.0, help="Capture duration in seconds.")
    p.add_argument(
        "--outdir",
        type=str,
        default=str(RAW_DIR_DEFAULT),
        help="TODO: override output directory for pcaps if not using data/raw.",
    )
    p.add_argument(
        "--iface",
        type=str,
        default=None,
        help="Network interface (None = Scapy default). TODO: set on your OS if needed.",
    )
    p.add_argument("--count", type=int, default=0, help="Max packets (0 = unlimited until timeout).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    idx = next_index(args.label, outdir)
    out_path = outdir / f"{args.label}_{idx:03d}.pcap"

    print(
        f"Capturing for {args.duration}s on iface={args.iface!r} -> {out_path}",
        flush=True,
    )
    kwargs = {"timeout": float(args.duration)}
    if args.iface:
        kwargs["iface"] = args.iface
    if int(args.count) > 0:
        kwargs["count"] = int(args.count)

    pkts = sniff(**kwargs)
    wrpcap(str(out_path), pkts)
    print(f"Wrote {len(pkts)} packets to {out_path}", flush=True)
    if len(pkts) == 0:
        print(
            "Warning: zero packets. On Windows, run as Administrator or choose --iface.",
            file=sys.stderr,
        )
