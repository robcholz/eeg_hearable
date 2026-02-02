#!/usr/bin/env python3
"""Extract a stereo BRIR for a yaw and report basic info.

Usage:
  python scripts/debug_brir_extract.py --brir PATH --yaw 123 --out /tmp/brir_stereo.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sound_foundry.pipeline.data_generator import _extract_brir_stereo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brir", required=True, type=Path)
    parser.add_argument("--yaw", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    _extract_brir_stereo(args.brir, args.yaw, args.out)
    print(f"wrote {args.out} size={args.out.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
