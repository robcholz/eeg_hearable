#!/usr/bin/env bash
# Usage:
#   scripts/debug_brir_stats.sh /path/to/file.wav
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/file.wav" >&2
  exit 1
fi

file="$1"

ffprobe -hide_banner -v error -select_streams a:0 \
  -show_entries stream=codec_name,codec_long_name,channels,channel_layout,sample_rate \
  -of default=nokey=1:noprint_wrappers=1 "$file"

echo "--- astats ---"
ffmpeg -hide_banner -nostats -i "$1" \
  -af "astats=metadata=0:reset=0" -f null - 2>&1 | grep -E "Max level|Peak level dB|RMS level dB"


echo "--- volumedetect ---"
ffmpeg -hide_banner -i "$file" -af volumedetect -f null - 2>&1 | \
  rg -n "max_volume|mean_volume" || true
