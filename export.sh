#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <version> [del]" >&2
  exit 1
fi

version="$1"
action="${2:-}"

if [[ "$action" == "del" ]]; then
  rm -f "${version}.zip" "${version}-deps.zip" "${version}-manifest.zip"
  exit 0
fi

zip -r "${version}.zip" "output_dataset/${version}"
zip -r "${version}-deps.zip" "output_dataset/${version}-deps"
zip -r "${version}-manifest.zip" "output_dataset/${version}-manifest"
