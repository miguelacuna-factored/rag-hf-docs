#!/usr/bin/env bash
# Unpack rag-cache.tar (from compress-all-data.sh) back into data/ and vectordb/
# at the repo root, restoring cached chunks/eval sets/vectors without recomputing them.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [ ! -f rag-cache.tar ]; then
    echo "Error: $repo_root/rag-cache.tar not found. Download it (e.g. from OneDrive) into the repo root first." >&2
    exit 1
fi

tar -xf rag-cache.tar

echo "Extracted:"
du -sh data vectordb
