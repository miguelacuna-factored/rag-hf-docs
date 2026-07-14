#!/usr/bin/env bash
# Bundle data/ and vectordb/ into rag-cache.tar at the repo root, for out-of-band
# transfer (e.g. OneDrive) between machines instead of recomputing chunks/vectors.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

for dir in data vectordb; do
    if [ ! -d "$dir" ]; then
        echo "Error: $repo_root/$dir does not exist. Build the pipeline first (see main.py)." >&2
        exit 1
    fi
done

tar -cf rag-cache.tar data vectordb

echo "Created $repo_root/rag-cache.tar ($(du -h rag-cache.tar | cut -f1))"
