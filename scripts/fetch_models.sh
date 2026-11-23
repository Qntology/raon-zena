#!/usr/bin/env bash
# Download the original KRAFTON Raon weights with parallel HTTP range requests and verify them
# against the SHA-256 that Hugging Face publishes for each LFS object.
#
#   scripts/fetch_models.sh [DEST=models]
#
# Network: huggingface.co and its CDN (us.aws.cdn.hf.co / cas-bridge.xethub.hf.co) must be reachable.
set -euo pipefail
DEST=${1:-models}
PARTS=${PARTS:-8}
HF=https://huggingface.co

file_size() {
  stat -c %s "$1" 2>/dev/null || stat -f %z "$1"
}

sha256_check() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c -
  else
    shasum -a 256 -c -
  fi
}

small() { # repo file...
  local repo=$1; shift
  for f in "$@"; do
    mkdir -p "$DEST/$repo/$(dirname "$f")"
    curl -fsSL --retry 3 -o "$DEST/$repo/$f" "$HF/$repo/resolve/main/$f"
  done
}

big() { # repo file
  local repo=$1 f=$2 out="$DEST/$1/$2"
  local size; size=$(curl -fsSI "$HF/$repo/resolve/main/$f" | tr -d '\r' | awk 'tolower($1)=="x-linked-size:"{print $2; exit}')
  [ -n "$size" ] || { echo "cannot size $repo/$f" >&2; return 1; }
  if [ -f "$out" ] && [ "$(file_size "$out")" -eq "$size" ]; then echo "have $repo/$f"; return 0; fi
  mkdir -p "$(dirname "$out")"; rm -f "$out"
  local chunk=$(( (size + PARTS - 1) / PARTS )) pids=()
  for i in $(seq 0 $((PARTS - 1))); do
    local s=$((i * chunk)) e=$(((i + 1) * chunk - 1)); [ $e -ge $((size - 1)) ] && e=$((size - 1)); [ $s -gt $e ] && continue
    ( part="$out.part$i"; want=$((e - s + 1))
      for t in 1 2 3 4 5 6; do
        got=0; [ -f "$part" ] && got=$(file_size "$part")
        [ "$got" -ge "$want" ] && exit 0
        curl -fsSL --retry 3 -r $((s + got))-$e "$HF/$repo/resolve/main/$f" >> "$part" || sleep $((t * 3))
      done; exit 1 ) & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  for i in $(seq 0 $((PARTS - 1))); do [ -f "$out.part$i" ] && cat "$out.part$i" >> "$out" && rm "$out.part$i"; done
  echo "got $repo/$f ($size bytes)"
}

verify() { # repo
  curl -fsSL "$HF/api/models/$1/tree/main?recursive=true" | python3 -I -c '
import json, sys
for e in json.load(sys.stdin):
    if e.get("lfs") and e["path"].endswith(".safetensors"):
        print(e["lfs"]["oid"] + "  " + sys.argv[1] + "/" + e["path"])' "$DEST/$1" | sha256_check
}

NANO=KRAFTON/Raon-1-nano
VE=KRAFTON/Raon-VisionEncoder
small $NANO config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja \
      model.safetensors.index.json mtp/mtp_config.json
small $VE config.json
for f in model-00001-of-00002.safetensors model-00002-of-00002.safetensors model-tied-lm-head.safetensors mtp/model.safetensors; do big $NANO $f; done
big $VE model.safetensors
verify $NANO
verify $VE
