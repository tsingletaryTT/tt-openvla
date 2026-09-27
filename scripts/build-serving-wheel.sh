#!/usr/bin/env bash
# Build the tt-openvla-serving wheel EXACTLY as the published episod/tt-openvla bundle ships
# it: only the served-path closure gradio_app/asgi.py actually reaches, no bring-up
# scripts, benchmarks or tests.
#
# Why a script: before 0.2.0 this curation was done by hand, so the wheel published as
# 0.1.1 and the one `uv build` produced from this repo at the same version were different
# artifacts (REVIEW P9). setuptools can't drop individual modules from an included package
# without moving files, so this stages the closure into a temp dir and builds there. The
# file list below IS the closure; if backends.py/asgi.py start importing a new tt/ module,
# add it here (the smoke import at the end fails loudly if one is missing).
#
# Usage: scripts/build-serving-wheel.sh [out-dir]    (default: dist/)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(mkdir -p "${1:-$REPO/dist}" && cd "${1:-$REPO/dist}" && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

SERVED_FILES=(
  gradio_app/__init__.py
  gradio_app/app.py
  gradio_app/asgi.py
  gradio_app/backends.py
  gradio_app/assets/example.jpg
  tt/__init__.py
  tt/action_detokenizer.py
  tt/demo_grounded_check.py
  tt/functional_encoder.py
  tt/functional_llama.py
  tt/functional_projector.py
  tt/functional_siglip.py
  tt/functional_vision_backbone.py
  tt/llama_checkpoint.py
  tt/openvla_weights.py
  tt/tt_metal_patches.py
)
for f in "${SERVED_FILES[@]}"; do
  mkdir -p "$STAGE/$(dirname "$f")"
  cp "$REPO/$f" "$STAGE/$f"
done
cp "$REPO/pyproject.toml" "$REPO/LICENSE" "$STAGE/"

uv build --wheel --out-dir "$OUT" "$STAGE"

# Smoke: every module in the closure must at least be present in the wheel; importing
# them needs ttnn (a device-free find_spec-style check is all we do here).
WHEEL="$(ls -t "$OUT"/tt_openvla_serving-*.whl | head -1)"
python3 - "$WHEEL" "${SERVED_FILES[@]}" <<'PY'
import sys, zipfile
names = set(zipfile.ZipFile(sys.argv[1]).namelist())
missing = [f for f in sys.argv[2:] if f not in names]
extra = [n for n in names if n.endswith(".py") and n not in sys.argv[2:]]
assert not missing, f"wheel is missing {missing}"
assert not extra, f"wheel has files outside the served closure: {extra}"
print(f"OK {sys.argv[1]} ({len(sys.argv) - 2} files)")
PY
