#!/usr/bin/env bash
# Stage the episod/tt-openvla v6 thin bundle in its canonical shape. Rerunnable; stages only.
#
#   scripts/package-thin.sh [OUT_DIR]            # default: dist/bundle
#   INSTALL_CHECK=1 scripts/package-thin.sh      # also run install.sh in a COPY of the staged dir
#   REFRESH_CONSTRAINTS=1 scripts/package-thin.sh  # re-resolve scripts/thin/constraints.txt first
#
# THIS SCRIPT NEVER PUSHES. `tt-model package-thin <org>/<name>` (a positional repo id)
# stages and then uploads. The command below has no repo id on purpose. To publish, a person
# re-runs that one command with `episod/tt-openvla` added after the verification in
# docs/E2E_RECIPE_V6.md (Step 2b onward), and reads the target repo first.
#
# The canonical shape (tt-model-manager docs/E2E_RECIPE_V6.md) and what each layer is here:
#
#   engine         stock ttnn==0.77.0 from PyPI                      scripts/thin/requirements.txt
#   custom kernels none: every op is stock ttnn, so there is no --ops-wheel of kernels and
#                  model.py has no ops import (deps.custom_ops stays empty, correctly)
#   tooling        tt-transformers 2.x needs ttnn==0.79.0 and does not carry the TTTv1
#                  Transformer/ModelArgs this model builds on, so no tooling wheel applies.
#                  The tt-metal v0.77.0 files the model really imports ship in extra_code/
#                  (28 files found from the AST, not the 158-file wheel the 0.2.1 bundle used)
#   model code     extra_code/{gradio_app,tt,models}                 --extra-code
#   dependencies   --constraints with every pin resolved            scripts/thin/constraints.txt
#                  The packages beyond ttnn and torch are declared by a code-free wheel
#                  (tt-openvla-runtime-deps), installed by path through --ops-wheel. Putting
#                  them in requirements.txt would resolve them against the PyTorch CPU index,
#                  which serves 2022 copies of idna/certifi/requests/urllib3.
#   capabilities   no --tool-parser / --reasoning-parser: the model has no chat API
#                  (kind tt-dit-server, /act returns a 7-DoF action)
#
# Inputs that do not live in git (recorded in PROVENANCE.json):
#   * the tt-metal v0.77.0 source files. They come from the wheel in the published
#     episod/tt-openvla bundle, pinned by revision and sha256 below. The wheel was cut from
#     tt-metal v0.77.0 (its METADATA says so). Override with CLOSURE_WHEEL=<local path>; the
#     sha256 is checked either way.
#
# Needs: tt-model (editable install of tenstorrent/tt-model-manager main), uv, hf, python3.
# No hardware: nothing here imports ttnn or opens /dev/tenstorrent.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIN="$REPO/scripts/thin"
OUT="${1:-$REPO/dist/bundle}"
WORK="${WORK:-$(mktemp -d)}"          # scratch: closure source, extra_code, deps wheel
mkdir -p "$WORK"

# ---- pinned inputs -------------------------------------------------------------------
BUNDLE_NAME="tt-openvla"
BUNDLE_VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$REPO/pyproject.toml" | head -1)"
TTNN_VERSION="0.77.0"
PYTHON_VERSION="3.12.12"              # the interpreter the 0.2.x bundle was verified on
WEIGHTS_REPO="openvla/openvla-7b"
WEIGHTS_REV="47a0ec7fc4ec123775a391911046cf33cf9ed83f"
CLOSURE_HF_REPO="episod/tt-openvla"
CLOSURE_HF_REV="a6b1716137edc72d380e38743f87a8d7d3feaa12"
CLOSURE_WHEEL_FILE="wheels/tt_openvla_models_closure-0.77.0-py3-none-any.whl"
CLOSURE_WHEEL_SHA="47b658fb62c30ffca47a58bfe3c2acbf3f3f17f20c3d38b06daa1e9812dbff8b"

for tool in tt-model uv python3; do
  command -v "$tool" >/dev/null || { echo "package-thin.sh: $tool not found on PATH" >&2; exit 1; }
done

# ---- optional: re-resolve constraints.txt --------------------------------------------
# Two resolves, because install step 1 (ttnn + torch) runs with the PyTorch CPU index in
# first position and step 3 (the runtime-deps wheel) runs without it:
#   1. resolve requirements.txt the way step 1 does (first-index), so every package the CPU
#      index carries lands on a version that index serves;
#   2. resolve everything with those pins fixed, taking the rest from PyPI.
if [ "${REFRESH_CONSTRAINTS:-0}" = "1" ]; then
  CPU_INDEX="https://download.pytorch.org/whl/cpu"
  PLAT=(--python-version "${PYTHON_VERSION%.*}" --python-platform x86_64-manylinux_2_34)
  grep -v '^[[:space:]]*#' "$THIN/runtime-deps.in" | sed 's/[[:space:]]*#.*//' | grep -v '^[[:space:]]*$' > "$WORK/deps.in"
  cat "$THIN/requirements.txt" "$WORK/deps.in" > "$WORK/all.in"
  uv pip compile "$THIN/requirements.txt" -c "$THIN/constraint-caps.txt" "${PLAT[@]}" \
      --extra-index-url "$CPU_INDEX" --no-header --no-annotate -o "$WORK/step1.txt"
  uv pip compile "$WORK/all.in" -c "$WORK/step1.txt" -c "$THIN/constraint-caps.txt" "${PLAT[@]}" \
      --extra-index-url "$CPU_INDEX" --index-strategy unsafe-best-match --no-header --no-annotate \
      -o "$WORK/resolved.txt"
  { cat <<EOF_HEADER
# Fully resolved pins applied to every dependency-resolving install step (--constraints).
# Generated by \`REFRESH_CONSTRAINTS=1 scripts/package-thin.sh\` on $(date +%F) for CPython
# ${PYTHON_VERSION%.*} (manylinux_2_34 x86_64) from requirements.txt, runtime-deps.in and
# constraint-caps.txt. Do not edit by hand. What the resolve guarantees:
#  - packages the PyTorch CPU index carries sit at versions that index serves, so install
#    step 1 resolves (uv takes those packages only from that index).
#  - everything else is the newest PyPI version that satisfies runtime-deps.in. certifi,
#    requests, urllib3 and idna are current. The CPU index's 2022 copies are not used.
EOF_HEADER
    cat "$WORK/resolved.txt"; } > "$WORK/constraints.new"
  mv "$WORK/constraints.new" "$THIN/constraints.txt"
  echo "constraints.txt re-resolved ($(grep -c '==' "$THIN/constraints.txt") pins)"
fi

# ---- 1. tt-metal closure source -------------------------------------------------------
# Fetch the pinned wheel (read-only download), check its hash, unpack it to a scratch dir.
if [ -n "${CLOSURE_WHEEL:-}" ]; then
  WHEEL_SRC="$CLOSURE_WHEEL"
else
  hf download "$CLOSURE_HF_REPO" "$CLOSURE_WHEEL_FILE" --revision "$CLOSURE_HF_REV" \
     --local-dir "$WORK/hf" >/dev/null
  WHEEL_SRC="$WORK/hf/$CLOSURE_WHEEL_FILE"
fi
ACTUAL_SHA="$(sha256sum "$WHEEL_SRC" | cut -d' ' -f1)"
[ "$ACTUAL_SHA" = "$CLOSURE_WHEEL_SHA" ] || { echo "closure wheel sha256 mismatch: $ACTUAL_SHA" >&2; exit 1; }
rm -rf "$WORK/closure_src" && mkdir -p "$WORK/closure_src"
python3 -I -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$WHEEL_SRC" "$WORK/closure_src"

# The committed list must equal what the served code can reach. A mismatch means the served
# code changed its tt-metal imports: regenerate with `closure.py derive` and review the diff.
python3 -I "$THIN/closure.py" check "$WORK/closure_src" "$THIN/served_files.txt" "$THIN/ttt_closure.txt"

# ---- 2. extra_code/ ---------------------------------------------------------------------
# The model's own code (served_files.txt) plus the tt-metal subset, laid out so run.sh's
# PYTHONPATH=$HERE/extra_code:... resolves `import gradio_app`, `import tt`, `import models`.
rm -rf "$WORK/extra_code" && mkdir -p "$WORK/extra_code"
while IFS= read -r f; do
  [ -z "$f" ] || [[ "$f" == \#* ]] && continue
  mkdir -p "$WORK/extra_code/$(dirname "$f")"
  cp "$REPO/$f" "$WORK/extra_code/$f"
done < "$THIN/served_files.txt"
while IFS= read -r f; do
  [ -z "$f" ] || [[ "$f" == \#* ]] && continue
  mkdir -p "$WORK/extra_code/$(dirname "$f")"
  cp "$WORK/closure_src/$f" "$WORK/extra_code/$f"
done < "$THIN/ttt_closure.txt"

# ---- 3. the runtime-deps wheel ----------------------------------------------------------
# A wheel with no code. Its Requires-Dist lines are runtime-deps.in. install.sh installs it by
# path in a step that has no PyTorch CPU index, under the same constraints file.
rm -rf "$WORK/depswheel" && mkdir -p "$WORK/depswheel/tt_openvla_runtime_deps" "$WORK/wheels"
python3 -I - "$THIN/runtime-deps.in" "$WORK/depswheel" "$BUNDLE_VERSION" <<'PY'
import json, pathlib, sys
src, dest, version = sys.argv[1:4]
deps = [ln.split("#", 1)[0].strip() for ln in pathlib.Path(src).read_text().splitlines()]
deps = [d for d in deps if d]
root = pathlib.Path(dest)
(root / "tt_openvla_runtime_deps" / "__init__.py").write_text(
    '"""Declares the Python packages the tt-openvla server imports. Contains no code."""\n')
(root / "pyproject.toml").write_text(
    '[build-system]\nrequires = ["setuptools==78.1.0"]\nbuild-backend = "setuptools.build_meta"\n\n'
    f'[project]\nname = "tt-openvla-runtime-deps"\nversion = "{version}"\n'
    'description = "Runtime dependencies of the tt-openvla v6 thin bundle. No code."\n'
    'requires-python = ">=3.10"\n'
    f"dependencies = {json.dumps(deps)}\n")
PY
# A fixed timestamp keeps the wheel byte-identical between runs.
SOURCE_DATE_EPOCH=315532800 uv build --wheel --out-dir "$WORK/wheels" "$WORK/depswheel" >/dev/null
DEPS_WHEEL="$(ls "$WORK"/wheels/tt_openvla_runtime_deps-*.whl)"

# ---- 4. install guards (--verify) -----------------------------------------------------
# Run by install.sh with the venv's python, after `uv pip check`. $HERE is not exported to
# them, so each locates the bundle from sys.prefix (the venv is <bundle>/venv). install.sh
# puts extra_code/ on PYTHONPATH for them, as run.sh does at serve time. None imports ttnn:
# importing it loads the runtime, and an installer must work on a host without a card.
VERIFY_TTNN='import importlib.metadata as m; v = m.version("ttnn"); assert v == "'"$TTNN_VERSION"'", f"ttnn {v}: want the stock PyPI release '"$TTNN_VERSION"'"'
# A whole-models/-tree wheel in site-packages would add a second location to the `models`
# namespace package, and so would a leftover tt_openvla_models_closure.
VERIFY_NO_MODELS_TREE='import importlib.util as u, pathlib, sys; b = pathlib.Path(sys.prefix).parent; s = u.find_spec("models"); loc = list(s.submodule_search_locations); assert loc == [str(b / "extra_code" / "models")], f"models resolves outside extra_code: {loc}"; assert u.find_spec("tt_openvla_models_closure") is None, "the 158-file models wheel is installed"'
VERIFY_EXTRA_CODE='import pathlib, sys; b = pathlib.Path(sys.prefix).parent / "extra_code"; need = ["gradio_app/asgi.py", "tt/functional_llama.py", "models/tt_transformers/tt/model.py", "models/common/lightweightmodule.py", "models/tt_transformers/tt/prefetcher/prefetcher_config.yaml"]; missing = [n for n in need if not (b / n).is_file()]; assert not missing, f"missing under extra_code/: {missing}"'
# The index-ranking failure this bundle exists to avoid: a 2022 CA bundle and CVE-era requests.
VERIFY_NOT_STALE='import importlib.metadata as m; stale = {"certifi": "2022.12.7", "requests": "2.28.1", "urllib3": "1.26.13", "idna": "3.4", "charset-normalizer": "2.1.1"}; bad = {k: v for k, v in stale.items() if m.version(k) == v}; assert not bad, f"stale packages installed: {bad}"'
VERIFY_PINS='import importlib.metadata as m; want = {"gradio": "5.40.0", "transformers": "4.52.4", "tokenizers": "0.21.4", "torch": "2.14.0+cpu"}; got = {k: m.version(k) for k in want}; assert got == want, f"{got} != {want}"'

# ---- 5. provenance ------------------------------------------------------------------------
# Staged by --provenance as PROVENANCE.json. No timestamp: two runs on one commit stage
# identical bytes (the manifest's created_at is the only expected difference).
GIT_SHA="$(git -C "$REPO" rev-parse HEAD)"
GIT_DIRTY="$(git -C "$REPO" status --porcelain --untracked-files=no | wc -l)"
python3 -I - "$WORK/provenance.json" "$GIT_SHA" "$GIT_DIRTY" "$BUNDLE_VERSION" "$CLOSURE_HF_REPO" "$CLOSURE_HF_REV" \
  "$CLOSURE_WHEEL_FILE" "$CLOSURE_WHEEL_SHA" "$DEPS_WHEEL" "$THIN/constraints.txt" "$THIN/requirements.txt" "$(tt-model --version 2>&1 | head -1)" <<'PY'
import hashlib, json, pathlib, sys
out, sha, dirty, ver, hf_repo, hf_rev, wfile, wsha, deps_wheel, cons, reqs, ttm = sys.argv[1:13]
h = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
pathlib.Path(out).write_text(json.dumps({
    "bundle": "tt-openvla", "version": ver,
    "source": {"repo": "tsingletaryTT/tt-openvla", "commit": sha, "uncommitted_tracked_changes": int(dirty)},
    "recipe": "scripts/package-thin.sh",
    "tt_metal_closure": {"source": f"{hf_repo}@{hf_rev}:{wfile}", "sha256": wsha,
                          "note": "tt-metal v0.77.0 models/; 28 files kept (scripts/thin/ttt_closure.txt)"},
    "runtime_deps_wheel": {"name": pathlib.Path(deps_wheel).name, "sha256": h(deps_wheel)},
    "constraints_sha256": h(cons), "requirements_sha256": h(reqs),
    "tt_model": ttm,
}, indent=2) + "\n")
PY

# ---- 6. stage (no repo id: this does not upload) ----------------------------------------
rm -rf "$OUT"
tt-model package-thin --out "$OUT" --name "$BUNDLE_NAME" \
  --kind tt-dit-server --app gradio_app.asgi:app --no-vllm \
  --model-py "$THIN/model.py" --extra-code "$WORK/extra_code" \
  --requirements "$THIN/requirements.txt" --constraints "$THIN/constraints.txt" \
  --python "$PYTHON_VERSION" \
  --ops-wheel "$DEPS_WHEEL" \
  --arch blackhole --mesh P300 --device-count 2 \
  --weights "$WEIGHTS_REPO" --weights-revision "$WEIGHTS_REV" \
  --verify "$VERIFY_TTNN" --verify "$VERIFY_NO_MODELS_TREE" --verify "$VERIFY_EXTRA_CODE" \
  --verify "$VERIFY_NOT_STALE" --verify "$VERIFY_PINS" \
  --readme "$THIN/card.md" --provenance "$WORK/provenance.json"
# Settings this model leaves unset, and why:
#   --mesh-fabric        gradio_app/backends.py calls ttnn.set_fabric_config itself, so the
#                        manifest stays null. Setting both would initialise fabric twice.
#   --tool-parser / --reasoning-parser   no chat API.
#   --max-model-len / --max-num-seqs / --block-size / --trace-region-bytes   vLLM settings.

# ---- 7. inspect the staged bundle ---------------------------------------------------------
python3 -I "$THIN/inspect_stage.py" "$OUT"

# ---- 8. optional: install.sh in a COPY (builds a venv; needs network and SFPI) -------------
# Never run it in $OUT: it would add venv/ and caches to the folder that gets published.
if [ "${INSTALL_CHECK:-0}" = "1" ]; then
  COPY="$WORK/install-test"
  rm -rf "$COPY" && cp -r "$OUT" "$COPY"
  bash "$COPY/install.sh"
  echo "install.sh passed in $COPY"
fi
echo "staged: $OUT"
