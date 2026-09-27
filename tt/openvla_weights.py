# SPDX-License-Identifier: MIT
"""Single source of truth for WHERE openvla-7b's files come from, and at WHICH revision.

Every weight this port runs -- LLaMA-2-7B backbone, BOTH vision towers, projector --
plus the processor's remote code and config.json's norm_stats, is read through this
module. Two bugs made this necessary (fixed in tt-openvla-serving 0.2.0):

1. **The vision towers were the wrong weights.** Earlier versions built the DINOv2 and
   SigLIP towers from timm's *generic* pretrained checkpoints
   (`timm.create_model(..., pretrained=True)`), not from openvla-7b's own
   `vision_backbone.featurizer.*` / `vision_backbone.fused_featurizer.*` tensors.
   OpenVLA fine-tunes its vision encoder, so those differ materially (relative L2
   0.20-0.46 on every sampled tensor). Both the TTNN path and the CPU reference used the
   generic weights, so their PCC agreed with each other while both disagreed with
   OpenVLA. `load_tower_state_dicts_timm_naming()` below reads the towers from the
   checkpoint itself.

2. **A fresh install could not start.** The LLaMA loader globbed
   `$HF_HOME/hub/models--openvla--openvla-7b/snapshots/*/` and never downloaded
   anything, so on any box without a pre-warmed HF cache it died with
   `AssertionError: expected 3 safetensors shards, found []`. `snapshot_dir()` below
   calls `huggingface_hub.snapshot_download`, which downloads on a cold cache and is a
   no-op (just resolves the path) on a warm one.

Configuration (all optional; the defaults reproduce the validated setup):

- `HF_MODEL` -- exported by the bundle's run.sh (`openvla/openvla-7b`). Either an HF repo
  id, or a path to a local directory that already holds the checkpoint files (e.g. the
  `weights/` dir `tt-model pull --with-weights` fills). A local dir is used as-is and no
  revision applies to it.
- `TT_MODEL_WEIGHTS_REVISION` -- the commit to fetch. tt-model-manager sets it from the
  manifest's `weights.revision`. When unset, `PINNED_REVISION` below is used, never
  "whatever main is today": `AutoProcessor.from_pretrained(..., trust_remote_code=True)`
  EXECUTES upstream Python, so an unpinned fetch runs code nobody reviewed.
- `HF_HOME` / `HF_HUB_OFFLINE` / `HF_TOKEN` -- honoured by huggingface_hub itself; nothing
  here second-guesses them.

This module deliberately imports neither ttnn nor timm: the CPU reference, the ASGI
module's import-time metadata fetch, and hardware-free tests all use it.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

DEFAULT_REPO_ID = "openvla/openvla-7b"

# openvla/openvla-7b's HEAD as of 2026-02-17 (`curl -s
# https://huggingface.co/api/models/openvla/openvla-7b | jq -r .sha`), the snapshot every
# number in this repo was measured against. Change it only together with re-running
# the correctness checks (tt/test_vision_towers_openvla_weights.py, the HF comparison).
PINNED_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"

# Everything the serving path reads: the 3 weight shards + their index, config.json
# (norm_stats), the processor/tokenizer files, and the remote-code .py files the
# processor imports. README/.gitattributes are skipped. `tokenizer.model` is the
# SentencePiece model a slow-tokenizer fallback needs.
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.py", "tokenizer.model"]

# The prefixes openvla-7b's safetensors index uses (verified against
# model.safetensors.index.json at PINNED_REVISION: 291 language_model.* keys, 6
# projector.* keys, 685 vision_backbone.* keys, all non-LLM keys in shard 1).
DINOV2_PREFIX = "vision_backbone.featurizer."
SIGLIP_PREFIX = "vision_backbone.fused_featurizer."
PROJECTOR_PREFIX = "projector."
LLM_PREFIX = "language_model."

# timm model ids + image size, straight from openvla-7b's config.json
# (`timm_model_ids`, `image_sizes`). Used for ARCHITECTURE ONLY (pretrained=False) by
# the CPU reference; the weights always come from the checkpoint.
DINOV2_TIMM_ID = "vit_large_patch14_reg4_dinov2.lvd142m"
SIGLIP_TIMM_ID = "vit_so400m_patch14_siglip_224"
IMAGE_SIZE = 224


# Captured ONCE, at import: tt/functional_llama.py's build_model_args later OVERWRITES
# os.environ["HF_MODEL"] with "NousResearch/Llama-2-7b-hf" (tt_transformers' ModelArgs
# reads its config/tokenizer skeleton from there). Reading the env lazily would then
# silently switch this module to NousResearch mid-process. backends.py imports this module
# before any ModelArgs exists, so the captured value is the operator's (run.sh's).
_HF_MODEL_AT_IMPORT = os.environ.get("HF_MODEL")


def model_id() -> str:
    """The HF repo id (or local dir) to load from: `$HF_MODEL` as it was when this module
    was first imported, else openvla/openvla-7b."""
    return _HF_MODEL_AT_IMPORT or DEFAULT_REPO_ID


def local_checkpoint_fingerprint(path: str) -> str:
    """12-hex id for a local checkpoint dir: its resolved path plus each weight/index file's
    name, size and mtime. Two different dirs, or one updated in place, get different
    tensor caches; hashing the multi-GB contents would cost more than the conversion."""
    import hashlib

    root = Path(path).resolve()
    h = hashlib.sha256(str(root).encode())
    for f in sorted(root.glob("*.safetensors")) + sorted(root.glob("*.index.json")):
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:12]


def weights_revision() -> str:
    """The revision to fetch: `$TT_MODEL_WEIGHTS_REVISION`, else the pinned sha."""
    return os.environ.get("TT_MODEL_WEIGHTS_REVISION") or PINNED_REVISION


def _is_local_dir(mid: str) -> bool:
    return os.path.isdir(mid)


@functools.lru_cache(maxsize=None)
def snapshot_dir() -> Path:
    """Local directory holding openvla-7b's files, downloading them first if needed.

    Cached per process: the backend, the processor, and config lookups all call this,
    and a warm `snapshot_download` still does an HTTP round-trip to resolve the
    revision unless HF_HUB_OFFLINE is set -- no reason to pay that more than once.
    """
    mid = model_id()
    if _is_local_dir(mid):
        path = Path(mid)
    else:
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(repo_id=mid, revision=weights_revision(), allow_patterns=ALLOW_PATTERNS)
        )
    if not (path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(
            f"{path} has no model.safetensors.index.json -- HF_MODEL={mid!r} does not look like "
            "an openvla-7b checkpoint directory"
        )
    return path


def processor_source() -> tuple[str, dict]:
    """(name_or_path, kwargs) for `AutoProcessor.from_pretrained(..., trust_remote_code=True)`.

    A repo id gets the SAME pinned revision as the weights, so the remote code that runs is
    the code that was reviewed. A local dir is taken as-is.
    """
    mid = model_id()
    if _is_local_dir(mid):
        return mid, {}
    return mid, {"revision": weights_revision()}


def load_processor():
    """openvla-7b's own processor (remote code), at the pinned revision."""
    from transformers import AutoProcessor

    name, kwargs = processor_source()
    return AutoProcessor.from_pretrained(name, trust_remote_code=True, **kwargs)


def load_config() -> dict:
    """openvla-7b's config.json (norm_stats, vocab padding, timm ids), from the snapshot."""
    with open(snapshot_dir() / "config.json") as f:
        return json.load(f)


def try_load_config_without_download() -> Optional[dict]:
    """config.json if it is reachable cheaply, else None -- for import-time metadata.

    Used by gradio_app/asgi.py to fill the unnorm_key dropdown at IMPORT time, which must
    not fail (the bundle's build-time verify imports that module with no network and no
    writable cache). Fetches only config.json, at the pinned revision; never the shards.
    """
    try:
        mid = model_id()
        if _is_local_dir(mid):
            with open(Path(mid) / "config.json") as f:
                return json.load(f)
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(mid, "config.json", revision=weights_revision())) as f:
            return json.load(f)
    except Exception:
        return None


def shard_paths() -> list[Path]:
    """The checkpoint's shard files, as listed by its own safetensors index (not a glob)."""
    root = snapshot_dir()
    with open(root / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    names = sorted(set(weight_map.values()))
    paths = [root / n for n in names]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"safetensors index lists shards that are not on disk: {missing}")
    return paths


def load_prefixed(prefix: str, *, strip: bool = True, dtype=torch.float32) -> dict:
    """Every tensor whose key starts with `prefix`, across all shards, cast to `dtype`."""
    out = {}
    for path in shard_paths():
        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    out[k[len(prefix):] if strip else k] = f.get_tensor(k).to(dtype)
    if not out:
        raise KeyError(f"no tensors with prefix {prefix!r} in {snapshot_dir()}")
    return out


def _timm_layerscale_names(sd: dict) -> dict:
    """openvla-7b stores LayerScale as `lsN.scale_factor` (modeling_prismatic.py's
    `ls_apply_patch` renames timm's `gamma` so HF's loader doesn't mangle it); timm's own
    modules, and this port's converter, call it `lsN.gamma`. Same tensor, renamed back."""
    return {k.replace(".scale_factor", ".gamma"): v for k, v in sd.items()}


def load_tower_state_dicts_timm_naming() -> tuple[dict, dict]:
    """(dinov2_sd, siglip_sd) from openvla-7b's checkpoint, in timm's own key naming.

    Suitable for `timm.create_model(<id>, pretrained=False, num_classes=0,
    img_size=224).load_state_dict(sd, strict=True)` -- verified to be an exact key match
    (no missing, no unexpected) for both towers under timm 1.0.29 and 0.9.16.
    """
    dinov2 = _timm_layerscale_names(load_prefixed(DINOV2_PREFIX))
    siglip = _timm_layerscale_names(load_prefixed(SIGLIP_PREFIX))
    return dinov2, siglip


def timm_siglip_state_dict_to_hf_style(raw: dict, *, num_layers: int) -> dict:
    """timm SigLIP naming (flat `blocks.N.*`, fused `attn.qkv`) -> the HF-style keys
    tt/functional_siglip.py's `Model.from_state_dict` reads. The attention-pool head
    (`attn_pool.*`) is dropped: OpenVLA reads the second-to-last block's patch tokens,
    so the pool never runs. `norm.*` maps to `post_layernorm.*` for completeness; the
    VisionBackbone path does not apply it either."""
    sd = {
        "embeddings.patch_embedding.weight": raw["patch_embed.proj.weight"],
        "embeddings.patch_embedding.bias": raw["patch_embed.proj.bias"],
        "embeddings.position_embedding.weight": raw["pos_embed"][0],
        "post_layernorm.weight": raw["norm.weight"],
        "post_layernorm.bias": raw["norm.bias"],
    }
    for i in range(num_layers):
        src, dst = f"blocks.{i}", f"encoder.layers.{i}"
        qkv_w, qkv_b = raw[f"{src}.attn.qkv.weight"], raw[f"{src}.attn.qkv.bias"]
        hidden = qkv_w.shape[1]
        q_w, k_w, v_w = qkv_w.split(hidden, dim=0)
        q_b, k_b, v_b = qkv_b.split(hidden, dim=0)
        sd[f"{dst}.self_attn.q_proj.weight"] = q_w
        sd[f"{dst}.self_attn.k_proj.weight"] = k_w
        sd[f"{dst}.self_attn.v_proj.weight"] = v_w
        sd[f"{dst}.self_attn.q_proj.bias"] = q_b
        sd[f"{dst}.self_attn.k_proj.bias"] = k_b
        sd[f"{dst}.self_attn.v_proj.bias"] = v_b
        sd[f"{dst}.self_attn.out_proj.weight"] = raw[f"{src}.attn.proj.weight"]
        sd[f"{dst}.self_attn.out_proj.bias"] = raw[f"{src}.attn.proj.bias"]
        sd[f"{dst}.layer_norm1.weight"] = raw[f"{src}.norm1.weight"]
        sd[f"{dst}.layer_norm1.bias"] = raw[f"{src}.norm1.bias"]
        sd[f"{dst}.layer_norm2.weight"] = raw[f"{src}.norm2.weight"]
        sd[f"{dst}.layer_norm2.bias"] = raw[f"{src}.norm2.bias"]
        sd[f"{dst}.mlp.fc1.weight"] = raw[f"{src}.mlp.fc1.weight"]
        sd[f"{dst}.mlp.fc1.bias"] = raw[f"{src}.mlp.fc1.bias"]
        sd[f"{dst}.mlp.fc2.weight"] = raw[f"{src}.mlp.fc2.weight"]
        sd[f"{dst}.mlp.fc2.bias"] = raw[f"{src}.mlp.fc2.bias"]
    return sd


def load_projector_state_dict() -> dict:
    """The 3-layer GELU MLP projector (`fc1/fc2/fc3.{weight,bias}`), float32."""
    return load_prefixed(PROJECTOR_PREFIX)


# The token upstream's `PrismaticForConditionalGeneration.predict_action` appends after
# the prompt's final ":" when it isn't already there ("insert it to match the inputs seen
# at training time"). OpenVLA's processor does NOT emit it (the tokenized
# "In: ...?\nOut:" ends in 29901, ':'), so any path that calls the LLM directly instead
# of predict_action must append it itself -- earlier versions of this port did not, which
# fed the model a sequence it never saw in training.
EMPTY_TOKEN_AFTER_COLON = 29871


def append_empty_token(input_ids: torch.Tensor) -> torch.Tensor:
    """predict_action's own input fix-up: append token 29871 unless it is already last."""
    if torch.all(input_ids[:, -1] == EMPTY_TOKEN_AFTER_COLON):
        return input_ids
    pad = torch.full((input_ids.shape[0], 1), EMPTY_TOKEN_AFTER_COLON, dtype=input_ids.dtype)
    return torch.cat([input_ids, pad], dim=1)
