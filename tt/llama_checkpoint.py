# SPDX-License-Identifier: MIT
"""Loads OpenVLA's actual (fine-tuned) LLaMA-2-7B backbone weights directly from the
real `openvla/openvla-7b` checkpoint -- NOT the separate, gated `meta-llama
/Llama-2-7b-hf` base checkpoint. Two things make this both correct and necessary:

  - OpenVLA's own checkpoint already contains the fully fine-tuned LLaMA-2-7B weights
    under a `language_model.*` prefix (confirmed directly against
    model.safetensors.index.json: 291 keys, all 32 layers, no biases -- consistent
    with attention_bias=False), so there is no reason to touch the base checkpoint at
    all for the WEIGHTS.
  - The base checkpoint's config VALUES (hidden_size, rope_theta, etc.) are public,
    undisputed constants, not gated content -- gating covers Meta's own weights, not
    the architecture metadata. get_llama2_config() below hardcodes them directly,
    cross-checked against tt-metal's own (separate, unfinished) OpenVLA attempt at
    models/experimental/openvla/tt/open_vla.py's `LLama2OpenVLAArgs
    ._set_params_from_dict`, which caught two values this port had guessed wrong
    initially: rms_norm_eps=1e-6 (not 1e-5) and max_position_embeddings=2048 (not
    4096, which is Llama-2's max *supported* length via later scaling, not the base
    checkpoint's own trained value).

Key renaming (HF's `language_model.model.layers.N.*` -> tt_transformers' internal
Meta-style `layers.N.attention.wq.weight` etc.) is a clean, standalone rewrite of the
same mapping already proven in open_vla.py's `map_openvla_hf_to_meta_keys` -- reused
as knowledge, not imported, since that module pulls in the rest of tt_transformers'
demo/Generator machinery this port is deliberately not depending on."""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
from safetensors import safe_open


def get_llama2_config(vocab_size: int = 32064, pad_token_id: int = 32000) -> dict:
    """Standard meta-llama/Llama-2-7b-hf config values, hardcoded (public/undisputed,
    not gated content) -- with OpenVLA's own vocab_size/pad_token_id override (32000
    base tokens + action-bin/special tokens, padded to a multiple of 64)."""
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,  # plain MHA, not GQA
        "hidden_act": "silu",
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": pad_token_id,
        "vocab_size": vocab_size,
        "torch_dtype": "bfloat16",
        "use_cache": True,
    }


_HF_TO_META_TOP_LEVEL = {
    "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
    "language_model.model.norm.weight": "norm.weight",
    "language_model.lm_head.weight": "output.weight",
}
_HF_TO_META_PER_LAYER = {
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attention.wq.weight",
    "self_attn.k_proj.weight": "attention.wk.weight",
    "self_attn.v_proj.weight": "attention.wv.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
}
_LAYER_KEY_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.(.+)$")


def hf_to_meta_key(hf_key: str) -> Optional[str]:
    """Rename one HF-style OpenVLA LLM key to tt_transformers' internal Meta-style
    naming. Returns None for keys with no mapping (there shouldn't be any, for the 291
    keys this checkpoint actually has)."""
    if hf_key in _HF_TO_META_TOP_LEVEL:
        return _HF_TO_META_TOP_LEVEL[hf_key]
    m = _LAYER_KEY_RE.match(hf_key)
    if m:
        layer_idx, rest = m.group(1), m.group(2)
        if rest in _HF_TO_META_PER_LAYER:
            return f"layers.{layer_idx}.{_HF_TO_META_PER_LAYER[rest]}"
    return None


def load_openvla_llama_state_dict(hf_cache_dir: Optional[str] = None) -> dict:
    """Loads all `language_model.*` weights from the real openvla/openvla-7b
    checkpoint's 3 safetensors shards and renames them to tt_transformers' expected
    keys. Returns float32 tensors (matching this repo's convention elsewhere of
    upcasting for host-side correctness comparisons; TTNN-side loaders cast to
    bf16/bfp8 themselves)."""
    if hf_cache_dir is None:
        hf_cache_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/*"
        )
    shard_paths = sorted(glob.glob(os.path.join(hf_cache_dir, "model-0000*-of-00003.safetensors")))
    assert len(shard_paths) == 3, f"expected 3 safetensors shards, found {shard_paths}"

    state_dict = {}
    for path in shard_paths:
        with safe_open(path, framework="pt") as f:
            for hf_key in f.keys():
                if not hf_key.startswith("language_model."):
                    continue
                meta_key = hf_to_meta_key(hf_key)
                assert meta_key is not None, f"no mapping for {hf_key}"
                state_dict[meta_key] = f.get_tensor(hf_key).float()

    expected_count = 32 * len(_HF_TO_META_PER_LAYER) + len(_HF_TO_META_TOP_LEVEL)
    assert len(state_dict) == expected_count, f"expected {expected_count} renamed keys, got {len(state_dict)}"
    return state_dict
