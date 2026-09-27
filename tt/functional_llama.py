# SPDX-License-Identifier: MIT
"""Wires OpenVLA's real LLaMA-2-7B backbone (weights extracted in llama_checkpoint.py)
into tt-metal's own `models.tt_transformers` `Transformer`/`ModelArgs` -- reused
directly rather than hand-ported, per the explicit decision to build this port's own
minimal, from-scratch integration while still using what tt-metal's existing framework
already gets right (attention/RoPE/KV-cache kernels, tuned program configs). See
llama_checkpoint.py's module docstring for why this targets openvla-7b's own weights,
not the separate gated meta-llama/Llama-2-7b-hf checkpoint.

`HF_MODEL` is pointed at `NousResearch/Llama-2-7b-hf`, a well-known ungated community
mirror of the same architecture -- used ONLY so `ModelArgs`' internal
`AutoConfig`/`AutoTokenizer.from_pretrained` calls have somewhere legitimate to read a
plain (not custom/trust_remote_code) Llama config + tokenizer skeleton from. Its actual
config VALUES are irrelevant here: `OpenVLALlamaArgs._set_params_from_dict` below
discards them in favor of `get_llama2_config()`'s explicit values, and its WEIGHTS are
never touched at all, since `load_state_dict` is overridden to return
`load_openvla_llama_state_dict()` instead of calling the base class' own (which would
otherwise try to fetch NousResearch's multi-GB shards).

Deliberately mirrors (in scope, not by importing) tt-metal's own unfinished
`models/experimental/openvla/tt/open_vla.py`'s `LLama2OpenVLAArgs` pattern, minus its
hard `HF_MODEL == "meta-llama/Llama-2-7b-hf"` assert -- see llama_checkpoint.py's
docstring for the values it caught this port getting wrong, and its README's
documented "Layer 0-2 Hidden State Divergence" bug, root-caused (by direct code
inspection) to passing the bare string `mode="prefill"` where `Transformer.forward`
compares against the `Mode.PREFILL` enum (`Mode` is a plain, non-`str` `Enum`, so the
comparison is always False) -- avoided here by always passing the actual enum."""

from __future__ import annotations

import os

import ttnn
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs

from tt.llama_checkpoint import get_llama2_config, load_openvla_llama_state_dict

_HF_MODEL_SKELETON = "NousResearch/Llama-2-7b-hf"  # ungated mirror, config/tokenizer only -- see module docstring


class OpenVLALlamaArgs(ModelArgs):
    def _set_params_from_dict(self, config):
        new_config = get_llama2_config()
        text_config = config.get("text_config", config)
        for key, value in text_config.items():
            if key not in new_config:
                new_config[key] = value
        return super()._set_params_from_dict(new_config)

    def load_state_dict(self):
        """openvla-7b's LLaMA weights, with wq/wk converted from HF to Meta RoPE layout.

        tt_transformers (with its default use_hf_rope=False) applies RoPE the Meta way:
        rotating INTERLEAVED pairs (x0,x1),(x2,x3),... of each head. HF's LLaMA checkpoints
        store wq/wk pre-permuted for HF's rotate_half, which pairs x_i with x_{i+64}. The
        base class's own HF loader fixes this with `convert_hf_qkv_to_meta_format`
        (reverse_permute on every q_proj/k_proj); this override bypasses that loader, and
        before tt-openvla-serving 0.2.0 it bypassed the permutation too -- RoPE rotated the
        wrong pairs, and the TTNN LLM disagreed with the real openvla-7b (prefill logits
        PCC 0.55-0.70, 0-1/7 action tokens). It went unnoticed because the CPU reference it
        was compared against had NO working RoPE at all (see
        llama_checkpoint.reinit_rope_buffers). Same reverse_permute tt_transformers uses."""
        from models.tt_transformers.tt.load_checkpoints import reverse_permute

        sd = load_openvla_llama_state_dict()
        for k in list(sd):
            if k.endswith(".attention.wq.weight") or k.endswith(".attention.wk.weight"):
                t = sd[k]
                sd[k] = reverse_permute(t, t.shape[0] // self.head_dim, t.shape[0], t.shape[1])
        return sd

    def weight_cache_path(self, dtype):
        """Versioned tensor-cache dir. tt_transformers caches converted weights by NAME
        only (e.g. `layers.0.attention.wqkv_sharded_2d...tensorbin`), so a cache written by
        0.1.1 -- un-permuted wq/wk, and possibly another checkpoint revision -- would be
        reloaded silently after this fix. Keying the dir on the revision + the q/k layout
        makes a stale cache miss instead of load. A local checkpoint dir has no revision, so
        it is keyed on a fingerprint of its path and shard files instead."""
        from tt.openvla_weights import local_checkpoint_fingerprint, model_id, weights_revision

        src = model_id()
        tag = f"local-{local_checkpoint_fingerprint(src)}" if os.path.isdir(src) else weights_revision()[:12]
        base = super().weight_cache_path(dtype)
        return base.parent / f"openvla-7b-{tag}-qk-meta" / base.name


def build_model_args(mesh_device, *, max_seq_len: int = 512, max_batch_size: int = 1, n_layers: int | None = None):
    import os

    os.environ["HF_MODEL"] = _HF_MODEL_SKELETON
    model_args = OpenVLALlamaArgs(
        mesh_device,
        dummy_weights=False,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        cache_hf=False,
    )
    if n_layers is not None:
        model_args.n_layers = n_layers
    return model_args


def build_model(model_args, mesh_device, *, dtype=None):
    dtype = dtype or ttnn.bfloat8_b
    state_dict = model_args.load_state_dict()
    if model_args.n_layers is not None:
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("layers.") or int(k.split(".")[1]) < model_args.n_layers
        }
    model = Transformer(
        args=model_args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
    )
    return model


def prefill_rot_mats(model, seq_len: int):
    """Slices the RotarySetup's precomputed cos/sin tables (already built by
    `Transformer.__init__`) to the actual prefill sequence length -- same slicing
    tt-metal's own open_vla.py does, just without going through `Generator`. A plain
    Python slice of a TILE-layout ttnn tensor doesn't always preserve TILE layout
    (hit directly: rotary_embedding_llama asserts cos/sin must be TILE), so re-tilize
    explicitly rather than assume the slice already is."""
    cos = model.rope_setup.cos_matrix[:, :, :seq_len, :]
    sin = model.rope_setup.sin_matrix[:, :, :seq_len, :]
    cos = ttnn.to_layout(cos, ttnn.TILE_LAYOUT)
    sin = ttnn.to_layout(sin, ttnn.TILE_LAYOUT)
    return [cos, sin]
