# SPDX-License-Identifier: MIT
"""TTNN functional bring-up of SigLIP ViT-So400M/14 (google/siglip-so400m-patch14-224),
the second of OpenVLA's two fused vision towers (the first, DINOv2 ViT-L/14, is in
functional_encoder.py).

Simpler than DINOv2 in two respects: no LayerScale, and no position-embedding
interpolation needed (this checkpoint's own native training resolution is already
224px, matching OpenVLA's usage -- unlike DINOv2's 518px default). Different in others:
no CLS token at all (SigLIP pools via a learned attention-pooling head instead, see
`AttentionPoolingHead`), separate (not fused) q/k/v projections, and
`gelu_pytorch_tanh` activation (`ttnn.gelu(..., variant=ttnn.GeluVariant.Tanh)` matches
this exactly -- verified against `torch.nn.functional.gelu(approximate="tanh")` in the
op's own docstring, not just assumed).

"So400M" (shape-optimized 400M) means non-round dims by design: hidden_size=1152,
intermediate_size=4304 (ratio ~=3.74, not the usual 4x) -- these are the checkpoint's
real values, not typos.

Reference: transformers.models.siglip.modeling_siglip (SiglipVisionEmbeddings,
SiglipEncoderLayer, SiglipAttention, SiglipMLP, SiglipMultiheadAttentionPoolingHead).
Checkpoint: google/siglip-so400m-patch14-224 via
`transformers.SiglipVisionModel.from_pretrained`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def _hifi_compute_kernel_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


@dataclass
class SiglipConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    num_layers: int = 27
    patch_size: int = 14
    in_chans: int = 3
    layer_norm_eps: float = 1e-6
    image_size: int = 224

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def padded_head_dim(self) -> int:
        """72 isn't a multiple of TILE_WIDTH (32) -- SigLIP's "shape-optimized" hidden_size
        (1152 = 16 heads * 72) means the fused-QKV fast attention path (borrowed from
        DINOv2, whose head_dim=64 tiles cleanly) needs each head zero-padded to 96 to
        satisfy nlp_create_qkv_heads/SDPA/nlp_concat_heads's tile-alignment requirement.
        Same fix as tt-metal's own models/experimental/pi0/tt/ttnn_siglip.py, which hits
        the identical head_dim=72 shape (PaliGemma's vision tower is this same checkpoint)."""
        return ((self.head_dim + 31) // 32) * 32

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def seq_len(self) -> int:
        return self.grid_size * self.grid_size  # no CLS token, unlike DINOv2


def _torch_linear_to_ttnn(weight: torch.Tensor, bias: torch.Tensor, device, dtype=ttnn.float32):
    w = ttnn.from_torch(weight.t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(bias.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return w, b


def _torch_norm_to_ttnn(weight: torch.Tensor, bias: torch.Tensor, device, dtype=ttnn.float32):
    w = ttnn.from_torch(weight.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return w, b


class Embeddings(LightweightModule):
    """PatchEmbed (non-overlapping Conv2d-as-Linear, same trick as DINOv2's) + a plain
    learned position embedding lookup -- no CLS token, no interpolation (native
    resolution already matches). Matches reference `SiglipVisionEmbeddings.forward`
    with `interpolate_pos_encoding=False` (the default, and correct here since
    num_patches == num_positions for this checkpoint at 224px)."""

    def __init__(self, patch_w, patch_b, pos_embed: torch.Tensor, cfg: SiglipConfig, device):
        self.patch_w, self.patch_b = patch_w, patch_b
        self.pos_embed_tt = ttnn.from_torch(
            pos_embed.unsqueeze(0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.cfg = cfg
        self.device = device

    @classmethod
    def from_state_dict(cls, state_dict, *, cfg: SiglipConfig, device):
        w = state_dict["embeddings.patch_embedding.weight"]  # (hidden, C, p, p)
        embed_dim = w.shape[0]
        w = w.reshape(embed_dim, -1)
        b = state_dict["embeddings.patch_embedding.bias"]
        patch_w, patch_b = _torch_linear_to_ttnn(w, b, device, dtype=ttnn.bfloat16)
        pos_embed = state_dict["embeddings.position_embedding.weight"]  # (num_patches, hidden), no CLS row
        return cls(patch_w, patch_b, pos_embed, cfg, device)

    def unfold_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, C, H, W = pixel_values.shape
        p = cfg.patch_size
        assert H % p == 0 and W % p == 0
        gH, gW = H // p, W // p
        x = pixel_values.reshape(B, C, gH, p, gW, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, gH, gW, C, p, p)
        return x.reshape(B, gH * gW, C * p * p)

    def forward(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        patches = self.unfold_patches(pixel_values)
        patches_tt = ttnn.from_torch(patches, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        embeddings = ttnn.linear(
            patches_tt, self.patch_w, bias=self.patch_b, compute_kernel_config=_hifi_compute_kernel_config()
        )
        return embeddings + self.pos_embed_tt


def _pad_heads_weight(weight: torch.Tensor, *, num_heads: int, head_dim: int, padded_head_dim: int, on_output: bool) -> torch.Tensor:
    """Zero-pad an nn.Linear weight's per-head slice from head_dim to padded_head_dim.
    `on_output=True` pads the output (row) side (q/k/v projections, whose output is
    num_heads*head_dim); `on_output=False` pads the input (column) side (out_proj,
    whose input is the concatenated heads). Padding with zeros is exact, not an
    approximation: zero columns in Q/K contribute 0 to every dot product, zero rows in
    V's weight force those output columns to 0, and zero rows in out_proj's weight
    (matching those same positions) drop them from the final sum -- see
    padded_head_dim's docstring for the tt-metal precedent this mirrors."""
    pad = padded_head_dim - head_dim
    if pad == 0:
        return weight
    if on_output:
        out_dim, in_dim = weight.shape
        w = weight.reshape(num_heads, head_dim, in_dim)
        w = torch.nn.functional.pad(w, (0, 0, 0, pad))  # pad head_dim
        return w.reshape(num_heads * padded_head_dim, in_dim)
    else:
        out_dim, in_dim = weight.shape
        w = weight.reshape(out_dim, num_heads, head_dim)
        w = torch.nn.functional.pad(w, (0, pad))  # pad head_dim
        return w.reshape(out_dim, num_heads * padded_head_dim)


def _pad_heads_bias(bias: torch.Tensor, *, num_heads: int, head_dim: int, padded_head_dim: int) -> torch.Tensor:
    pad = padded_head_dim - head_dim
    if pad == 0:
        return bias
    b = bias.reshape(num_heads, head_dim)
    b = torch.nn.functional.pad(b, (0, pad))
    return b.reshape(num_heads * padded_head_dim)


class SelfAttention(LightweightModule):
    """Standard bidirectional multi-head self-attention. Reference has separate
    query/key/value Linear weights; concatenated host-side into one fused weight to
    reuse the same nlp_create_qkv_heads/SDPA fast path as functional_encoder.py's.

    Unlike DINOv2 (head_dim=64, already tile-aligned), SigLIP's head_dim=72 is not a
    multiple of TILE_WIDTH (32), which nlp_create_qkv_heads/SDPA/nlp_concat_heads
    require -- see SiglipConfig.padded_head_dim. Q/K/V and out_proj weights are
    zero-padded per-head to padded_head_dim (96) host-side before conversion."""

    def __init__(self, qkv_w, qkv_b, proj_w, proj_b, cfg: SiglipConfig):
        self.qkv_w, self.qkv_b = qkv_w, qkv_b
        self.proj_w, self.proj_b = proj_w, proj_b
        self.cfg = cfg

    @classmethod
    def from_state_dict(cls, state_dict, *, prefix: str, cfg: SiglipConfig, device):
        H, D, PD = cfg.num_heads, cfg.head_dim, cfg.padded_head_dim
        q_w = _pad_heads_weight(state_dict[f"{prefix}.self_attn.q_proj.weight"], num_heads=H, head_dim=D, padded_head_dim=PD, on_output=True)
        k_w = _pad_heads_weight(state_dict[f"{prefix}.self_attn.k_proj.weight"], num_heads=H, head_dim=D, padded_head_dim=PD, on_output=True)
        v_w = _pad_heads_weight(state_dict[f"{prefix}.self_attn.v_proj.weight"], num_heads=H, head_dim=D, padded_head_dim=PD, on_output=True)
        q_b = _pad_heads_bias(state_dict[f"{prefix}.self_attn.q_proj.bias"], num_heads=H, head_dim=D, padded_head_dim=PD)
        k_b = _pad_heads_bias(state_dict[f"{prefix}.self_attn.k_proj.bias"], num_heads=H, head_dim=D, padded_head_dim=PD)
        v_b = _pad_heads_bias(state_dict[f"{prefix}.self_attn.v_proj.bias"], num_heads=H, head_dim=D, padded_head_dim=PD)
        qkv_w, qkv_b = _torch_linear_to_ttnn(
            torch.cat([q_w, k_w, v_w], dim=0), torch.cat([q_b, k_b, v_b], dim=0), device, ttnn.bfloat16
        )
        out_w = _pad_heads_weight(
            state_dict[f"{prefix}.self_attn.out_proj.weight"], num_heads=H, head_dim=D, padded_head_dim=PD, on_output=False
        )
        proj_w, proj_b = _torch_linear_to_ttnn(
            out_w, state_dict[f"{prefix}.self_attn.out_proj.bias"], device, ttnn.bfloat16,
        )
        return cls(qkv_w, qkv_b, proj_w, proj_b, cfg)

    def forward(self, x: "ttnn.Tensor", batch: int, seq_len: int) -> "ttnn.Tensor":
        cfg = self.cfg
        H, PD = cfg.num_heads, cfg.padded_head_dim
        residual_dtype = x.dtype
        x = ttnn.typecast(x, ttnn.bfloat16)

        qkv = ttnn.linear(x, self.qkv_w, bias=self.qkv_b, compute_kernel_config=_hifi_compute_kernel_config())
        qkv = ttnn.reshape(qkv, (batch, 1, seq_len, 3 * H * PD))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, transpose_k_heads=False)

        out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=cfg.head_dim**-0.5, compute_kernel_config=_hifi_compute_kernel_config()
        )
        out = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (batch, seq_len, H * PD))
        out = ttnn.linear(out, self.proj_w, bias=self.proj_b, compute_kernel_config=_hifi_compute_kernel_config())
        return ttnn.typecast(out, residual_dtype)


class EncoderLayer(LightweightModule):
    """norm1 -> SelfAttention -> residual -> norm2 -> MLP(gelu_tanh) -> residual.
    Matches reference `SiglipEncoderLayer.forward` exactly -- no LayerScale here,
    unlike DINOv2's version of this same shape."""

    def __init__(self, attn, norm1, norm2, fc1_w, fc1_b, fc2_w, fc2_b, cfg):
        self.attn = attn
        self.norm1_w, self.norm1_b = norm1
        self.norm2_w, self.norm2_b = norm2
        self.fc1_w, self.fc1_b = fc1_w, fc1_b
        self.fc2_w, self.fc2_b = fc2_w, fc2_b
        self.cfg = cfg

    @classmethod
    def from_state_dict(cls, state_dict, *, layer_idx: int, cfg: SiglipConfig, device):
        prefix = f"encoder.layers.{layer_idx}"
        attn = SelfAttention.from_state_dict(state_dict, prefix=prefix, cfg=cfg, device=device)
        norm1 = _torch_norm_to_ttnn(state_dict[f"{prefix}.layer_norm1.weight"], state_dict[f"{prefix}.layer_norm1.bias"], device)
        norm2 = _torch_norm_to_ttnn(state_dict[f"{prefix}.layer_norm2.weight"], state_dict[f"{prefix}.layer_norm2.bias"], device)
        fc1_w, fc1_b = _torch_linear_to_ttnn(
            state_dict[f"{prefix}.mlp.fc1.weight"], state_dict[f"{prefix}.mlp.fc1.bias"], device, ttnn.bfloat16
        )
        fc2_w, fc2_b = _torch_linear_to_ttnn(
            state_dict[f"{prefix}.mlp.fc2.weight"], state_dict[f"{prefix}.mlp.fc2.bias"], device, ttnn.bfloat16
        )
        return cls(attn, norm1, norm2, fc1_w, fc1_b, fc2_w, fc2_b, cfg)

    def forward(self, x: "ttnn.Tensor", batch: int, seq_len: int) -> "ttnn.Tensor":
        eps = self.cfg.layer_norm_eps
        residual = x
        h = ttnn.layer_norm(
            x, weight=self.norm1_w, bias=self.norm1_b, epsilon=eps, compute_kernel_config=_hifi_compute_kernel_config()
        )
        h = self.attn(h, batch, seq_len)
        x = residual + h

        residual = x
        h = ttnn.layer_norm(
            x, weight=self.norm2_w, bias=self.norm2_b, epsilon=eps, compute_kernel_config=_hifi_compute_kernel_config()
        )
        h = ttnn.typecast(h, ttnn.bfloat16)
        h = ttnn.linear(h, self.fc1_w, bias=self.fc1_b, compute_kernel_config=_hifi_compute_kernel_config())
        h = ttnn.gelu(h, variant=ttnn.GeluVariant.Tanh)
        h = ttnn.linear(h, self.fc2_w, bias=self.fc2_b, compute_kernel_config=_hifi_compute_kernel_config())
        h = ttnn.typecast(h, x.dtype)
        x = residual + h
        return x


class Model(LightweightModule):
    """Embeddings -> 27x EncoderLayer -> post_layernorm. Matches reference
    `SiglipVisionTransformer.forward` up to (not including) the attention-pooling
    head -- OpenVLA's own fusion mechanism (still unconfirmed -- see README) determines
    whether it even uses the pooled output or the full per-patch sequence, so the head
    isn't ported yet; this is the `last_hidden_state` equivalent."""

    def __init__(self, embeddings: Embeddings, layers: list, final_norm, cfg: SiglipConfig):
        self.embeddings = embeddings
        self.layers = layers
        self.final_norm_w, self.final_norm_b = final_norm
        self.cfg = cfg

    @classmethod
    def from_state_dict(cls, state_dict, *, cfg: SiglipConfig, device):
        embeddings = Embeddings.from_state_dict(state_dict, cfg=cfg, device=device)
        layers = [
            EncoderLayer.from_state_dict(state_dict, layer_idx=i, cfg=cfg, device=device)
            for i in range(cfg.num_layers)
        ]
        final_norm = _torch_norm_to_ttnn(state_dict["post_layernorm.weight"], state_dict["post_layernorm.bias"], device)
        return cls(embeddings, layers, final_norm, cfg)

    def forward(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        x = self.embeddings(pixel_values)
        for layer in self.layers:
            x = layer(x, batch=pixel_values.shape[0], seq_len=self.cfg.seq_len)
        return ttnn.layer_norm(
            x, weight=self.final_norm_w, bias=self.final_norm_b, epsilon=self.cfg.layer_norm_eps,
            compute_kernel_config=_hifi_compute_kernel_config(),
        )
