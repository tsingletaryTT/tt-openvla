# SPDX-License-Identifier: MIT
"""TTNN functional bring-up of DINOv2 ViT-L/14, one of the two vision towers OpenVLA
fuses (the other is SigLIP ViT-So400M/14, functional_siglip.py).

Bidirectional ViT encoder, standard (non-RoPE) multi-head self-attention -- simpler
than tt-vjepa2's V-JEPA2 encoder in that one specific respect, but with one
architectural piece V-JEPA2 doesn't have: LayerScale, a learned per-channel scale
applied to each sub-block's output before the residual add.

OpenVLA runs this encoder at 224px, not DINOv2's own 518px default config -- the
pretrained position embeddings (37x37 patches, +1 CLS token in the plain checkpoint)
need bicubic interpolation down to 16x16 for a 224px/14px-patch input. That
interpolation is a static, one-time operation on the position-embedding *weights*, not
something that needs to run per-inference on activations -- so it happens on the host
in plain PyTorch (see `interpolate_position_embeddings`), the same way tt-vjepa2
precomputes RoPE tables on the host rather than building a rotary-embedding kernel that
recomputes them per call.

OpenVLA's ACTUAL tower is not the plain checkpoint: `configuration_prismatic.py`'s
`dinosiglip-vit-so-224px` maps to timm id `vit_large_patch14_reg4_dinov2.lvd142m` --
the "with registers" variant (4 extra learnable tokens, "Vision Transformers Need
Registers"), loaded via `timm.create_model(...)`, not `transformers.Dinov2Model`. This
config's `num_register_tokens`/`no_embed_class` fields (default 0/False, matching the
plain checkpoint validated first) support both. Confirmed the two are NOT
interchangeable even at the reference level: HF's own port at
`facebook/dinov2-with-registers-large` gives the CLS token a nonzero learned position
embedding, while timm's `no_embed_class=True` scheme (the one this checkpoint actually
uses -- see `Embeddings`' docstring) adds position embeddings to patches only, before
CLS/register tokens are ever concatenated in -- so the two checkpoints, despite
identical parameter values for cls_token/patch_embed/register_tokens (PCC 1.0, checked
directly), do NOT compute the same forward function. This port targets timm's actual
checkpoint and forward semantics as ground truth, not HF's reimplementation -- see
`_timm_vit_reg_state_dict_to_hf_style`.

Reference (plain checkpoint, num_register_tokens=0): transformers.models.dinov2
.modeling_dinov2 (Dinov2Embeddings, Dinov2PatchEmbeddings, Dinov2Layer,
Dinov2SelfAttention, Dinov2LayerScale, Dinov2MLP). Checkpoint: facebook/dinov2-large
via `transformers.Dinov2Model.from_pretrained`, HF key naming (`embeddings.*`,
`encoder.layer.N.*`).
Reference (register variant, OpenVLA's actual tower): timm.models.vision_transformer
.VisionTransformer, `vit_large_patch14_reg4_dinov2.lvd142m` via
`timm.create_model(..., pretrained=True)`, timm's own flat key naming
(`blocks.N.*`, fused `attn.qkv`) -- translated to this module's HF-style keys by
`_timm_vit_reg_state_dict_to_hf_style` so `EncoderLayer`/`SelfAttention`/`Model` stay
checkpoint-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule


def _hifi_compute_kernel_config():
    """HiFi4 + fp32 dest accumulation for every linear/layer_norm/attention call --
    same correctness-first default tt-vjepa2's encoder uses; a bf16-weights precision
    sweep is a ttm-optimize question for later, not a blocker for proving this port
    correct."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


@dataclass
class Dinov2Config:
    hidden_size: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    mlp_ratio: int = 4
    patch_size: int = 14
    in_chans: int = 3
    layer_norm_eps: float = 1e-6
    image_size: int = 224  # OpenVLA's actual input size, not the checkpoint's own 518 default
    pretrained_image_size: int = 518  # what embeddings.position_embeddings was trained at
    num_register_tokens: int = 0  # 4 for OpenVLA's actual tower (vit_large_patch14_reg4_dinov2) -- see functional_encoder.py's module docstring
    no_embed_class: bool = False  # True for the register variant -- see Embeddings' docstring

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_prefix_tokens(self) -> int:
        return 1 + self.num_register_tokens  # CLS + registers


def _torch_linear_to_ttnn(weight: torch.Tensor, bias: torch.Tensor, device, dtype=ttnn.float32):
    """HF nn.Linear weight is (out, in); ttnn.linear wants (in, out)."""
    w = ttnn.from_torch(weight.t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(bias.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return w, b


def _torch_norm_to_ttnn(weight: torch.Tensor, bias: torch.Tensor, device, dtype=ttnn.float32):
    w = ttnn.from_torch(weight.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return w, b


def interpolate_position_embeddings(position_embeddings: torch.Tensor, cfg: Dinov2Config) -> torch.Tensor:
    """Host-side, one-time: bicubic-resizes the pretrained (518px-grid) position
    embeddings down to the actual (224px-grid) input this runs at.

    Two layouts, matching the two reference implementations this module targets:
      - `no_embed_class=False` (plain checkpoint): position_embeddings holds a CLS row
        plus patch rows; split, interpolate patches only, re-attach the CLS row.
        Matches reference `Dinov2Embeddings.interpolate_pos_encoding` exactly (same
        bicubic mode, align_corners=False, fp32 interpolation).
      - `no_embed_class=True` (register variant): timm's `no_embed_class` scheme means
        position_embeddings holds ONLY patch rows to begin with (no CLS/register
        slots at all -- see `Embeddings`' docstring), so the whole tensor is
        interpolated as patches, no split needed.

    Both verified by this module's own correctness tests comparing the *encoder's*
    output end to end, which would fail immediately if this diverged from either
    reference's own interpolation."""
    dim = position_embeddings.shape[-1]
    old_grid = cfg.pretrained_image_size // cfg.patch_size
    new_grid = cfg.grid_size

    if cfg.no_embed_class:
        if old_grid == new_grid:
            return position_embeddings
        patch_pos_embed = position_embeddings.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.to(torch.float32), size=(new_grid, new_grid), mode="bicubic", align_corners=False
        )
        return patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)

    class_pos_embed = position_embeddings[:, :1]
    patch_pos_embed = position_embeddings[:, 1:]
    if old_grid == new_grid:
        return position_embeddings

    patch_pos_embed = patch_pos_embed.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    patch_pos_embed = F.interpolate(
        patch_pos_embed.to(torch.float32), size=(new_grid, new_grid), mode="bicubic", align_corners=False
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


class PatchEmbed(LightweightModule):
    """Non-overlapping Conv2d(stride == kernel == 14) is exactly reshape-into-patches +
    Linear, same trick tt-vjepa2's PatchEmbed3D uses for its (tubelet, patch, patch)
    voxels -- here there's no tubelet axis (single image, not a video clip), so this is
    the simpler 2D case of the same idea."""

    def __init__(self, weight_out_in: "ttnn.Tensor", bias: "ttnn.Tensor", cfg: Dinov2Config, device):
        self.weight = weight_out_in
        self.bias = bias
        self.cfg = cfg
        self.device = device

    @classmethod
    def from_state_dict(cls, state_dict, *, cfg: Dinov2Config, device):
        w = state_dict["embeddings.patch_embeddings.projection.weight"]  # (hidden, C, p, p)
        embed_dim = w.shape[0]
        w = w.reshape(embed_dim, -1)  # (hidden, C*p*p)
        b = state_dict["embeddings.patch_embeddings.projection.bias"]
        weight_out_in, bias = _torch_linear_to_ttnn(w, b, device, dtype=ttnn.bfloat16)
        return cls(weight_out_in, bias, cfg, device)

    def unfold_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Host-side only: (B,C,H,W) -> (B, num_patches, C*p*p), same raster order
        Conv2d + flatten(2) + transpose(1,2) produces."""
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
        x_tt = ttnn.from_torch(patches, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        return ttnn.linear(x_tt, self.weight, bias=self.bias, compute_kernel_config=_hifi_compute_kernel_config())


class Embeddings(LightweightModule):
    """PatchEmbed -> combine with CLS (and, for the register variant, register) tokens
    and position embeddings. Two orderings, matching the two checkpoints this module
    targets (see module docstring):

      - `no_embed_class=False` (plain checkpoint): prepend CLS to patches, THEN add
        position embeddings (which include a CLS row) to the combined sequence.
        Matches reference `Dinov2Embeddings.forward`.
      - `no_embed_class=True` (register variant, OpenVLA's actual tower): add position
        embeddings to patches FIRST (position_embeddings has no CLS/register rows at
        all), THEN prepend [CLS, register_0..3] -- which therefore carry NO positional
        information whatsoever, by construction. This is timm's `no_embed_class`
        scheme (`VisionTransformer._pos_embed`), and it is NOT equivalent to
        HF's `Dinov2WithRegistersEmbeddings.forward` (which gives CLS a learned
        position and adds registers post-hoc) -- see module docstring for the PCC
        check that confirmed these diverge despite sharing parameter values.

    (Dropout is a no-op in eval mode, so omitted either way.)"""

    def __init__(self, patch_embed: PatchEmbed, cls_token: torch.Tensor, pos_embed: torch.Tensor, device, *, register_tokens: torch.Tensor | None = None, no_embed_class: bool = False):
        self.patch_embed = patch_embed
        self.cls_token = cls_token  # (1,1,hidden), plain torch -- concatenated host-side
        self.register_tokens = register_tokens  # (1,num_register_tokens,hidden) or None
        self.no_embed_class = no_embed_class
        self.pos_embed_tt = ttnn.from_torch(pos_embed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        self.device = device

    @classmethod
    def from_state_dict(cls, state_dict, *, cfg: Dinov2Config, device):
        patch_embed = PatchEmbed.from_state_dict(state_dict, cfg=cfg, device=device)
        cls_token = state_dict["embeddings.cls_token"]  # (1,1,hidden)
        register_tokens = state_dict.get("embeddings.register_tokens") if cfg.num_register_tokens > 0 else None
        pos_embed = interpolate_position_embeddings(state_dict["embeddings.position_embeddings"], cfg)
        return cls(patch_embed, cls_token, pos_embed, device, register_tokens=register_tokens, no_embed_class=cfg.no_embed_class)

    def _prefix_tokens_tt(self, batch: int) -> list:
        toks = [self.cls_token.expand(batch, -1, -1)]
        if self.register_tokens is not None:
            toks.append(self.register_tokens.expand(batch, -1, -1))
        return [ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device) for t in toks]

    def forward(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        B = pixel_values.shape[0]
        embeddings = self.patch_embed(pixel_values)  # (B, num_patches, hidden)

        if self.no_embed_class:
            embeddings = embeddings + self.pos_embed_tt
            return ttnn.concat(self._prefix_tokens_tt(B) + [embeddings], dim=1)

        embeddings = ttnn.concat(self._prefix_tokens_tt(B) + [embeddings], dim=1)
        return embeddings + self.pos_embed_tt


class SelfAttention(LightweightModule):
    """Standard bidirectional multi-head self-attention, no RoPE. Reference has
    separate query/key/value Linear weights (not V-JEPA2's fused qkv); concatenated
    host-side into one fused weight so this can reuse the same
    nlp_create_qkv_heads/SDPA fast path tt-vjepa2's RoPEAttention uses."""

    def __init__(self, qkv_w, qkv_b, proj_w, proj_b, cfg: Dinov2Config):
        self.qkv_w, self.qkv_b = qkv_w, qkv_b
        self.proj_w, self.proj_b = proj_w, proj_b
        self.cfg = cfg

    @classmethod
    def from_state_dict(cls, state_dict, *, prefix: str, cfg: Dinov2Config, device):
        q_w = state_dict[f"{prefix}.attention.attention.query.weight"]
        k_w = state_dict[f"{prefix}.attention.attention.key.weight"]
        v_w = state_dict[f"{prefix}.attention.attention.value.weight"]
        q_b = state_dict[f"{prefix}.attention.attention.query.bias"]
        k_b = state_dict[f"{prefix}.attention.attention.key.bias"]
        v_b = state_dict[f"{prefix}.attention.attention.value.bias"]
        qkv_w, qkv_b = _torch_linear_to_ttnn(
            torch.cat([q_w, k_w, v_w], dim=0), torch.cat([q_b, k_b, v_b], dim=0), device, ttnn.bfloat16
        )
        proj_w, proj_b = _torch_linear_to_ttnn(
            state_dict[f"{prefix}.attention.output.dense.weight"],
            state_dict[f"{prefix}.attention.output.dense.bias"],
            device, ttnn.bfloat16,
        )
        return cls(qkv_w, qkv_b, proj_w, proj_b, cfg)

    def forward(self, x: "ttnn.Tensor", batch: int, seq_len: int) -> "ttnn.Tensor":
        cfg = self.cfg
        H, D = cfg.num_heads, cfg.head_dim
        residual_dtype = x.dtype
        x = ttnn.typecast(x, ttnn.bfloat16)

        qkv = ttnn.linear(x, self.qkv_w, bias=self.qkv_b, compute_kernel_config=_hifi_compute_kernel_config())
        qkv = ttnn.reshape(qkv, (batch, 1, seq_len, 3 * H * D))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, transpose_k_heads=False)

        out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, compute_kernel_config=_hifi_compute_kernel_config()
        )
        out = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (batch, seq_len, H * D))
        out = ttnn.linear(out, self.proj_w, bias=self.proj_b, compute_kernel_config=_hifi_compute_kernel_config())
        return ttnn.typecast(out, residual_dtype)


class EncoderLayer(LightweightModule):
    """norm1 -> SelfAttention -> LayerScale1 -> residual -> norm2 -> MLP(gelu) ->
    LayerScale2 -> residual. Matches reference `Dinov2Layer.forward`. LayerScale is the
    one piece V-JEPA2's EncoderBlock doesn't have: a learned per-channel scale
    (`lambda1`), applied elementwise before each residual add, not a matmul."""

    def __init__(self, attn, norm1, norm2, fc1_w, fc1_b, fc2_w, fc2_b, ls1, ls2, cfg):
        self.attn = attn
        self.norm1_w, self.norm1_b = norm1
        self.norm2_w, self.norm2_b = norm2
        self.fc1_w, self.fc1_b = fc1_w, fc1_b
        self.fc2_w, self.fc2_b = fc2_w, fc2_b
        self.ls1, self.ls2 = ls1, ls2  # ttnn tensors, shape (1,1,1,hidden)
        self.cfg = cfg

    @classmethod
    def from_state_dict(cls, state_dict, *, layer_idx: int, cfg: Dinov2Config, device):
        prefix = f"encoder.layer.{layer_idx}"
        attn = SelfAttention.from_state_dict(state_dict, prefix=prefix, cfg=cfg, device=device)
        norm1 = _torch_norm_to_ttnn(state_dict[f"{prefix}.norm1.weight"], state_dict[f"{prefix}.norm1.bias"], device)
        norm2 = _torch_norm_to_ttnn(state_dict[f"{prefix}.norm2.weight"], state_dict[f"{prefix}.norm2.bias"], device)
        fc1_w, fc1_b = _torch_linear_to_ttnn(
            state_dict[f"{prefix}.mlp.fc1.weight"], state_dict[f"{prefix}.mlp.fc1.bias"], device, ttnn.bfloat16
        )
        fc2_w, fc2_b = _torch_linear_to_ttnn(
            state_dict[f"{prefix}.mlp.fc2.weight"], state_dict[f"{prefix}.mlp.fc2.bias"], device, ttnn.bfloat16
        )
        ls1 = ttnn.from_torch(
            state_dict[f"{prefix}.layer_scale1.lambda1"].reshape(1, 1, 1, -1),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
        )
        ls2 = ttnn.from_torch(
            state_dict[f"{prefix}.layer_scale2.lambda1"].reshape(1, 1, 1, -1),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
        )
        return cls(attn, norm1, norm2, fc1_w, fc1_b, fc2_w, fc2_b, ls1, ls2, cfg)

    def forward(self, x: "ttnn.Tensor", batch: int, seq_len: int) -> "ttnn.Tensor":
        eps = self.cfg.layer_norm_eps
        residual = x
        h = ttnn.layer_norm(
            x, weight=self.norm1_w, bias=self.norm1_b, epsilon=eps, compute_kernel_config=_hifi_compute_kernel_config()
        )
        h = self.attn(h, batch, seq_len)
        h = h * self.ls1
        x = residual + h

        residual = x
        h = ttnn.layer_norm(
            x, weight=self.norm2_w, bias=self.norm2_b, epsilon=eps, compute_kernel_config=_hifi_compute_kernel_config()
        )
        h = ttnn.typecast(h, ttnn.bfloat16)
        h = ttnn.linear(h, self.fc1_w, bias=self.fc1_b, compute_kernel_config=_hifi_compute_kernel_config())
        h = ttnn.gelu(h)
        h = ttnn.linear(h, self.fc2_w, bias=self.fc2_b, compute_kernel_config=_hifi_compute_kernel_config())
        h = ttnn.typecast(h, x.dtype)
        h = h * self.ls2
        x = residual + h
        return x


class Model(LightweightModule):
    """Full DINOv2 encoder: Embeddings -> 24x EncoderLayer -> final LayerNorm. Matches
    reference `Dinov2Model.forward` exactly (the reference's `pooled_output` is just
    `sequence_output[:, 0, :]`, the CLS token -- no separate pooler weights to port)."""

    def __init__(self, embeddings: Embeddings, layers: list, final_norm, cfg: Dinov2Config):
        self.embeddings = embeddings
        self.layers = layers
        self.final_norm_w, self.final_norm_b = final_norm
        self.cfg = cfg

    @classmethod
    def from_state_dict(cls, state_dict, *, cfg: Dinov2Config, device):
        embeddings = Embeddings.from_state_dict(state_dict, cfg=cfg, device=device)
        layers = [
            EncoderLayer.from_state_dict(state_dict, layer_idx=i, cfg=cfg, device=device)
            for i in range(cfg.num_layers)
        ]
        final_norm = _torch_norm_to_ttnn(state_dict["layernorm.weight"], state_dict["layernorm.bias"], device)
        return cls(embeddings, layers, final_norm, cfg)

    def forward(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        seq_len = self.cfg.grid_size * self.cfg.grid_size + self.cfg.num_prefix_tokens
        x = self.embeddings(pixel_values)
        for layer in self.layers:
            x = layer(x, batch=pixel_values.shape[0], seq_len=seq_len)
        return ttnn.layer_norm(
            x, weight=self.final_norm_w, bias=self.final_norm_b, epsilon=self.cfg.layer_norm_eps,
            compute_kernel_config=_hifi_compute_kernel_config(),
        )


def timm_vit_reg_state_dict_to_hf_style(state_dict: dict, *, num_layers: int) -> dict:
    """Rename a timm `VisionTransformer` state dict (flat `blocks.N.*`, fused
    `attn.qkv`) into this module's HF-style keys (`embeddings.*`, `encoder.layer.N.*`,
    separate query/key/value), so `Embeddings`/`SelfAttention`/`EncoderLayer`/`Model`
    stay checkpoint-agnostic and don't need a parallel timm-flavored implementation.

    The qkv split-then-immediately-reconcatenated-by-SelfAttention roundtrip is
    wasteful but exactly correct and zero-risk against already-validated code --
    optimizing it away is a ttm-optimize question for later, not a correctness one."""
    out = {}
    out["embeddings.cls_token"] = state_dict["cls_token"]
    if "reg_token" in state_dict:
        out["embeddings.register_tokens"] = state_dict["reg_token"]
    out["embeddings.position_embeddings"] = state_dict["pos_embed"]
    out["embeddings.patch_embeddings.projection.weight"] = state_dict["patch_embed.proj.weight"]
    out["embeddings.patch_embeddings.projection.bias"] = state_dict["patch_embed.proj.bias"]

    for i in range(num_layers):
        src, dst = f"blocks.{i}", f"encoder.layer.{i}"
        out[f"{dst}.norm1.weight"] = state_dict[f"{src}.norm1.weight"]
        out[f"{dst}.norm1.bias"] = state_dict[f"{src}.norm1.bias"]
        qkv_w = state_dict[f"{src}.attn.qkv.weight"]  # (3*hidden, hidden), fused in q,k,v order
        qkv_b = state_dict[f"{src}.attn.qkv.bias"]
        hidden = qkv_w.shape[1]
        q_w, k_w, v_w = qkv_w.split(hidden, dim=0)
        q_b, k_b, v_b = qkv_b.split(hidden, dim=0)
        out[f"{dst}.attention.attention.query.weight"] = q_w
        out[f"{dst}.attention.attention.key.weight"] = k_w
        out[f"{dst}.attention.attention.value.weight"] = v_w
        out[f"{dst}.attention.attention.query.bias"] = q_b
        out[f"{dst}.attention.attention.key.bias"] = k_b
        out[f"{dst}.attention.attention.value.bias"] = v_b
        out[f"{dst}.attention.output.dense.weight"] = state_dict[f"{src}.attn.proj.weight"]
        out[f"{dst}.attention.output.dense.bias"] = state_dict[f"{src}.attn.proj.bias"]
        out[f"{dst}.layer_scale1.lambda1"] = state_dict[f"{src}.ls1.gamma"]
        out[f"{dst}.layer_scale2.lambda1"] = state_dict[f"{src}.ls2.gamma"]
        out[f"{dst}.norm2.weight"] = state_dict[f"{src}.norm2.weight"]
        out[f"{dst}.norm2.bias"] = state_dict[f"{src}.norm2.bias"]
        out[f"{dst}.mlp.fc1.weight"] = state_dict[f"{src}.mlp.fc1.weight"]
        out[f"{dst}.mlp.fc1.bias"] = state_dict[f"{src}.mlp.fc1.bias"]
        out[f"{dst}.mlp.fc2.weight"] = state_dict[f"{src}.mlp.fc2.weight"]
        out[f"{dst}.mlp.fc2.bias"] = state_dict[f"{src}.mlp.fc2.bias"]

    out["layernorm.weight"] = state_dict["norm.weight"]
    out["layernorm.bias"] = state_dict["norm.bias"]
    return out
