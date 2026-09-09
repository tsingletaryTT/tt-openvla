# SPDX-License-Identifier: MIT
"""TTNN port of OpenVLA's `PrismaticVisionBackbone`: fuses the DINOv2 (register-token
variant, functional_encoder.py) and SigLIP (functional_siglip.py) towers, both already
independently validated against their own real checkpoints up to their final layer.

Reference (`modeling_prismatic.py`'s `PrismaticVisionBackbone.forward`, confirmed by
direct source reading -- see functional_encoder.py's and functional_siglip.py's module
docstrings for the checkpoint-provenance work this depends on):
  1. The 224x224x3 input image is preprocessed twice (once per tower's own
     normalization stats) and stacked into a 6-channel tensor upstream of this module;
     `torch.split(pixel_values, [3, 3], dim=1)` recovers the two 3-channel images.
  2. Each tower runs through all but its OWN LAST transformer block -- timm's
     `get_intermediate_layers(x, n={len(blocks) - 2})`, which (see this module's own
     correctness test for the exact indexing this reproduces) captures the hidden
     state right after block index `num_layers - 2`, i.e. `num_layers - 1` blocks
     total, never applying the final norm and never running the true last block.
  3. Prefix tokens (DINOv2's CLS + 4 register tokens; SigLIP has none) are dropped,
     leaving only the per-patch tokens -- 256 for each tower at 224px/patch14.
  4. The two towers' patch tokens are concatenated along the feature dim:
     `torch.cat([dinov2_patches, siglip_patches], dim=2)` -> (B, 256, 1024+1152=2176).

This module stops at that fused (B, 256, 2176) tensor -- the projector MLP into
LLaMA's embedding space (`PrismaticProjector`) is a separate, not-yet-ported step."""

from __future__ import annotations

import torch

import ttnn
from tt.functional_encoder import Model as Dinov2Model
from tt.functional_siglip import Model as SiglipModel


class VisionBackbone:
    def __init__(self, dinov2: Dinov2Model, siglip: SiglipModel):
        assert dinov2.cfg.grid_size == siglip.cfg.grid_size, (
            "both towers must produce the same patch-grid size to be concatenated per-patch "
            f"(got dinov2={dinov2.cfg.grid_size}, siglip={siglip.cfg.grid_size})"
        )
        self.dinov2 = dinov2
        self.siglip = siglip

    @classmethod
    def from_models(cls, dinov2: Dinov2Model, siglip: SiglipModel) -> "VisionBackbone":
        return cls(dinov2, siglip)

    def _run_all_but_last_block(self, model, pixel_values: torch.Tensor, seq_len: int) -> "ttnn.Tensor":
        """Shared by both towers: embeddings -> all layers except the last one, no
        final norm. `model.layers` is a plain list (populated by each Model's own
        `from_state_dict`), so this reuses the already-validated per-layer forward
        exactly as-is -- no new method needed on either Model class."""
        x = model.embeddings(pixel_values)
        for layer in model.layers[:-1]:
            x = layer(x, batch=pixel_values.shape[0], seq_len=seq_len)
        return x

    def forward(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        B = pixel_values.shape[0]
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)

        dcfg = self.dinov2.cfg
        num_patches = dcfg.grid_size * dcfg.grid_size
        d_seq_len = num_patches + dcfg.num_prefix_tokens
        dinov2_out = self._run_all_but_last_block(self.dinov2, img, d_seq_len)
        # Rank-agnostic slice: ttnn may report an extra tile-padding dim beyond the
        # logical (B, seq, hidden), but the sequence axis is always second-to-last.
        shape = list(dinov2_out.shape)
        seq_dim = len(shape) - 2
        begins = [0] * len(shape)
        begins[seq_dim] = dcfg.num_prefix_tokens
        dinov2_patches = ttnn.slice(dinov2_out, begins, shape)
        # Force both operands to a known, matching rank before concatenating --
        # avoids relying on however ttnn happens to report padded rank internally.
        dinov2_patches = ttnn.reshape(dinov2_patches, (B, num_patches, dcfg.hidden_size))

        scfg = self.siglip.cfg
        siglip_patches = self._run_all_but_last_block(self.siglip, img_fused, scfg.seq_len)
        siglip_patches = ttnn.reshape(siglip_patches, (B, scfg.seq_len, scfg.hidden_size))

        return ttnn.concat([dinov2_patches, siglip_patches], dim=2)
