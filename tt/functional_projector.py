# SPDX-License-Identifier: MIT
"""TTNN port of OpenVLA's `PrismaticProjector` (fused-backbone variant): the 3-layer
GELU MLP that projects the fused vision backbone's per-patch tokens
(functional_vision_backbone.py's output, 2176-dim) into LLaMA-2-7B's embedding space
(4096-dim) before they're fed into the language model alongside the text tokens.

No attention, no norm layers -- just fc1 -> GELU -> fc2 -> GELU -> fc3. Simpler than
either vision tower, but its weights are a downstream fine-tuned artifact that only
exists in the real `openvla/openvla-7b` checkpoint (not derivable from a generic
vision or language-model checkpoint), so this is the first component in this repo
validated against openvla-7b itself rather than one of its two frozen vision towers.

Reference: modeling_prismatic.py's `PrismaticProjector.forward` (the
`use_fused_vision_backbone=True` branch -- OpenVLA always sets this for its actual
dinosiglip-vit-so-224px backbone). Checkpoint: `openvla/openvla-7b`'s own
`projector.{fc1,fc2,fc3}.{weight,bias}` (shard 1 of 3 in its safetensors index).
`initial_projection_dim = 4 * vision_dim` matches the reference exactly, not a
guessed round number -- confirmed from the real fc1 weight's output shape."""

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
class ProjectorConfig:
    vision_dim: int = 2176  # DINOv2 (1024) + SigLIP (1152) per-patch feature concat
    llm_dim: int = 4096  # LLaMA-2-7B hidden_size

    @property
    def initial_projection_dim(self) -> int:
        return 4 * self.vision_dim


def _torch_linear_to_ttnn(weight: torch.Tensor, bias: torch.Tensor, device, dtype=ttnn.float32):
    w = ttnn.from_torch(weight.t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(bias.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return w, b


class Projector(LightweightModule):
    def __init__(self, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b, cfg: ProjectorConfig):
        self.fc1_w, self.fc1_b = fc1_w, fc1_b
        self.fc2_w, self.fc2_b = fc2_w, fc2_b
        self.fc3_w, self.fc3_b = fc3_w, fc3_b
        self.cfg = cfg

    @classmethod
    def from_state_dict(cls, state_dict, *, cfg: ProjectorConfig, device):
        fc1_w, fc1_b = _torch_linear_to_ttnn(state_dict["fc1.weight"], state_dict["fc1.bias"], device, ttnn.bfloat16)
        fc2_w, fc2_b = _torch_linear_to_ttnn(state_dict["fc2.weight"], state_dict["fc2.bias"], device, ttnn.bfloat16)
        fc3_w, fc3_b = _torch_linear_to_ttnn(state_dict["fc3.weight"], state_dict["fc3.bias"], device, ttnn.bfloat16)
        return cls(fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b, cfg)

    def forward(self, img_patches: "ttnn.Tensor") -> "ttnn.Tensor":
        x = ttnn.typecast(img_patches, ttnn.bfloat16)
        x = ttnn.linear(x, self.fc1_w, bias=self.fc1_b, compute_kernel_config=_hifi_compute_kernel_config())
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.fc2_w, bias=self.fc2_b, compute_kernel_config=_hifi_compute_kernel_config())
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.fc3_w, bias=self.fc3_b, compute_kernel_config=_hifi_compute_kernel_config())
        return x
