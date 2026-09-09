# SPDX-License-Identifier: MIT
"""Correctness check for the Projector MLP against its real weights from
openvla/openvla-7b (shard 1 of 3 -- projector.* lives entirely in
model-00001-of-00003.safetensors, confirmed via the checkpoint's own
model.safetensors.index.json before downloading the ~7GB shard)."""

import glob
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise RuntimeError("Set TT_METAL_HOME to a tt-metal checkout.")
sys.path.insert(0, TT_METAL_HOME)

import ttnn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tt.functional_projector import Projector, ProjectorConfig  # noqa: E402


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def reference_projector(sd: dict, img_patches: torch.Tensor) -> torch.Tensor:
    """Direct reproduction of modeling_prismatic.py's PrismaticProjector.forward,
    use_fused_vision_backbone=True branch."""
    x = F.linear(img_patches, sd["fc1.weight"], sd["fc1.bias"])
    x = F.gelu(x)
    x = F.linear(x, sd["fc2.weight"], sd["fc2.bias"])
    x = F.gelu(x)
    x = F.linear(x, sd["fc3.weight"], sd["fc3.bias"])
    return x


def main():
    from safetensors import safe_open

    cfg = ProjectorConfig()
    assert cfg.initial_projection_dim == 8704, cfg.initial_projection_dim

    print("Loading real openvla/openvla-7b projector weights...")
    shard_path = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/*/model-00001-of-00003.safetensors"
        )
    )[0]
    sd = {}
    with safe_open(shard_path, framework="pt") as f:
        for k in f.keys():
            if k.startswith("projector."):
                sd[k[len("projector."):]] = f.get_tensor(k).float()

    assert sd["fc1.weight"].shape == (cfg.initial_projection_dim, cfg.vision_dim)
    assert sd["fc2.weight"].shape == (cfg.llm_dim, cfg.initial_projection_dim)
    assert sd["fc3.weight"].shape == (cfg.llm_dim, cfg.llm_dim)

    torch.manual_seed(0)
    img_patches = torch.randn(1, 256, cfg.vision_dim)  # matches VisionBackbone's real output shape

    print("Running reference (real PrismaticProjector.forward logic)...")
    with torch.no_grad():
        ref_out = reference_projector(sd, img_patches)
    print(f"reference output shape: {tuple(ref_out.shape)}")

    print("Running TTNN Projector...")
    device = ttnn.open_device(device_id=0)
    try:
        tt_model = Projector.from_state_dict(sd, cfg=cfg, device=device)
        tt_out = tt_model(ttnn.from_torch(img_patches, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device))

        out_pcc = pcc(ttnn.to_torch(tt_out).reshape(ref_out.shape), ref_out)
        print(f"projector output PCC: {out_pcc:.6f}")
        assert out_pcc >= 0.995, f"projector PCC {out_pcc} < 0.995"
        print("PASS")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
