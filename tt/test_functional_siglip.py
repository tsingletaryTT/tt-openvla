# SPDX-License-Identifier: MIT
"""Full-model correctness check: all 27 SigLIP layers, TTNN vs the real
`transformers.SiglipVisionModel` reference (up to last_hidden_state, before the
attention-pooling head -- see functional_siglip.py's Model docstring), on the real
`google/siglip-so400m-patch14-224` checkpoint, at its native 224px (no interpolation
needed, unlike DINOv2)."""

import os
import sys
from pathlib import Path

import torch

TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise RuntimeError("Set TT_METAL_HOME to a tt-metal checkout.")
sys.path.insert(0, TT_METAL_HOME)

import ttnn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tt.functional_siglip import Model, SiglipConfig  # noqa: E402


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    from transformers import SiglipVisionModel

    cfg = SiglipConfig()

    print("Loading real google/siglip-so400m-patch14-224 checkpoint...")
    ref_model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-224")
    ref_model.eval()
    # This transformers version (4.52.4) nests everything under `vision_model.`;
    # functional_siglip.py's from_state_dict expects the unprefixed keys (matching
    # SiglipVisionTransformer's own internal naming), so strip the prefix here.
    sd = {
        k[len("vision_model."):]: v.float()
        for k, v in ref_model.state_dict().items()
        if k.startswith("vision_model.")
    }

    torch.manual_seed(0)
    pixel_values = torch.randn(1, cfg.in_chans, cfg.image_size, cfg.image_size)

    print("Running reference (transformers.SiglipVisionModel, full 27-layer forward)...")
    with torch.no_grad():
        ref_out = ref_model.vision_model.encoder(ref_model.vision_model.embeddings(pixel_values)).last_hidden_state
        ref_out = ref_model.vision_model.post_layernorm(ref_out)

    print("Running TTNN port (full 27-layer forward)...")
    device = ttnn.open_device(device_id=0)
    try:
        tt_model = Model.from_state_dict(sd, cfg=cfg, device=device)
        tt_out = tt_model(pixel_values)

        full_pcc = pcc(ttnn.to_torch(tt_out).reshape(ref_out.shape), ref_out)
        print(f"full 27-layer model output PCC: {full_pcc:.6f}")
        assert full_pcc >= 0.995, f"full model PCC {full_pcc} < 0.995"
        print("PASS")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
