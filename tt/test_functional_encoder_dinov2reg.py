# SPDX-License-Identifier: MIT
"""Full-model correctness check for OpenVLA's ACTUAL DINOv2 tower: the register-token
variant (`vit_large_patch14_reg4_dinov2.lvd142m`), against a real `timm` reference --
not `transformers.Dinov2Model`/`Dinov2WithRegistersModel`, since OpenVLA's own
`PrismaticVisionBackbone` loads this checkpoint via `timm.create_model(...)` directly,
and functional_encoder.py's module docstring documents why HF's own reimplementation
of the register variant does NOT compute the same forward function (a nonzero learned
CLS position embedding that timm's `no_embed_class` scheme doesn't have).

timm resizes `pos_embed` to the requested `img_size` itself at model-construction time
(confirmed: shape (1,256,1024) when built with img_size=224, vs (1,1369,1024) at its
518px pretrained default) -- so `Dinov2Config.pretrained_image_size` is set to 224
here too, making this module's own `interpolate_position_embeddings` a no-op
passthrough on the already-resized weights rather than a second, redundant resize."""

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
from tt.functional_encoder import Dinov2Config, Model, timm_vit_reg_state_dict_to_hf_style  # noqa: E402


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    import timm

    cfg = Dinov2Config(num_register_tokens=4, no_embed_class=True, pretrained_image_size=224)

    print("Loading real vit_large_patch14_reg4_dinov2.lvd142m checkpoint (timm)...")
    ref_model = timm.create_model(
        "vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=cfg.image_size
    )
    ref_model.eval()
    assert ref_model.no_embed_class and ref_model.num_prefix_tokens == 1 + cfg.num_register_tokens
    sd = timm_vit_reg_state_dict_to_hf_style(
        {k: v.float() for k, v in ref_model.state_dict().items()}, num_layers=cfg.num_layers
    )

    torch.manual_seed(0)
    pixel_values = torch.randn(1, cfg.in_chans, cfg.image_size, cfg.image_size)

    print("Running reference (timm forward_features, full 24-layer forward)...")
    with torch.no_grad():
        ref_out = ref_model.forward_features(pixel_values)

    print("Running TTNN port (full 24-layer forward)...")
    device = ttnn.open_device(device_id=0)
    try:
        tt_model = Model.from_state_dict(sd, cfg=cfg, device=device)
        tt_out = tt_model(pixel_values)

        full_pcc = pcc(ttnn.to_torch(tt_out).reshape(ref_out.shape), ref_out)
        print(f"full 24-layer model output PCC (incl. CLS + 4 register tokens): {full_pcc:.6f}")
        assert full_pcc >= 0.995, f"full model PCC {full_pcc} < 0.995"

        patches_pcc = pcc(
            ttnn.to_torch(tt_out).reshape(ref_out.shape)[:, cfg.num_prefix_tokens :],
            ref_out[:, cfg.num_prefix_tokens :],
        )
        print(f"patch-tokens-only PCC (what PrismaticVisionBackbone actually consumes): {patches_pcc:.6f}")
        assert patches_pcc >= 0.995, f"patch-tokens PCC {patches_pcc} < 0.995"
        print("PASS")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
