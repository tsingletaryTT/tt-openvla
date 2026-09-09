# SPDX-License-Identifier: MIT
"""Correctness check for the fused vision backbone: TTNN VisionBackbone (DINOv2
register-variant + SigLIP, each truncated to all-but-last-block, concatenated
per-patch) against a real reference built directly from `modeling_prismatic.py`'s
`PrismaticVisionBackbone.forward` logic, using the two real, correctly-tagged timm
checkpoints -- not the full HF `PrismaticForConditionalGeneration` wrapper, which would
require LLaMA weights irrelevant to this component."""

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
from tt.functional_encoder import Dinov2Config, Model as Dinov2Model, timm_vit_reg_state_dict_to_hf_style  # noqa: E402
from tt.functional_siglip import Model as SiglipModel, SiglipConfig  # noqa: E402
from tt.functional_vision_backbone import VisionBackbone  # noqa: E402


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def reference_vision_backbone(dinov2_ref, siglip_ref, pixel_values: torch.Tensor) -> torch.Tensor:
    """Direct reproduction of modeling_prismatic.py's PrismaticVisionBackbone.forward
    (channel split -> per-tower get_intermediate_layers(n={num_blocks-2}) ->
    per-patch concat), using the two real timm models as the two "featurizers"."""
    img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
    with torch.no_grad():
        patches = dinov2_ref.get_intermediate_layers(img, n={len(dinov2_ref.blocks) - 2})[0]
        patches_fused = siglip_ref.get_intermediate_layers(img_fused, n={len(siglip_ref.blocks) - 2})[0]
    return torch.cat([patches, patches_fused], dim=2)


def main():
    import timm

    dcfg = Dinov2Config(num_register_tokens=4, no_embed_class=True, pretrained_image_size=224)
    scfg = SiglipConfig()

    print("Loading real timm checkpoints (dinov2-reg4 + siglip)...")
    dinov2_ref = timm.create_model(
        "vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=dcfg.image_size
    )
    dinov2_ref.eval()
    siglip_ref = timm.create_model(
        "vit_so400m_patch14_siglip_224.webli", pretrained=True, num_classes=0, img_size=scfg.image_size
    )
    siglip_ref.eval()

    dinov2_sd = timm_vit_reg_state_dict_to_hf_style(
        {k: v.float() for k, v in dinov2_ref.state_dict().items()}, num_layers=dcfg.num_layers
    )
    # timm's SigLIP state dict uses flat blocks.N.* / fused attn.qkv naming (like
    # DINOv2's), not HF's encoder.layers.N.self_attn.{q,k,v}_proj -- convert directly.
    raw = {k: v.float() for k, v in siglip_ref.state_dict().items()}
    siglip_sd = {
        "embeddings.patch_embedding.weight": raw["patch_embed.proj.weight"],
        "embeddings.patch_embedding.bias": raw["patch_embed.proj.bias"],
        "embeddings.position_embedding.weight": raw["pos_embed"][0],
        "post_layernorm.weight": raw["norm.weight"],
        "post_layernorm.bias": raw["norm.bias"],
    }
    for i in range(scfg.num_layers):
        src, dst = f"blocks.{i}", f"encoder.layers.{i}"
        qkv_w, qkv_b = raw[f"{src}.attn.qkv.weight"], raw[f"{src}.attn.qkv.bias"]
        hidden = qkv_w.shape[1]
        q_w, k_w, v_w = qkv_w.split(hidden, dim=0)
        q_b, k_b, v_b = qkv_b.split(hidden, dim=0)
        siglip_sd[f"{dst}.self_attn.q_proj.weight"] = q_w
        siglip_sd[f"{dst}.self_attn.k_proj.weight"] = k_w
        siglip_sd[f"{dst}.self_attn.v_proj.weight"] = v_w
        siglip_sd[f"{dst}.self_attn.q_proj.bias"] = q_b
        siglip_sd[f"{dst}.self_attn.k_proj.bias"] = k_b
        siglip_sd[f"{dst}.self_attn.v_proj.bias"] = v_b
        siglip_sd[f"{dst}.self_attn.out_proj.weight"] = raw[f"{src}.attn.proj.weight"]
        siglip_sd[f"{dst}.self_attn.out_proj.bias"] = raw[f"{src}.attn.proj.bias"]
        siglip_sd[f"{dst}.layer_norm1.weight"] = raw[f"{src}.norm1.weight"]
        siglip_sd[f"{dst}.layer_norm1.bias"] = raw[f"{src}.norm1.bias"]
        siglip_sd[f"{dst}.layer_norm2.weight"] = raw[f"{src}.norm2.weight"]
        siglip_sd[f"{dst}.layer_norm2.bias"] = raw[f"{src}.norm2.bias"]
        siglip_sd[f"{dst}.mlp.fc1.weight"] = raw[f"{src}.mlp.fc1.weight"]
        siglip_sd[f"{dst}.mlp.fc1.bias"] = raw[f"{src}.mlp.fc1.bias"]
        siglip_sd[f"{dst}.mlp.fc2.weight"] = raw[f"{src}.mlp.fc2.weight"]
        siglip_sd[f"{dst}.mlp.fc2.bias"] = raw[f"{src}.mlp.fc2.bias"]

    torch.manual_seed(0)
    pixel_values = torch.randn(1, 6, 224, 224)  # 6-channel: [dinov2-normed img, siglip-normed img]

    print("Running reference (real PrismaticVisionBackbone.forward logic)...")
    ref_out = reference_vision_backbone(dinov2_ref, siglip_ref, pixel_values)
    print(f"reference fused output shape: {tuple(ref_out.shape)}")

    print("Running TTNN VisionBackbone...")
    device = ttnn.open_device(device_id=0)
    try:
        dinov2_tt = Dinov2Model.from_state_dict(dinov2_sd, cfg=dcfg, device=device)
        siglip_tt = SiglipModel.from_state_dict(siglip_sd, cfg=scfg, device=device)
        backbone = VisionBackbone.from_models(dinov2_tt, siglip_tt)
        tt_out = backbone.forward(pixel_values)

        fused_pcc = pcc(ttnn.to_torch(tt_out).reshape(ref_out.shape), ref_out)
        print(f"fused vision backbone output PCC: {fused_pcc:.6f}")
        assert fused_pcc >= 0.995, f"fused backbone PCC {fused_pcc} < 0.995"
        print("PASS")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
