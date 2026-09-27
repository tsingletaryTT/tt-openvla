# SPDX-License-Identifier: MIT
"""Regression test for the vision-tower weight bug fixed in tt-openvla-serving 0.2.0.

The bug: the served path (`demo_grounded_check.get_vision_projector_weights`, which
`gradio_app/backends.py`'s TTNNBackend reuses) built DINOv2 and SigLIP from timm's GENERIC
pretrained checkpoints instead of openvla-7b's own fine-tuned
`vision_backbone.featurizer.*` / `vision_backbone.fused_featurizer.*` tensors. The old
tower tests could not catch it: their reference loaded the same generic timm weights, so
they compared the wrong towers against the wrong towers and passed at PCC 0.996-0.999.

This test puts the guard at the layer that failed -- WHICH WEIGHTS the served loader hands
to TTNN -- and then measures the result:

  1. Wiring (host-only, instant): the state dicts `get_vision_projector_weights()` returns
     must be bit-identical to the checkpoint's own tensors, read straight from the
     safetensors shards here (not through the loader under test).
  2. Numerics (device): each TTNN tower, built from those state dicts, vs a CPU timm
     reference with `pretrained=False` + the checkpoint's own tensors, on a real image
     through openvla-7b's own processor. Per-tower PCC >= 0.995 (this repo's vision bar),
     plus the fused backbone -> projector output.

Seen failing against the pre-0.2.0 loader (git main @ 50fdee5, same test file): step 1
fails at the first compared tensor, and step 2 measures per-tower PCC well below the bar
-- see the repo's CLAUDE.md log for the numbers. The CPU reference itself was checked
against upstream's real modeling_prismatic.py (timm 0.9.16) tower outputs; see FIX notes
in the CLAUDE.md log.

Needs a device: run under a gozer lease, e.g.
  gozer run --chips 2 --who "claude:openvla-test" --reason "vision tower test" -- \\
      python tt/test_vision_towers_openvla_weights.py
"""

import os
import sys
from pathlib import Path

import torch

TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise RuntimeError("Set TT_METAL_HOME to a tt-metal checkout (or the ttnn package dir).")
sys.path.insert(0, TT_METAL_HOME)

import ttnn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from tt import openvla_weights  # noqa: E402
from tt.demo_grounded_check import get_vision_projector_weights  # noqa: E402  (the code under test)
from tt.functional_encoder import Model as Dinov2Model  # noqa: E402
from tt.functional_projector import Projector  # noqa: E402
from tt.functional_siglip import Model as SiglipModel  # noqa: E402
from tt.functional_vision_backbone import VisionBackbone  # noqa: E402

PCC_BAR = 0.995
EXAMPLE_IMAGE = REPO_ROOT / "gradio_app" / "assets" / "example.jpg"


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def checkpoint_tensors(prefix: str) -> dict:
    """Read every tensor under `prefix` DIRECTLY from the shards (float32), independent of
    the loader being tested. LayerScale is renamed scale_factor -> gamma, timm's naming."""
    from safetensors import safe_open

    out = {}
    for path in openvla_weights.shard_paths():
        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    out[k[len(prefix):].replace(".scale_factor", ".gamma")] = f.get_tensor(k).float()
    assert out, f"no {prefix}* tensors in the checkpoint"
    return out


def check_wiring(dinov2_sd, siglip_sd, proj_sd, ck_dino, ck_sig, ck_proj):
    """Step 1: the served loader hands TTNN the checkpoint's own tensors, bit for bit."""
    pairs = [
        ("dinov2 patch_embed", dinov2_sd["embeddings.patch_embeddings.projection.weight"], ck_dino["patch_embed.proj.weight"]),
        ("dinov2 reg_token", dinov2_sd["embeddings.register_tokens"], ck_dino["reg_token"]),
        ("dinov2 pos_embed", dinov2_sd["embeddings.position_embeddings"], ck_dino["pos_embed"]),
        ("siglip patch_embed", siglip_sd["embeddings.patch_embedding.weight"], ck_sig["patch_embed.proj.weight"]),
        ("siglip pos_embed", siglip_sd["embeddings.position_embedding.weight"], ck_sig["pos_embed"][0]),
        ("projector fc1", proj_sd["fc1.weight"], ck_proj["fc1.weight"]),
    ]
    for i in (0, 12, 22):
        pairs.append((f"dinov2 block {i} fc1", dinov2_sd[f"encoder.layer.{i}.mlp.fc1.weight"], ck_dino[f"blocks.{i}.mlp.fc1.weight"]))
        pairs.append((f"dinov2 block {i} ls1", dinov2_sd[f"encoder.layer.{i}.layer_scale1.lambda1"], ck_dino[f"blocks.{i}.ls1.gamma"]))
        q_ck = ck_sig[f"blocks.{i}.attn.qkv.weight"].split(ck_sig[f"blocks.{i}.attn.qkv.weight"].shape[1], dim=0)[0]
        pairs.append((f"siglip block {i} q_proj", siglip_sd[f"encoder.layers.{i}.self_attn.q_proj.weight"], q_ck))
        pairs.append((f"siglip block {i} fc1", siglip_sd[f"encoder.layers.{i}.mlp.fc1.weight"], ck_sig[f"blocks.{i}.mlp.fc1.weight"]))
    mismatches = []
    for name, got, want in pairs:
        got = got.float()
        rel = ((got - want).norm() / want.norm()).item()
        print(f"  wiring {name:24s} rel_l2 vs checkpoint = {rel:.3e}")
        if not torch.equal(got, want):
            mismatches.append(f"{name} (rel_l2={rel:.3f})")
    # Returned rather than asserted here so step 2 still runs and reports PCC on a broken
    # loader too -- both halves of the evidence, then one combined verdict in main().
    return mismatches


def reference_towers(ck_dino, ck_sig):
    """CPU timm towers: architecture only (pretrained=False), checkpoint weights, strict."""
    import timm

    dino = timm.create_model(openvla_weights.DINOV2_TIMM_ID, pretrained=False, num_classes=0, img_size=224)
    dino.load_state_dict(ck_dino, strict=True)
    sig = timm.create_model(openvla_weights.SIGLIP_TIMM_ID, pretrained=False, num_classes=0, img_size=224)
    sig.load_state_dict(ck_sig, strict=True)
    return dino.eval(), sig.eval()


def main():
    from PIL import Image

    print("Reading openvla-7b's own tower + projector tensors straight from the shards...")
    ck_dino = checkpoint_tensors(openvla_weights.DINOV2_PREFIX)
    ck_sig = checkpoint_tensors(openvla_weights.SIGLIP_PREFIX)
    ck_proj = checkpoint_tensors(openvla_weights.PROJECTOR_PREFIX)

    # Print WHICH file is under test: a regular `tt` package installed in site-packages (the
    # serving wheel) outranks a checkout's tt/ on sys.path if the checkout's tt/ has no
    # __init__.py -- that exact shadowing once made this test pass against the wrong code.
    import tt.demo_grounded_check as _under_test

    print(f"Code under test: {_under_test.__file__}")
    print("Loading the served path's weights (get_vision_projector_weights)...")
    dcfg, dinov2_sd, scfg, siglip_sd, pcfg, proj_sd = get_vision_projector_weights()

    print("Step 1: wiring")
    mismatches = check_wiring(dinov2_sd, siglip_sd, proj_sd, ck_dino, ck_sig, ck_proj)

    print("Step 2: numerics. Real image through openvla-7b's processor...")
    processor = openvla_weights.load_processor()
    image = Image.open(EXAMPLE_IMAGE).convert("RGB")
    pixel_values = processor("In: What action should the robot take to open the drawer?\nOut:", image)["pixel_values"].float()
    img, img_fused = torch.split(pixel_values, [3, 3], dim=1)

    dino_ref, sig_ref = reference_towers(ck_dino, ck_sig)
    with torch.no_grad():
        ref_dino = dino_ref.get_intermediate_layers(img, n={len(dino_ref.blocks) - 2})[0]
        ref_sig = sig_ref.get_intermediate_layers(img_fused, n={len(sig_ref.blocks) - 2})[0]
        fused = torch.cat([ref_dino, ref_sig], dim=2)
        x = torch.nn.functional.gelu(torch.nn.functional.linear(fused, ck_proj["fc1.weight"], ck_proj["fc1.bias"]))
        x = torch.nn.functional.gelu(torch.nn.functional.linear(x, ck_proj["fc2.weight"], ck_proj["fc2.bias"]))
        ref_proj = torch.nn.functional.linear(x, ck_proj["fc3.weight"], ck_proj["fc3.bias"])

    device = ttnn.open_device(device_id=0)
    try:
        dinov2 = Dinov2Model.from_state_dict(dinov2_sd, cfg=dcfg, device=device)
        siglip = SiglipModel.from_state_dict(siglip_sd, cfg=scfg, device=device)
        backbone = VisionBackbone.from_models(dinov2, siglip)

        n_patches = dcfg.grid_size * dcfg.grid_size
        d_out = backbone._run_all_but_last_block(dinov2, img, n_patches + dcfg.num_prefix_tokens)
        tt_dino = ttnn.to_torch(d_out).float().reshape(1, -1, dcfg.hidden_size)[:, dcfg.num_prefix_tokens:, :]
        s_out = backbone._run_all_but_last_block(siglip, img_fused, scfg.seq_len)
        tt_sig = ttnn.to_torch(s_out).float().reshape(1, scfg.seq_len, scfg.hidden_size)

        projector = Projector.from_state_dict(proj_sd, cfg=pcfg, device=device)
        tt_proj = ttnn.to_torch(projector(backbone.forward(pixel_values))).float().reshape(ref_proj.shape)

        results = {
            "dinov2 (featurizer)": pcc(tt_dino, ref_dino),
            "siglip (fused_featurizer)": pcc(tt_sig, ref_sig),
            "backbone+projector": pcc(tt_proj, ref_proj),
        }
    finally:
        ttnn.close_device(device)

    for name, v in results.items():
        print(f"  PCC {name:26s} = {v:.6f}")
    failed = {k: round(v, 6) for k, v in results.items() if v < PCC_BAR}
    assert not mismatches, (
        "served loader is NOT handing TTNN openvla-7b's own tensors: " + "; ".join(mismatches)
        + (f" | and PCC below {PCC_BAR}: {failed}" if failed else "")
    )
    assert not failed, f"PCC below {PCC_BAR}: {failed}"
    print("PASS")
    return results


def test_vision_towers_use_openvla_weights():
    main()


if __name__ == "__main__":
    main()
