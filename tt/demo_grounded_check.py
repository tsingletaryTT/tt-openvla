# SPDX-License-Identifier: MIT
"""End-to-end "Grounded Check": a real image + a real prompt, through this port's own
full pipeline (VisionBackbone -> Projector -> fused embeddings -> LLaMA-2-7B prefill +
7 greedy decode steps -> action detokenization), producing an actual 7-DoF action --
not a synthetic shape/PCC check, a real inference.

Image: `gradio_app/assets/example.jpg` (the COCO "two cats on a couch" photo, the same
picture as `huggingface/cats-image`; a real photograph -- NOT a matched robot scene; the
several real BridgeData V2/Open-X-Embodiment image sources this port tried were either
stale or not easily streamable as plain images in this environment). Documented
honestly: this demo proves the full real-weight pipeline runs and reproduces OpenVLA's
own output for this input, not that the ANSWER is semantically meaningful for opening a
drawer. (Earlier versions pulled the image via `datasets.load_dataset`; the bundled file
removes that network dependency.)

Prompt: upstream deploy.py's exact format,
`"In: What action should the robot take to {instruction.lower()}?\nOut:"`, tokenized by
openvla-7b's own processor, then with token 29871 appended the way upstream
`predict_action` does (tt/openvla_weights.py's `append_empty_token`; versions before
0.2.0 skipped it).

Weights: everything -- both vision towers, projector, LLaMA -- from openvla-7b's own
checkpoint at the pinned revision (tt/openvla_weights.py). Versions before 0.2.0 used
timm's generic pretrained towers; see `get_vision_projector_weights`.

Ground truth: the REAL upstream model, `AutoModelForVision2Seq.from_pretrained(
"openvla/openvla-7b", trust_remote_code=True, revision=<pinned>).predict_action(...)`,
on CPU. Its modeling_prismatic.py hard-requires timm 0.9.x, which this port's serving
stack (timm 1.x) can't import alongside, so it runs in a separate venv; the recorded
token ids live in test_demo_grounded_check.py (REFERENCE_GENERATED_IDS) with the recipe
to regenerate them. Earlier versions compared against this port's own composed PyTorch
pipeline, which shared both the generic-tower bug and the missing-29871 bug with the
TTNN path, so agreement with it proved nothing about agreement with OpenVLA."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise RuntimeError("Set TT_METAL_HOME to a tt-metal checkout.")
sys.path.insert(0, TT_METAL_HOME)

import ttnn  # noqa: E402
from models.tt_transformers.tt.common import Mode  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tt.action_detokenizer import detokenize_actions  # noqa: E402
from tt.functional_encoder import Dinov2Config, Model as Dinov2Model, timm_vit_reg_state_dict_to_hf_style  # noqa: E402
from tt.functional_llama import build_model, build_model_args, prefill_rot_mats  # noqa: E402
from tt.functional_projector import Projector, ProjectorConfig  # noqa: E402
from tt.functional_siglip import Model as SiglipModel, SiglipConfig  # noqa: E402
from tt.llama_checkpoint import get_llama2_config, load_openvla_llama_state_dict  # noqa: E402
from tt.openvla_weights import append_empty_token, load_config, load_processor  # noqa: E402

ACTION_DIM = 7
UNNORM_KEY = "bridge_orig"
EXAMPLE_IMAGE_PATH = Path(__file__).resolve().parents[1] / "gradio_app" / "assets" / "example.jpg"


def get_real_inputs():
    """Real image + real prompt, through OpenVLA's own real processor -- exactly the
    same preprocessing (6-channel dual-normalized pixel_values, tokenized prompt) the
    real model itself uses. See module docstring for the image-source note."""
    from PIL import Image

    processor = load_processor()  # pinned revision -- it runs upstream remote code
    image = Image.open(EXAMPLE_IMAGE_PATH).convert("RGB")
    prompt = "In: What action should the robot take to open the drawer?\nOut:"
    inputs = processor(prompt, image)
    # Same fix-up upstream predict_action applies before generating (see module docstring).
    return inputs["pixel_values"], append_empty_token(inputs["input_ids"]), processor


def get_vision_projector_weights():
    """openvla-7b's OWN fine-tuned vision towers + projector, from its safetensors shards,
    mapped into the state-dict format this port's TTNN modules read.

    Before tt-openvla-serving 0.2.0 this function built both towers from timm's GENERIC
    pretrained checkpoints (`timm.create_model(..., pretrained=True)` for
    `vit_large_patch14_reg4_dinov2.lvd142m` / `vit_so400m_patch14_siglip_224.webli`)
    and only the projector came from openvla-7b. OpenVLA fine-tunes its vision encoder,
    so those are different weights (relative L2 0.20-0.46 per tensor) and every earlier
    vision PCC was measured against the wrong towers. Now all three come from the
    checkpoint's `vision_backbone.featurizer.*`, `vision_backbone.fused_featurizer.*`
    and `projector.*` tensors (see tt/openvla_weights.py), timm is not imported, and
    nothing is downloaded from timm's hub repos."""
    from tt import openvla_weights

    dcfg = Dinov2Config(num_register_tokens=4, no_embed_class=True, pretrained_image_size=224)
    scfg = SiglipConfig()
    pcfg = ProjectorConfig()

    # Both come back in timm's own flat key naming (blocks.N.attn.qkv, ls1.gamma, ...),
    # which the two converters below already understand -- the same converters the
    # generic-timm path used, so only the SOURCE of the tensors changed.
    dinov2_raw, siglip_raw = openvla_weights.load_tower_state_dicts_timm_naming()
    dinov2_sd = timm_vit_reg_state_dict_to_hf_style(dinov2_raw, num_layers=dcfg.num_layers)
    siglip_sd = openvla_weights.timm_siglip_state_dict_to_hf_style(siglip_raw, num_layers=scfg.num_layers)
    proj_sd = openvla_weights.load_projector_state_dict()
    return dcfg, dinov2_sd, scfg, siglip_sd, pcfg, proj_sd


def run_vision_pipeline(device, pixel_values: torch.Tensor) -> torch.Tensor:
    """pixel_values: (1,6,224,224) real dual-normalized image -> (1,256,4096) real
    projected vision tokens, via this port's own already-validated VisionBackbone +
    Projector."""
    from tt.functional_vision_backbone import VisionBackbone

    dcfg, dinov2_sd, scfg, siglip_sd, pcfg, proj_sd = get_vision_projector_weights()
    dinov2 = Dinov2Model.from_state_dict(dinov2_sd, cfg=dcfg, device=device)
    siglip = SiglipModel.from_state_dict(siglip_sd, cfg=scfg, device=device)
    backbone = VisionBackbone.from_models(dinov2, siglip)
    fused_patches_tt = backbone.forward(pixel_values)  # (B,256,2176) ttnn

    projector = Projector.from_state_dict(proj_sd, cfg=pcfg, device=device)
    vision_embeds_tt = projector(fused_patches_tt)  # (B,256,4096) ttnn
    return ttnn.to_torch(vision_embeds_tt).reshape(1, 256, pcfg.llm_dim).float()


def build_fused_embeddings(input_ids: torch.Tensor, vision_embeds: torch.Tensor, tok_embed_w: torch.Tensor) -> torch.Tensor:
    """[BOS] + [256 vision tokens] + [rest of text] -- same splice point
    tt-metal's own open_vla.py uses (insert right after the first, BOS, token)."""
    text_embeds = torch.nn.functional.embedding(input_ids, tok_embed_w)  # (1, T, hidden)
    return torch.cat([text_embeds[:, :1, :], vision_embeds, text_embeds[:, 1:, :]], dim=1)


def main():
    cfg = get_llama2_config()
    print("Fetching real image + prompt via OpenVLA's own processor...")
    pixel_values, input_ids, processor = get_real_inputs()
    print(f"pixel_values {tuple(pixel_values.shape)}, input_ids {tuple(input_ids.shape)}")

    print("Loading real LLaMA-2-7B weights...")
    llama_sd = load_openvla_llama_state_dict()

    print("Opening single device for the vision pipeline (unchanged from its own tests)...")
    vision_device = ttnn.open_device(device_id=0)
    try:
        print("Running vision pipeline (DINOv2 + SigLIP + Projector) on the real image...")
        vision_embeds = run_vision_pipeline(vision_device, pixel_values)
        print(f"vision_embeds {tuple(vision_embeds.shape)}")
    finally:
        ttnn.close_device(vision_device)

    print("Opening 1x2 mesh device (2 chips) for the LLM...")
    # A single chip hits a hard L1 overflow in decode's QKV matmul -- confirmed
    # independent of both weight dtype (bfloat16 vs bfloat8_b, identical overflow byte
    # count) and max_seq_len (1024 vs 384, still identical), meaning it's a fixed
    # per-op weight-sharding cost for this shape/core-grid combination that simply
    # doesn't fit one chip's L1 -- exactly what tt-metal's own models/experimental
    # /openvla notes ("Full BF16 attention ... requires N300 / 2 devices"). A real
    # 1x2 mesh (this board's other chip) needs its own custom inputs_embeds sharded
    # the same way the framework's own Embedding module shards its weight -- checked
    # directly in tt_transformers/tt/embedding.py: `ttnn.ShardTensor2dMesh(mesh_device,
    # dims=(None, 3), mesh_shape=args.cluster_shape)`, i.e. hidden_size (dim 3 of a
    # [1,1,seq,hidden] tensor) is WIDTH-SHARDED across devices, not replicated. This
    # port's own custom fused embeddings need the identical sharding, and the model's
    # output (also sharded along its last dim -- see tt_transformers/tt/common.py's
    # own `ttnn.to_torch(..., mesh_composer=ttnn.ConcatMeshToTensor(mesh_device,
    # dim=-1))` pattern) needs the matching composer to read back.
    ttnn.set_fabric_config(True)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2))
    try:
        fused_embeds = build_fused_embeddings(input_ids, vision_embeds, llama_sd["tok_embeddings.weight"])
        real_seq_len = fused_embeds.shape[1]
        # Only ACTION_DIM=7 more tokens are ever generated past the real content, so
        # pad to the smallest 128-multiple that covers real_seq_len + 7 (rather than
        # get_padded_prefill_len's much larger 1024 default bucket).
        padded_seq_len = ((real_seq_len + ACTION_DIM + 127) // 128) * 128
        print(f"fused sequence: {real_seq_len} real tokens, padded to {padded_seq_len}")

        pad_amount = padded_seq_len - real_seq_len
        padded_embeds = torch.nn.functional.pad(fused_embeds, (0, 0, 0, pad_amount))

        print("Building LLaMA-2-7B Transformer (full 32 layers, real weights)...")
        model_args = build_model_args(mesh_device, max_seq_len=padded_seq_len)
        model = build_model(model_args, mesh_device, dtype=ttnn.bfloat16)

        shard_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=model_args.cluster_shape)
        concat_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=-1)

        last_real_idx = real_seq_len - 1
        tile_aligned = (last_real_idx // 32) * 32
        row_in_tile = last_real_idx - tile_aligned

        embeds_tt = ttnn.from_torch(
            padded_embeds.unsqueeze(0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=mesh_device,
            mesh_mapper=shard_mapper,
        )
        rot_mats = prefill_rot_mats(model, padded_seq_len)

        print(f"Prefill: real content ends at index {last_real_idx}, reading tile at {tile_aligned}...")
        tt_out = model.forward(
            x=embeds_tt, current_pos=None, rot_mats_global=rot_mats, mode=Mode.PREFILL,
            page_table=None, kv_cache=None, get_last_token=tile_aligned,
        )
        logits = ttnn.to_torch(tt_out, mesh_composer=concat_composer).float().reshape(-1, cfg["vocab_size"])
        next_token = int(logits[row_in_tile].argmax())
        generated = [next_token]
        print(f"first generated (action) token: {next_token}")

        current_pos = last_real_idx + 1
        for step in range(ACTION_DIM - 1):
            token_embed = torch.nn.functional.embedding(
                torch.tensor([[generated[-1]]]), llama_sd["tok_embeddings.weight"]
            )  # (1,1,hidden)
            token_embed_tt = ttnn.from_torch(
                token_embed.unsqueeze(0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=mesh_device,
                mesh_mapper=shard_mapper,
            )
            pos_tensor = torch.tensor([current_pos])
            rot_mats_decode = model.rope_setup.get_rot_mats(pos_tensor)
            current_pos_tt = ttnn.from_torch(
                pos_tensor, device=mesh_device, dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
            )

            tt_out = model.forward(
                x=token_embed_tt, current_pos=current_pos_tt, rot_mats_global=rot_mats_decode,
                mode=Mode.DECODE, page_table=None, kv_cache=None, get_last_token=-1,
            )
            # Decode also returns a 32-row tile, but (unlike PREFILL's sequence-
            # position tile) this is a padded BATCH dimension -- max_batch_size=1
            # means only row 0 is real; the other 31 rows are uninitialized memory
            # for unused batch slots. Confirmed directly: reading row -1 gave
            # non-deterministic, sometimes wildly out-of-vocabulary-range tokens
            # across repeated runs of this same fixed-input, greedy-decoded pipeline
            # (a textbook uninitialized-memory signature -- see CLAUDE.md's own
            # "trust the subject, verify the instrument" notes), while row 0 is the
            # real prediction.
            step_logits = ttnn.to_torch(tt_out, mesh_composer=concat_composer).float().reshape(-1, cfg["vocab_size"])
            next_token = int(step_logits[0].argmax())
            generated.append(next_token)
            current_pos += 1
            print(f"decode step {step + 1}: token {next_token}")

        print(f"generated action token ids: {generated}")

        openvla_config = load_config()  # same pinned snapshot as the weights
        norm_stats = openvla_config["norm_stats"][UNNORM_KEY]["action"]
        vocab_size = openvla_config["text_config"]["vocab_size"] - openvla_config["pad_to_multiple_of"]

        action = None
        if len(generated) == ACTION_DIM:
            import numpy as np

            action = detokenize_actions(
                np.array(generated), vocab_size=vocab_size, action_norm_stats=norm_stats, n_action_bins=256,
            )
            print(f"DECODED ACTION (7-DoF, bridge_orig unnormalized): {action}")
        else:
            print(f"only {len(generated)}/{ACTION_DIM} action tokens generated -- skipping detokenization")
        return generated, action
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
