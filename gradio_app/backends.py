# SPDX-License-Identifier: MIT
"""Two interchangeable model backends for the demo app: one that runs this repo's own
TTNN port on real Blackhole hardware (needs a gozer lease held by the caller, and a
real 2-device mesh -- see tt/demo_grounded_check.py's own notes on why), and one that
runs a composed real-PyTorch reference on CPU (what an HF Space without Tenstorrent
hardware would use). Same interface (`predict_action(image, prompt) -> dict`), so the
app doesn't care which one is driving it -- mirrors tt-vjepa2's gradio_app/backends.py
pattern.

Unlike tt/demo_grounded_check.py (a one-shot correctness-test script that rebuilds
everything from scratch on every run), both backends here load weights ONCE in
__init__ and reuse them across repeated `predict_action` calls -- necessary for an
interactive demo, since reloading the vision towers + a 32-layer 7B model on every
button click would make the UI unusable."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tt.action_detokenizer import detokenize_actions  # noqa: E402
from tt.llama_checkpoint import get_llama2_config, load_openvla_llama_state_dict  # noqa: E402

ACTION_DIM = 7
DEFAULT_UNNORM_KEY = "bridge_orig"

_META_TO_HF_LAYER = {
    "attention_norm.weight": "input_layernorm.weight", "ffn_norm.weight": "post_attention_layernorm.weight",
    "attention.wq.weight": "self_attn.q_proj.weight", "attention.wk.weight": "self_attn.k_proj.weight",
    "attention.wv.weight": "self_attn.v_proj.weight", "attention.wo.weight": "self_attn.o_proj.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight", "feed_forward.w3.weight": "mlp.up_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
}


def _get_processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)


def _get_norm_stats_and_vocab():
    import json

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download("openvla/openvla-7b", "config.json")
    openvla_config = json.load(open(config_path))
    vocab_size = openvla_config["text_config"]["vocab_size"] - openvla_config["pad_to_multiple_of"]
    return openvla_config["norm_stats"], vocab_size


class ReferenceBackend:
    """Pure PyTorch, CPU. Real timm vision towers + real transformers.LlamaForCausalLM,
    built from the exact same openvla-7b weights this repo's TTNN port uses -- see
    tt/test_demo_grounded_check.py's reference_generate() for the one-shot version this
    is adapted from. attn_implementation="sdpa" is load-bearing, not cosmetic: the
    default "eager" implementation produced non-deterministic NaN logits at this real
    sequence length in this environment -- see that module's docstring."""

    name = "reference-pytorch"

    def __init__(self):
        from safetensors import safe_open
        from transformers import LlamaConfig, LlamaForCausalLM

        self.processor = _get_processor()
        self.norm_stats, self.vocab_size = _get_norm_stats_and_vocab()
        self.cfg = get_llama2_config()

        # Build LLaMA BEFORE importing/constructing timm's vision models -- load-
        # bearing ordering, not stylistic. Confirmed directly: `timm.create_model`
        # (for either DINOv2 or SigLIP) leaves some global torch SDPA backend state
        # behind that reliably poisons transformers' LlamaForCausalLM SDPA attention
        # afterward -- silent NaN logits, reproducible even with attn_implementation=
        # "sdpa" and even on plain random-token input having nothing to do with vision
        # at all. Building LLaMA first and loading timm's models only after sidesteps
        # it entirely (also verified directly, 3/3 clean trials). Root cause not fully
        # chased down (likely timm's fused-attention backend toggle leaking through a
        # process-global torch.backends.cuda.*_sdp flag), but the fix is real and
        # deterministic either way.
        print("[reference] loading LLaMA-2-7B (32 layers)...")
        sd = load_openvla_llama_state_dict()
        config = LlamaConfig(
            vocab_size=self.cfg["vocab_size"], hidden_size=self.cfg["hidden_size"],
            intermediate_size=self.cfg["intermediate_size"], num_hidden_layers=self.cfg["num_hidden_layers"],
            num_attention_heads=self.cfg["num_attention_heads"], num_key_value_heads=self.cfg["num_key_value_heads"],
            hidden_act=self.cfg["hidden_act"], max_position_embeddings=self.cfg["max_position_embeddings"],
            rms_norm_eps=self.cfg["rms_norm_eps"], rope_theta=self.cfg["rope_theta"],
            attention_bias=self.cfg["attention_bias"], tie_word_embeddings=self.cfg["tie_word_embeddings"],
            pad_token_id=self.cfg["pad_token_id"], bos_token_id=self.cfg["bos_token_id"],
            eos_token_id=self.cfg["eos_token_id"], attn_implementation="sdpa",
        )
        with torch.device("meta"):
            self.model = LlamaForCausalLM(config)
        self.model = self.model.to_empty(device="cpu")
        hf_sd = {
            "model.embed_tokens.weight": sd["tok_embeddings.weight"], "model.norm.weight": sd["norm.weight"],
            "lm_head.weight": sd["output.weight"],
        }
        for k, v in sd.items():
            m = re.match(r"layers\.(\d+)\.(.+)", k)
            if m:
                idx, rest = m.group(1), m.group(2)
                hf_sd[f"model.layers.{idx}.{_META_TO_HF_LAYER[rest]}"] = v
        missing, unexpected = self.model.load_state_dict(hf_sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        self.model.eval()
        self.tok_embed_w = sd["tok_embeddings.weight"]

        print("[reference] loading DINOv2 + SigLIP + Projector...")
        import timm

        self.dinov2 = timm.create_model(
            "vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224
        ).eval()
        self.siglip = timm.create_model(
            "vit_so400m_patch14_siglip_224.webli", pretrained=True, num_classes=0, img_size=224
        ).eval()

        import glob

        shard_path = glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/*/model-00001-of-00003.safetensors"
            )
        )[0]
        self.proj_sd = {}
        with safe_open(shard_path, framework="pt") as f:
            for k in f.keys():
                if k.startswith("projector."):
                    self.proj_sd[k[len("projector."):]] = f.get_tensor(k).float()
        print("[reference] ready")

    def predict_action(self, image, prompt: str, unnorm_key: str = DEFAULT_UNNORM_KEY) -> dict:
        t0 = time.perf_counter()
        inputs = self.processor(prompt, image)
        pixel_values, input_ids = inputs["pixel_values"], inputs["input_ids"]

        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        with torch.no_grad():
            patches = self.dinov2.get_intermediate_layers(img, n={len(self.dinov2.blocks) - 2})[0]
            patches_fused = self.siglip.get_intermediate_layers(img_fused, n={len(self.siglip.blocks) - 2})[0]
            fused_patches = torch.cat([patches, patches_fused], dim=2)
            x = torch.nn.functional.linear(fused_patches, self.proj_sd["fc1.weight"], self.proj_sd["fc1.bias"])
            x = torch.nn.functional.gelu(x)
            x = torch.nn.functional.linear(x, self.proj_sd["fc2.weight"], self.proj_sd["fc2.bias"])
            x = torch.nn.functional.gelu(x)
            vision_embeds = torch.nn.functional.linear(x, self.proj_sd["fc3.weight"], self.proj_sd["fc3.bias"])

        text_embeds = torch.nn.functional.embedding(input_ids, self.tok_embed_w)
        fused_embeds = torch.cat([text_embeds[:, :1, :], vision_embeds, text_embeds[:, 1:, :]], dim=1)
        attention_mask = torch.ones(fused_embeds.shape[:2], dtype=torch.long)

        with torch.no_grad():
            out = self.model.generate(
                inputs_embeds=fused_embeds, attention_mask=attention_mask, max_new_tokens=ACTION_DIM,
                do_sample=False, use_cache=True, pad_token_id=self.cfg["pad_token_id"],
            )
        generated = out[0].tolist()
        latency_ms = (time.perf_counter() - t0) * 1000

        norm_stats = self.norm_stats[unnorm_key]["action"]
        action = detokenize_actions(
            np.array(generated), vocab_size=self.vocab_size, action_norm_stats=norm_stats, n_action_bins=256,
        )
        return {"action": action, "tokens": generated, "latency_ms": latency_ms}


class TTNNBackend:
    """This repo's own TTNN port, real Blackhole hardware. Caller must already hold a
    gozer lease covering 2 chips and have TT_VISIBLE_DEVICES set -- this class does not
    acquire one. Needs a real 2-device mesh (not 1) -- see
    tt/demo_grounded_check.py's mesh-device comment for why a single chip overflows L1
    in decode mode, and why custom embeddings need explicit tensor-parallel sharding."""

    name = "ttnn-blackhole"

    def __init__(self, tt_metal_home: str | None = None, vision_device_id: int = 0):
        tt_metal_home = tt_metal_home or os.environ.get("TT_METAL_HOME")
        if not tt_metal_home:
            raise RuntimeError("Set TT_METAL_HOME (or pass tt_metal_home=) to a tt-metal checkout.")
        sys.path.insert(0, tt_metal_home)

        import ttnn
        from models.tt_transformers.tt.common import Mode

        from tt.demo_grounded_check import ACTION_DIM as _AD  # noqa: F401 (sanity: keep in sync)
        from tt.demo_grounded_check import build_fused_embeddings, get_vision_projector_weights
        from tt.functional_encoder import Model as Dinov2Model
        from tt.functional_llama import build_model, build_model_args, prefill_rot_mats
        from tt.functional_projector import Projector
        from tt.functional_siglip import Model as SiglipModel
        from tt.functional_vision_backbone import VisionBackbone

        self.ttnn = ttnn
        self.Mode = Mode
        self.build_fused_embeddings = build_fused_embeddings
        self.prefill_rot_mats = prefill_rot_mats

        self.processor = _get_processor()
        self.norm_stats, self.vocab_size = _get_norm_stats_and_vocab()
        self.cfg = get_llama2_config()
        self.llama_sd = load_openvla_llama_state_dict()

        print("[ttnn] opening vision device + building DINOv2/SigLIP/Projector...")
        self.vision_device = ttnn.open_device(device_id=vision_device_id)
        dcfg, dinov2_sd, scfg, siglip_sd, pcfg, proj_sd = get_vision_projector_weights()
        self.dinov2 = Dinov2Model.from_state_dict(dinov2_sd, cfg=dcfg, device=self.vision_device)
        self.siglip = SiglipModel.from_state_dict(siglip_sd, cfg=scfg, device=self.vision_device)
        self.backbone = VisionBackbone.from_models(self.dinov2, self.siglip)
        self.projector = Projector.from_state_dict(proj_sd, cfg=pcfg, device=self.vision_device)
        self.pcfg = pcfg

        print("[ttnn] setting fabric config + opening 1x2 mesh device for LLaMA-2-7B...")
        ttnn.set_fabric_config(True)
        self.mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2))
        # A fixed max_seq_len is chosen up front so the Transformer is built once here,
        # not rebuilt per call -- generously covers a normal instruction-length prompt
        # (real prompts in the OpenVLA paper's examples run well under 64 tokens) plus
        # 256 vision tokens plus ACTION_DIM generated tokens, rounded to a 128-multiple.
        self.max_seq_len = 512
        self.model_args = build_model_args(self.mesh_device, max_seq_len=self.max_seq_len)
        self.model = build_model(self.model_args, self.mesh_device, dtype=ttnn.bfloat16)
        self.shard_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device, dims=(None, 3), mesh_shape=self.model_args.cluster_shape
        )
        self.concat_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
        print("[ttnn] ready")

    def close(self):
        self.ttnn.close_mesh_device(self.mesh_device)
        self.ttnn.set_fabric_config(self.ttnn.FabricConfig.DISABLED)
        self.ttnn.close_device(self.vision_device)

    def _run_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        fused_patches_tt = self.backbone.forward(pixel_values)
        vision_embeds_tt = self.projector(fused_patches_tt)
        return self.ttnn.to_torch(vision_embeds_tt).reshape(1, 256, self.pcfg.llm_dim).float()

    def predict_action(self, image, prompt: str, unnorm_key: str = DEFAULT_UNNORM_KEY) -> dict:
        ttnn, Mode = self.ttnn, self.Mode
        t0 = time.perf_counter()

        inputs = self.processor(prompt, image)
        pixel_values, input_ids = inputs["pixel_values"], inputs["input_ids"]

        vision_embeds = self._run_vision(pixel_values)
        fused_embeds = self.build_fused_embeddings(input_ids, vision_embeds, self.llama_sd["tok_embeddings.weight"])
        real_seq_len = fused_embeds.shape[1]
        padded_seq_len = ((real_seq_len + ACTION_DIM + 127) // 128) * 128
        if padded_seq_len > self.max_seq_len:
            raise ValueError(
                f"prompt too long: padded sequence {padded_seq_len} exceeds this backend's "
                f"max_seq_len={self.max_seq_len} (built once at startup); use a shorter prompt."
            )
        pad_amount = padded_seq_len - real_seq_len
        padded_embeds = torch.nn.functional.pad(fused_embeds, (0, 0, 0, pad_amount))

        last_real_idx = real_seq_len - 1
        tile_aligned = (last_real_idx // 32) * 32
        row_in_tile = last_real_idx - tile_aligned

        embeds_tt = ttnn.from_torch(
            padded_embeds.unsqueeze(0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
            mesh_mapper=self.shard_mapper,
        )
        rot_mats = self.prefill_rot_mats(self.model, padded_seq_len)

        tt_out = self.model.forward(
            x=embeds_tt, current_pos=None, rot_mats_global=rot_mats, mode=Mode.PREFILL,
            page_table=None, kv_cache=None, get_last_token=tile_aligned,
        )
        logits = ttnn.to_torch(tt_out, mesh_composer=self.concat_composer).float().reshape(-1, self.cfg["vocab_size"])
        next_token = int(logits[row_in_tile].argmax())
        generated = [next_token]

        current_pos = last_real_idx + 1
        for _ in range(ACTION_DIM - 1):
            token_embed = torch.nn.functional.embedding(
                torch.tensor([[generated[-1]]]), self.llama_sd["tok_embeddings.weight"]
            )
            token_embed_tt = ttnn.from_torch(
                token_embed.unsqueeze(0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
                mesh_mapper=self.shard_mapper,
            )
            pos_tensor = torch.tensor([current_pos])
            rot_mats_decode = self.model.rope_setup.get_rot_mats(pos_tensor)
            current_pos_tt = ttnn.from_torch(
                pos_tensor, device=self.mesh_device, dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, None), mesh_shape=self.model_args.cluster_shape
                ),
            )
            tt_out = self.model.forward(
                x=token_embed_tt, current_pos=current_pos_tt, rot_mats_global=rot_mats_decode,
                mode=Mode.DECODE, page_table=None, kv_cache=None, get_last_token=-1,
            )
            # Decode's output tile has 32 rows because max_batch_size gets tile-padded
            # (row 0 is the only real batch slot; rows 1-31 are uninitialized garbage
            # for unused slots) -- NOT a sequence-position tile like PREFILL's. See
            # tt/demo_grounded_check.py's comment on this same line for how this was
            # caught (three repeated runs of a fixed-input pipeline gave three
            # different, sometimes vocabulary-range-violating results with row -1).
            step_logits = ttnn.to_torch(tt_out, mesh_composer=self.concat_composer).float().reshape(
                -1, self.cfg["vocab_size"]
            )
            next_token = int(step_logits[0].argmax())
            generated.append(next_token)
            current_pos += 1

        latency_ms = (time.perf_counter() - t0) * 1000
        norm_stats = self.norm_stats[unnorm_key]["action"]
        action = detokenize_actions(
            np.array(generated), vocab_size=self.vocab_size, action_norm_stats=norm_stats, n_action_bins=256,
        )
        return {"action": action, "tokens": generated, "latency_ms": latency_ms}
