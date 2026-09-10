# SPDX-License-Identifier: MIT
"""Smoke test for the tt_transformers-backed LLaMA integration: builds a real
OpenVLALlamaArgs + Transformer with real weights but truncated to 1 layer (cheap to
build/run), and confirms the wiring (mesh device, ModelArgs subclass, weight loading,
a single Mode.PREFILL forward call) actually works end to end, before attempting the
full 32-layer real-weight correctness validation. Compares the 1-layer TTNN output
against a real transformers.LlamaModel truncated the same way (hidden_states[1], the
state right after decoder layer 0 -- output_hidden_states[0] is the pre-layer
embedding output)."""

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
from tt.functional_llama import build_model, build_model_args, prefill_rot_mats  # noqa: E402
from tt.llama_checkpoint import get_llama2_config, load_openvla_llama_state_dict  # noqa: E402


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def reference_hidden_state_after_layer0(sd: dict, cfg: dict, input_ids: torch.Tensor) -> torch.Tensor:
    from transformers import LlamaConfig, LlamaModel

    config = LlamaConfig(
        vocab_size=cfg["vocab_size"], hidden_size=cfg["hidden_size"], intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=1, num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"], hidden_act=cfg["hidden_act"],
        max_position_embeddings=cfg["max_position_embeddings"], rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"], attention_bias=cfg["attention_bias"],
        tie_word_embeddings=cfg["tie_word_embeddings"], pad_token_id=cfg["pad_token_id"],
        bos_token_id=cfg["bos_token_id"], eos_token_id=cfg["eos_token_id"],
    )
    with torch.device("meta"):
        model = LlamaModel(config)
    model = model.to_empty(device="cpu")

    hf_sd = {"embed_tokens.weight": sd["tok_embeddings.weight"], "norm.weight": sd["norm.weight"]}
    meta_to_hf_layer = {
        "attention_norm.weight": "input_layernorm.weight", "ffn_norm.weight": "post_attention_layernorm.weight",
        "attention.wq.weight": "self_attn.q_proj.weight", "attention.wk.weight": "self_attn.k_proj.weight",
        "attention.wv.weight": "self_attn.v_proj.weight", "attention.wo.weight": "self_attn.o_proj.weight",
        "feed_forward.w1.weight": "mlp.gate_proj.weight", "feed_forward.w3.weight": "mlp.up_proj.weight",
        "feed_forward.w2.weight": "mlp.down_proj.weight",
    }
    import re
    for k, v in sd.items():
        m = re.match(r"layers\.0\.(.+)", k)  # only layer 0 -- rest are unused by a 1-layer LlamaModel
        if m and m.group(1) in meta_to_hf_layer:
            hf_sd[f"layers.0.{meta_to_hf_layer[m.group(1)]}"] = v

    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    model.eval()
    # NOTE: for a 1-layer model, HF's hidden_states[1] is silently overwritten to equal
    # last_hidden_state (POST final-norm) -- a known HF convention (the last tuple
    # entry always equals last_hidden_state), not the raw pre-norm decoder output.
    # Capture the true raw layer-0 output directly via a forward hook instead, to
    # match TTNN's get_last_token=-1 (which returns pre-norm/pre-lm_head hidden states).
    captured = {}

    def _hook(module, args, output):
        captured["raw"] = output[0] if isinstance(output, tuple) else output

    handle = model.layers[0].register_forward_hook(_hook)
    with torch.no_grad():
        model(input_ids, output_hidden_states=True)
    handle.remove()
    return captured["raw"]


def main():
    cfg = get_llama2_config()
    print("Loading real openvla-7b LLM weights...")
    sd = load_openvla_llama_state_dict()

    torch.manual_seed(0)
    seq_len = 128  # tt_transformers' prefill attention requires seq_len % 128 == 0
    input_ids = torch.randint(0, cfg["vocab_size"], (1, seq_len))

    print("Running reference (real transformers.LlamaModel, truncated to layer 0)...")
    ref_hidden = reference_hidden_state_after_layer0(sd, cfg, input_ids)
    print(f"reference hidden_states[1] shape: {tuple(ref_hidden.shape)}")

    print("Opening 1x1 mesh device...")
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        print("Building OpenVLALlamaArgs + Transformer (n_layers=1)...")
        model_args = build_model_args(mesh_device, max_seq_len=128, n_layers=1)
        model = build_model(model_args, mesh_device, dtype=ttnn.bfloat16)

        print("Building inputs_embeds via the real tok_embeddings weight...")
        embed_w = sd["tok_embeddings.weight"]  # (vocab, hidden)
        inputs_embeds = torch.nn.functional.embedding(input_ids, embed_w)  # (1, seq_len, hidden)
        inputs_embeds_tt = ttnn.from_torch(
            inputs_embeds.unsqueeze(0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=mesh_device
        )  # (1,1,seq_len,hidden)

        rot_mats = prefill_rot_mats(model, seq_len)

        print("Running Mode.PREFILL forward (get_last_token=-1 -> raw hidden states, no norm/lm_head)...")
        tt_out = model.forward(
            x=inputs_embeds_tt,
            current_pos=None,
            rot_mats_global=rot_mats,
            mode=Mode.PREFILL,
            page_table=None,
            kv_cache=None,
            get_last_token=-1,
        )

        tt_hidden = ttnn.to_torch(tt_out).reshape(ref_hidden.shape)
        out_pcc = pcc(tt_hidden, ref_hidden)
        print(f"1-layer hidden-state PCC: {out_pcc:.6f}")
        assert out_pcc >= 0.99, f"1-layer smoke-test PCC {out_pcc} < 0.99"
        print("PASS")
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
