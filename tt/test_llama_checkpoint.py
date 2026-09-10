# SPDX-License-Identifier: MIT
"""Confirms llama_checkpoint.py's extraction+renaming of OpenVLA's real LLaMA-2-7B
weights is complete and correct: loading them into a real `transformers
.LlamaForCausalLM` built from `get_llama2_config()` must produce zero missing/
unexpected keys, and a forward pass must run cleanly. This is a config/weights
self-consistency check, not a hardware test -- no ttnn/gozer involved -- but it's the
one thing standing between "these values look plausible" and "these values are
actually the right shape and actually load"."""

import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _hf_state_dict_from_meta(sd: dict) -> dict:
    """Reverses llama_checkpoint.py's HF->meta renaming, back into
    transformers.LlamaForCausalLM's own key naming, so the SAME renamed state dict this
    port will hand to tt_transformers can also be checked against the real HF model."""
    hf_sd = {
        "model.embed_tokens.weight": sd["tok_embeddings.weight"],
        "model.norm.weight": sd["norm.weight"],
        "lm_head.weight": sd["output.weight"],
    }
    meta_to_hf_layer = {
        "attention_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attention.wq.weight": "self_attn.q_proj.weight",
        "attention.wk.weight": "self_attn.k_proj.weight",
        "attention.wv.weight": "self_attn.v_proj.weight",
        "attention.wo.weight": "self_attn.o_proj.weight",
        "feed_forward.w1.weight": "mlp.gate_proj.weight",
        "feed_forward.w3.weight": "mlp.up_proj.weight",
        "feed_forward.w2.weight": "mlp.down_proj.weight",
    }
    for k, v in sd.items():
        m = re.match(r"layers\.(\d+)\.(.+)", k)
        if m:
            idx, rest = m.group(1), m.group(2)
            hf_sd[f"model.layers.{idx}.{meta_to_hf_layer[rest]}"] = v
    return hf_sd


def main():
    from transformers import LlamaConfig, LlamaForCausalLM

    from tt.llama_checkpoint import get_llama2_config, load_openvla_llama_state_dict

    cfg_dict = get_llama2_config()
    config = LlamaConfig(
        vocab_size=cfg_dict["vocab_size"],
        hidden_size=cfg_dict["hidden_size"],
        intermediate_size=cfg_dict["intermediate_size"],
        num_hidden_layers=cfg_dict["num_hidden_layers"],
        num_attention_heads=cfg_dict["num_attention_heads"],
        num_key_value_heads=cfg_dict["num_key_value_heads"],
        hidden_act=cfg_dict["hidden_act"],
        max_position_embeddings=cfg_dict["max_position_embeddings"],
        rms_norm_eps=cfg_dict["rms_norm_eps"],
        rope_theta=cfg_dict["rope_theta"],
        attention_bias=cfg_dict["attention_bias"],
        tie_word_embeddings=cfg_dict["tie_word_embeddings"],
        pad_token_id=cfg_dict["pad_token_id"],
        bos_token_id=cfg_dict["bos_token_id"],
        eos_token_id=cfg_dict["eos_token_id"],
    )

    print("Building LlamaForCausalLM module tree (meta device, no weight init cost)...")
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    model = model.to_empty(device="cpu")

    print("Loading real openvla/openvla-7b LLM weights via llama_checkpoint.py...")
    sd = load_openvla_llama_state_dict()
    hf_sd = _hf_state_dict_from_meta(sd)

    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    print(f"load_state_dict: 0 missing, 0 unexpected ({len(hf_sd)} keys)")

    model.eval()
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg_dict["vocab_size"], (1, 8))
    with torch.no_grad():
        out = model(input_ids)
    assert out.logits.shape == (1, 8, cfg_dict["vocab_size"])
    assert torch.isfinite(out.logits).all()
    print(f"forward pass OK, logits shape {tuple(out.logits.shape)}, all finite")
    print("PASS")


if __name__ == "__main__":
    main()
