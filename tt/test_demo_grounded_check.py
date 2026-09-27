# SPDX-License-Identifier: MIT
"""End-to-end check of the SERVED pipeline (gradio_app/backends.py's TTNNBackend: vision
towers -> projector -> 32-layer LLaMA prefill + 6 decode steps on a 1x2 mesh) against the
REAL upstream model's action tokens for one fixed input.

Ground truth: `AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b",
trust_remote_code=True, revision=<pinned>).predict_action(..., unnorm_key="bridge_orig",
do_sample=False)` on CPU, recorded by tt/hf_reference.py (its own venv: upstream needs timm
0.9.x). Input: gradio_app/assets/example.jpg, instruction "open the drawer", prompt built by
upstream deploy.py's template. fp32 and bf16 upstream runs agree on all 7 tokens:

    REFERENCE_GENERATED_IDS = [31885, 31872, 31852, 31881, 31894, 31876, 31744]

History, because the old numbers in this file were wrong for two independent reasons
(both fixed in tt-openvla-serving 0.2.0):
  - The old reference was this repo's own composed PyTorch pipeline, built from timm's
    GENERIC vision towers (not openvla-7b's) and a LlamaForCausalLM whose RoPE inv_freq was
    left uninitialized by `to_empty()` (so: no positional encoding). Its recorded ids
    [31883, 31895, ...] were not OpenVLA's output. With both fixed, that composed reference
    matches upstream 7/7 on all four cases tried.
  - The TTNN LLM received HF-layout wq/wk while tt_transformers applies Meta-layout RoPE
    (see functional_llama.OpenVLALlamaArgs.load_state_dict), and the prompt lacked the token
    29871 upstream predict_action appends. Before the fix: 0/7 tokens vs upstream, prefill
    logits PCC 0.55. After: prefill logits PCC 0.995, 5/7 tokens on this input.

What is asserted, and why not 7/7:
  - 7 finite action tokens -> a finite 7-DoF action (pipeline integrity).
  - The FIRST token matches upstream exactly. Step 0 is the one step not affected by
    autoregressive drift (every later step conditions on the previous greedy pick), so it
    is the cleanest single-token correctness check. On this input it holds with a margin:
    upstream's top-1/top-2 logit gap here is 1.18.
  - Full 7-token exact match is REPORTED, not asserted: on this input TTNN (bf16 weights and
    activations) diverges at step 4, where upstream's own logit margins are small; per-layer
    PCC shows a smooth bf16 drift (last-token PCC 0.9999 at layer 0 -> ~0.998 at layer 30),
    not a structural bug. Across four inputs measured 2026-09-27 the exact-match counts were
    5/7, 7/7, 1/7, 7/7 (the 1/7 case diverges at step 0, where upstream's top-2 margin is only
    0.68; upstream's own bf16 CPU run also flips a token on that input). Tightening this needs
    a precision change (e.g. fp32 LM-head / higher-fidelity decode), not a looser test.

Run under a lease:
  gozer run --chips 2 --who "claude:openvla-test" --reason "grounded e2e" -- \\
      python tt/test_demo_grounded_check.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gradio_app"))

REFERENCE_GENERATED_IDS = [31885, 31872, 31852, 31881, 31894, 31876, 31744]
REFERENCE_ACTION_BRIDGE_ORIG = [-0.00311655, -0.00042412, 0.01223037, -0.00524763, -0.02221627, -0.0081257, 0.99607843]
INSTRUCTION = "open the drawer"


def main():
    import numpy as np
    from PIL import Image

    from backends import ACTION_DIM, TTNNBackend, build_openvla_prompt

    image = Image.open(REPO_ROOT / "gradio_app" / "assets" / "example.jpg").convert("RGB")
    backend = TTNNBackend()
    try:
        result = backend.predict_action(image, build_openvla_prompt(INSTRUCTION), unnorm_key="bridge_orig")
    finally:
        backend.close()
    generated, action = result["tokens"], np.asarray(result["action"])
    print(f"TTNN generated:      {generated}")
    print(f"upstream reference:  {REFERENCE_GENERATED_IDS}")

    assert len(generated) == ACTION_DIM, f"expected {ACTION_DIM} action tokens, got {len(generated)}"
    assert action.shape == (ACTION_DIM,) and np.isfinite(action).all(), f"bad decoded action: {action}"

    exact = sum(a == b for a, b in zip(generated, REFERENCE_GENERATED_IDS))
    max_diff = float(np.max(np.abs(action - np.asarray(REFERENCE_ACTION_BRIDGE_ORIG))))
    print(f"exact token matches vs upstream: {exact}/{ACTION_DIM} (reported; see module docstring)")
    print(f"max |action - upstream action| (bridge_orig): {max_diff:.5f}")
    assert generated[0] == REFERENCE_GENERATED_IDS[0], (
        f"first action token {generated[0]} != upstream {REFERENCE_GENERATED_IDS[0]} -- "
        "step 0 has no autoregressive drift; a mismatch here is a pipeline bug, not bf16 noise"
    )
    print("PASS")
    return generated, action


def test_demo_grounded_check():
    main()


if __name__ == "__main__":
    main()
