# tt-openvla

A TT-Metal / TTNN bring-up of [OpenVLA](https://github.com/openvla/openvla): a
vision-language-action model for robot manipulation, combining a fused
DINOv2 + SigLIP vision backbone with a LLaMA-2-7B language model, fine-tuned on the
Open X-Embodiment dataset. Given an image and a natural-language instruction, it
predicts a 7-DoF end-effector action -- represented not by a separate action head, but
as extra tokens appended to the LLaMA tokenizer's vocabulary, generated
autoregressively like ordinary text and decoded back to continuous values via a bin
mapping.

A model card for this port is published at
[huggingface.co/episod/tt-openvla](https://huggingface.co/episod/tt-openvla) (no
separate checkpoint hosted there -- see its own README for why).

This is a separate repo from [tt-vjepa2](https://github.com/tsingletaryTT/tt-vjepa2)
deliberately: OpenVLA shares no architecture or checkpoint with V-JEPA2 (unlike, say,
a hypothetical V-JEPA 2.1 bring-up, which would reuse the existing encoder port
directly) -- the relationship between the two projects is topical (both are TT
bring-ups in the world-model/robotics-manipulation space), not code-sharing.

## Status

Core pipeline validated end to end, component by component, each against real
checkpoint weights at >=0.995 PCC: DINOv2 (register variant) -> SigLIP -> fused
VisionBackbone -> Projector -> full 32-layer LLaMA-2-7B backbone -> action
detokenization. See `tt/` for the individual modules and their correctness tests.

A real "Grounded Check" demo (`tt/demo_grounded_check.py`) runs a real image and
OpenVLA's own documented prompt through the whole pipeline end to end: vision -> fused
embeddings -> one PREFILL pass + 6 real `Mode.DECODE` steps through all 32 real
LLaMA-2-7B layers, on a real 2-device mesh -- producing an actual, deterministic
decoded 7-DoF action from real weights. Getting decode working took two real fixes:
custom embeddings needed the same tensor-parallel width-sharding
(`ttnn.ShardTensor2dMesh`) the framework's own `Embedding` module uses internally, and
a genuine non-determinism bug in this port's own decode-loop code (reading the wrong
row of a batch-padded output tile -- caught by rerunning the same fixed-input,
greedy-decoded pipeline and getting different, sometimes vocabulary-range-violating
results each time, then reverified deterministic after the fix). See that file's
module docstring for the full detail, including a separate, reproducible `transformers`
library bug hit and root-caused along the way (non-deterministic NaN logits from its
default "eager" attention implementation at this real sequence length -- fixed by
requesting `sdpa`).

## License

This repo's own code: MIT, matching [openvla/openvla](https://github.com/openvla/openvla)'s
own license. **The `openvla/openvla-7b` checkpoint itself is a different matter**: it's
a fine-tune of Meta's Llama-2-7B, so using those weights is subject to the
[Llama Community License](https://ai.meta.com/llama/license/) separately from this
repo's own MIT terms -- stated here plainly since the two licenses cover different
things (this repo's code vs. Meta's weights) and it would be easy to conflate them.

## Architecture (confirmed so far)

- **Vision**: DINOv2 ViT-L/14, register-token variant (`vit_large_patch14_reg4_dinov2
  .lvd142m` via timm -- NOT the plain `facebook/dinov2-large` this repo validated
  first; corrected after reading OpenVLA's own `configuration_prismatic.py`, which
  names the real timm checkpoint id) + SigLIP ViT-So400M/14 (`vit_so400m_patch14
  _siglip_224.webli` via timm, bit-identical to `google/siglip-so400m-patch14-224` --
  checked directly after finding the *bare*, untagged timm name now silently resolves
  to a SigLIP2 checkpoint that didn't exist when OpenVLA was released). Both run at
  224px (not DINOv2's own 518px default -- position embeddings need interpolating,
  handled host-side).
  **Fusion mechanism, confirmed from `modeling_prismatic.py`'s
  `PrismaticVisionBackbone`**: the 224x224x3 input image is preprocessed twice (once
  per tower's own normalization) and stacked into a 6-channel tensor; each tower runs
  through all but its *last* transformer block (`timm`'s
  `get_intermediate_layers(n={num_blocks-2})` -- no final norm, and CLS/register
  tokens are dropped, leaving only the 256 per-patch tokens); the two towers' patch
  tokens are concatenated along the feature dim (1024+1152=2176), then projected
  through a 3-layer GELU MLP (2176 -> 4x -> llm_dim -> llm_dim) into LLaMA's embedding
  space. All of the above is ported and validated (PCC 0.996-0.999) against real
  weights, including the Projector's own weights, which only exist in the
  `openvla/openvla-7b` checkpoint itself (a downstream fine-tuned artifact, not
  derivable from either frozen vision tower's own pretrained weights).
- **Backbone**: LLaMA-2-7B (Llama Community License), all 32 layers, real fine-tuned
  `openvla-7b` weights, validated end to end (PCC 0.9966 on final logits against a real
  `transformers.LlamaForCausalLM`). Built by reusing tt-metal's own
  `models/tt_transformers` `Transformer`/`ModelArgs` directly (its attention/RoPE/
  KV-cache kernels and tuned program configs, not hand-ported) rather than the
  separate, unfinished `models/experimental/openvla` attempt already in tt-metal
  (its own README documents "Full Model PCC: TBD" and an unresolved layer 0-2
  divergence bug, root-caused by inspection to passing `mode="prefill"` as a bare
  string where the framework compares against the `Mode.PREFILL` enum -- avoided here
  by always passing the actual enum). Deliberately targets `openvla-7b`'s own weights
  directly rather than the separate, gated `meta-llama/Llama-2-7b-hf` checkpoint: the
  fine-tuned weights already live in `openvla-7b`'s own (ungated) checkpoint, and the
  base architecture's config values are public, undisputed constants, not gated
  content -- only Meta's own weights are gated, which this port never touches.
- **Reference implementation**: real and strong -- `openvla/openvla-7b` is integrated
  into the standard `transformers` library (`AutoModelForVision2Seq`), giving a
  canonical implementation to validate correctness against at every stage, the same
  way [tt-vjepa2](https://github.com/tsingletaryTT/tt-vjepa2) validates against
  `facebookresearch/vjepa2`.
- **Evaluation precedent**: BridgeData V2 (real WidowX robot) and LIBERO (simulation)
  -- neither available in this environment. A real eval story here will likely look
  like tt-vjepa2's "Grounded Check": a real Open-X-Embodiment episode with a known
  ground-truth action, not a full benchmark suite, until proven otherwise.
