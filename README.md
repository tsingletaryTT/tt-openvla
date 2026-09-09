# tt-openvla

A TT-Metal / TTNN bring-up of [OpenVLA](https://github.com/openvla/openvla): a
vision-language-action model for robot manipulation, combining a fused
DINOv2 + SigLIP vision backbone with a LLaMA-2-7B language model, fine-tuned on the
Open X-Embodiment dataset. Given an image and a natural-language instruction, it
predicts a 7-DoF end-effector action -- represented not by a separate action head, but
as extra tokens appended to the LLaMA tokenizer's vocabulary, generated
autoregressively like ordinary text and decoded back to continuous values via a bin
mapping.

This is a separate repo from [tt-vjepa2](https://github.com/tsingletaryTT/tt-vjepa2)
deliberately: OpenVLA shares no architecture or checkpoint with V-JEPA2 (unlike, say,
a hypothetical V-JEPA 2.1 bring-up, which would reuse the existing encoder port
directly) -- the relationship between the two projects is topical (both are TT
bring-ups in the world-model/robotics-manipulation space), not code-sharing.

## Status

Early bring-up. Currently validating feasibility component-by-component before
attempting the full model -- see `tt/` for what's been ported and correctness-tested
so far.

## License

This repo's own code: MIT, matching [openvla/openvla](https://github.com/openvla/openvla)'s
own license. **The `openvla/openvla-7b` checkpoint itself is a different matter**: it's
a fine-tune of Meta's Llama-2-7B, so using those weights is subject to the
[Llama Community License](https://ai.meta.com/llama/license/) separately from this
repo's own MIT terms -- stated here plainly since the two licenses cover different
things (this repo's code vs. Meta's weights) and it would be easy to conflate them.

## Architecture (confirmed so far)

- **Vision**: DINOv2 ViT-L/14 (`facebook/dinov2-large`: 24 blocks, 1024 hidden, 16
  heads, patch 14, LayerScale after each sub-block) + SigLIP ViT-So400M/14, both at
  224px (not DINOv2's own 518px default -- position embeddings need interpolating).
  Exact fusion mechanism between the two towers is not yet confirmed from documentation
  alone; this gets nailed down by reading `transformers`' actual OpenVLA/Prismatic
  model source once the individual encoders are validated.
- **Backbone**: LLaMA-2-7B (Llama Community License).
- **Reference implementation**: real and strong -- `openvla/openvla-7b` is integrated
  into the standard `transformers` library (`AutoModelForVision2Seq`), giving a
  canonical implementation to validate correctness against at every stage, the same
  way [tt-vjepa2](https://github.com/tsingletaryTT/tt-vjepa2) validates against
  `facebookresearch/vjepa2`.
- **Evaluation precedent**: BridgeData V2 (real WidowX robot) and LIBERO (simulation)
  -- neither available in this environment. A real eval story here will likely look
  like tt-vjepa2's "Grounded Check": a real Open-X-Embodiment episode with a known
  ground-truth action, not a full benchmark suite, until proven otherwise.
