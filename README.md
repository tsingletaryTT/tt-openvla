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

**0.2.0 (2026-09-27): correctness against the real upstream model.** Earlier versions were
validated only against this repo's own composed PyTorch reference, and that reference shared
bugs with the TTNN path, so the old PCC figures (0.996-0.999) did not measure agreement with
OpenVLA. 0.2.0 fixes them and measures against upstream's own
`AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", trust_remote_code=True,
revision=47a0ec7f...)` on CPU (`tt/hf_reference.py`). Fixed:

- the vision towers were timm's *generic* pretrained DINOv2/SigLIP, not openvla-7b's
  fine-tuned towers (now loaded from the checkpoint's `vision_backbone.*` tensors);
- the TTNN LLaMA applied Meta-style RoPE to HF-layout wq/wk (wrong dimension pairs);
- the CPU reference had no RoPE at all (`to_empty()` left `inv_freq` uninitialized);
- the prompt lacked token 29871, which upstream `predict_action` appends;
- a fresh install could not find the weights; `/act` had no lock and hung on concurrent calls.

Measured on one p300c (2 chips), four image/instruction pairs, against upstream fp32:

| stage | result |
|---|---|
| DINOv2 / SigLIP towers, projector (TTNN, served mesh path) | PCC 0.9998 / 0.9997 / 0.9996-0.9998 |
| prefill logits (last position) | PCC 0.994-0.995 |
| 7 greedy action tokens, exact match | 5/7, 7/7, 1/7, 7/7 |
| max \|action diff\| (bridge_orig) | 0.011, 0.0, 0.045, 0.0 |

Token divergences happen at steps where upstream's own top-1/top-2 logit margin is small
(the 1/7 case: 0.68 at step 0; upstream's own bf16 CPU run also flips a token on that input).
Per-layer PCC decays smoothly from 0.9999 to about 0.998 across the 32 layers: accumulated
bf16 error, not a structural mismatch. Treat actions as close to, but **not bit-identical
with**, upstream OpenVLA. No task-success evaluation (BridgeData V2 / LIBERO) has been run.

## Interactive demo

`gradio_app/app.py` is a real Gradio UI: upload (or use the bundled example) image,
write an instruction, click "Predict action," and get back an actual decoded 7-DoF
action from a real forward pass -- the same pipeline `tt/demo_grounded_check.py`
validates, wrapped for repeated interactive use (weights load once at startup, not per
click). Two backends, same interface (`gradio_app/backends.py`, mirroring
[tt-vjepa2](https://github.com/tsingletaryTT/tt-vjepa2)'s own
`gradio_app/backends.py` pattern):

```bash
# Real Blackhole hardware (default) -- under a gozer lease covering 2 chips
gozer run --chips 2 --who "you:openvla" --reason "demo" -- \
    env TT_METAL_HOME=/path/to/tt-metal .venv/bin/python3 gradio_app/app.py

# CPU-only reference (what an HF Space without Tenstorrent hardware runs)
.venv/bin/python3 gradio_app/app.py --backend reference
```

Two real hardware/library bugs were found and fixed getting this working end to end
(both documented in `gradio_app/backends.py`'s own docstrings, worth reading before
touching this code): a `timm`-then-`transformers` load-order interaction that silently
poisons `LlamaForCausalLM`'s SDPA attention into NaN, and a fabric-handshake hang from
opening a plain single device alongside an open 2-device mesh in the same process
(fixed by running vision on the same mesh, replicated, instead of a separate device).

**Catalog entry**: `.disco/app.yaml` registers this demo with
[tt-discolike](https://github.com/tsingletaryTT/tt-discolike) (`chips: 2`, matching the
LLaMA backbone's real tensor-parallel mesh requirement) for one-click start/stop
alongside this machine's other TT gradio demos (tt-vjepa2, tt-animatediff).

## REST: `POST /act`

The served bundle (`gradio_app.asgi:app`) also exposes `/act`, modelled on upstream
`openvla/openvla`'s `vla-scripts/deploy.py`. **Requests** match upstream: a json_numpy body
`{"image": uint8 HxWx3, "instruction": str, "unnorm_key": str?}` or the double-encoded
`{"encoded": "<json_numpy string>"}`; the prompt is upstream's template with
`instruction.lower()`. **Responses differ** from upstream, deliberately and documented:
always `{"action": [7 floats]}` (upstream returns the bare array), `unnorm_key` defaults to
`bridge_orig` (upstream's `None` fails for openvla-7b), and errors are HTTP 4xx/5xx
(upstream returns 200 with `"error"`). An upstream client needs `r.json()["action"]`.
Concurrent requests queue behind one lock and are answered one at a time.

## Benchmarks

Real end-to-end latency (`tt/benchmark.py` / `tt/cpu_benchmark.py`) for the full
pipeline at the demo's real operating shape -- not traced replay (unlike
tt-vjepa2's own benchmark): this pipeline's shapes change per call (padded prefill
length depends on prompt length; decode position advances every step) in ways traced
replay doesn't tolerate, so this is real device execution time including host
dispatch -- an honest number for what the demo itself experiences, not a best-case
device-only figure.

| | latency/call | relative |
|---|---|---|
| Blackhole (TTNN, 2-chip mesh, kernel cache warm) | ~320 ms | 1x |
| same, first-ever call (one-time kernel JIT compilation, cached to disk after) | ~9-11 s | ~30x slower, once |
| CPU reference (composed real PyTorch, same host machine) | ~10.0 s | ~31x slower |

The one-time compilation cost is exactly that -- once: it persists in tt-metal's own
on-disk build cache across process restarts, not just within one, so a real user pays
it once per machine, not once per demo session.

## Publishing

Two publishing surfaces beyond GitHub + the HF model card above:

- **[tt-discolike](https://github.com/tsingletaryTT/tt-discolike)**: done, see
  *Interactive demo* above.
- **[tt-model-manager](https://github.com/tenstorrent/tt-model-manager)**: done, as a
  v6 **thin** bundle (`tt-model pull episod/tt-openvla --with-weights && tt-model serve
  episod/tt-openvla`) -- the earlier `tt-model.yaml` v5.1 CONTAINER attempt is gone
  (that whole schema is no longer the project's target; its build was blocked on this
  machine's shared TT_METAL_HOME checkout having uninitialized git submodules, never
  resolved). `pyproject.toml` builds `tt-openvla-serving`, the served-path closure
  (`gradio_app`'s ASGI app + the `tt/` functional_* modules it reaches); a second wheel
  vendors the `models/common` + `models/tt_transformers/tt` closure already
  hardware-verified the same day for episod/tt-tnt (identical tt-metal v0.77.0 source).
  This was the first non-diffusion model on the `tt-dit-server` kind (so far only
  exercised by FLUX.2-dev/tt-animatediff/tt-skyreels) -- hardware-verified end to end:
  mesh open on a real 2-chip P300, weights fetched, and a real prediction through the
  Gradio UI returned a genuine 7-DoF action (dx/dy/dz/droll/dpitch/dyaw/gripper) in
  ~45s on a cold cache.

## License

This repo's own code: MIT, matching [openvla/openvla](https://github.com/openvla/openvla)'s
own license. **The `openvla/openvla-7b` checkpoint itself is a different matter**: it's
a fine-tune of Meta's Llama-2-7B, so using those weights is subject to the
[Llama Community License](https://ai.meta.com/llama/license/) separately from this
repo's own MIT terms -- stated here plainly since the two licenses cover different
things (this repo's code vs. Meta's weights) and it would be easy to conflate them.

## Architecture (confirmed so far)

- **Vision**: DINOv2 ViT-L/14, register-token variant (architecture of timm's
  `vit_large_patch14_reg4_dinov2.lvd142m`) + SigLIP ViT-So400M/14 (architecture of
  `vit_so400m_patch14_siglip_224`), both at 224px. **Weights: openvla-7b's own fine-tuned
  `vision_backbone.featurizer.*` / `vision_backbone.fused_featurizer.*` tensors**, not timm's
  generic pretrained checkpoints (versions before 0.2.0 used the generic ones by mistake;
  the two differ by 0.2-0.46 relative L2 per tensor). timm is used only for the CPU
  reference's architecture (`pretrained=False`) and nothing is downloaded from timm's repos.
  **Fusion mechanism, confirmed from `modeling_prismatic.py`'s
  `PrismaticVisionBackbone`**: the 224x224x3 input image is preprocessed twice (once
  per tower's own normalization) and stacked into a 6-channel tensor; each tower runs
  through all but its *last* transformer block (`timm`'s
  `get_intermediate_layers(n={num_blocks-2})` -- no final norm, and CLS/register
  tokens are dropped, leaving only the 256 per-patch tokens); the two towers' patch
  tokens are concatenated along the feature dim (1024+1152=2176), then projected
  through a 3-layer GELU MLP (2176 -> 4x -> llm_dim -> llm_dim) into LLaMA's embedding
  space. All of the above is ported and validated against upstream's own modules on
  openvla-7b's weights (towers PCC 0.9998/0.9997, projector 0.9997; see Status). Note the
  towers are fine-tuned too, not frozen: only the checkpoint has the right weights.
- **Backbone**: LLaMA-2-7B (Llama Community License), all 32 layers, real fine-tuned
  `openvla-7b` weights, validated against upstream's own model (prefill logits PCC
  0.994-0.995; the older "0.9966" figure was against a RoPE-less reference and is void).
  wq/wk are converted to Meta RoPE layout (`reverse_permute`) before tt_transformers sees
  them. Built by reusing tt-metal's own
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
