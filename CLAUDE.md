# tt-openvla: project log

OpenVLA-7B on Tenstorrent Blackhole via TTNN (DINOv2+SigLIP towers, projector, LLaMA-2-7B
via tt_transformers), served as a v6 thin bundle (`episod/tt-openvla`, kind `tt-dit-server`,
app `gradio_app.asgi:app`) on one p300c board (2 chips, 1x2 mesh).

Earlier history (bring-up, Gradio, packaging) lives in the git log and README. This file
starts with the 0.2.0 correctness fix.

## Working notes

* Every device-touching command goes through gozer (`gozer run --chips 2 --who ...`).
  `import ttnn` opens the runtime: no bare imports outside a lease.
* Build the published wheel with `scripts/build-serving-wheel.sh`, never plain `uv build`
  (that one includes tests and bring-up scripts).
* Correctness ground truth is the REAL upstream model (`tt/hf_reference.py`, own venv from
  `tt/hf_reference.requirements.txt`, because upstream needs timm 0.9.x). Never validate
  against this repo's own composed reference alone: it once shared bugs with the TTNN path.
* A test under `tt/` imports `tt.*`. If a `tt` regular package is installed in the venv (the
  serving wheel), a checkout whose `tt/` lacks `__init__.py` is SHADOWED by the wheel. Tests
  print the file under test for this reason.

## 2026-09-27: tt-openvla-serving 0.2.0 (branch `fix/openvla-weights-and-towers`)

**Prompt (orchestrator):** fix three confirmed bugs from the HF-bundle review/bench: (1) the
vision towers were timm's generic pretrained weights, not openvla-7b's fine-tuned ones;
(2) a fresh install died with `expected 3 safetensors shards, found []` (loader globbed the HF
cache, never downloaded); (3) a 4-way concurrent `/act` burst hung the server. Also accept
upstream deploy.py's request shape, pin the weights revision, verify on hardware against the
real HF model, and bump 0.1.1 -> 0.2.0.

**What was fixed (7 bugs; the last four were found while verifying):**

1. Vision towers + projector now load from openvla-7b's own `vision_backbone.featurizer.*` /
   `vision_backbone.fused_featurizer.*` / `projector.*` tensors (`tt/openvla_weights.py`).
   Key names are timm's own plus `lsN.scale_factor` (renamed back to `gamma`); exact strict
   match against timm's architecture. timm is now architecture-only (`pretrained=False`) in
   the CPU reference and not used at all on the TTNN path.
2. Weights resolve via `huggingface_hub.snapshot_download` (honours `HF_HOME`, `HF_MODEL`
   repo-id-or-local-dir, `TT_MODEL_WEIGHTS_REVISION`, default pin `47a0ec7f...`). Processor
   (trust_remote_code) and config.json use the same pin. Shard names come from the index.
3. `/act` concurrency: a `threading.Lock` inside `TTNNBackend.predict_action` (both Gradio's
   worker thread and `/act`'s `asyncio.to_thread` reach it from threads, so a thread lock, not
   an asyncio one). `/act` accepts upstream's `encoded` payload and uses `instruction.lower()`.
   Response stays `{"action": [...]}`; the differences from upstream are documented in the
   `/act` docstring instead of claiming parity.
4. **Token 29871.** Upstream `predict_action` appends it after "Out:"; the processor doesn't.
   Neither path here did. Now `openvla_weights.append_empty_token` in both backends.
5. **CPU reference had no RoPE.** `with torch.device("meta"): ...; to_empty()` leaves the
   non-persistent `inv_freq` buffer uninitialized (~0 on this box), so every CPU LLaMA
   reference in the repo ran without positional encoding. `reinit_rope_buffers()` fixes all
   five sites. (Likely also the real cause of the old "eager attention -> NaN" note.)
6. **TTNN RoPE on the wrong pairs.** The port renamed HF keys to Meta names but skipped
   tt_transformers' `reverse_permute` of wq/wk, so Meta-style RoPE rotated wrong dimension
   pairs. Fixed in `OpenVLALlamaArgs.load_state_dict`; the tensor cache dir is now versioned
   (`openvla-7b-<rev12>-qk-meta/`) so a 0.1.1 cache can't be reloaded silently.
   Bugs 5 and 6 hid each other: a RoPE-less reference agreed with a wrong-RoPE TTNN model
   at "PCC 0.9966", which is why the old LLaMA test passed.
7. Gradio UI's default instruction was the whole question, then wrapped in the template
   again. One `build_openvla_prompt` now serves UI and `/act`.

**Evidence (details + numbers in the orchestrator's FIX.md):**
* New `tt/test_vision_towers_openvla_weights.py` FAILS on main@50fdee5 (wiring mismatch on
  every tower tensor; PCC dinov2 0.546, siglip 0.419, backbone+projector 0.457) and PASSES
  now (0.99979 / 0.99975 / 0.99970). First attempt at the fail-first run falsely PASSED: the
  0.2.0 wheel in the venv shadowed the old checkout's namespace `tt/`.
* CPU composed reference vs real HF model: towers PCC 1.0, 7/7 tokens on 4/4 cases.
* TTNN vs real HF (fp32): prefill logits PCC 0.994-0.995; tokens 5/7, 7/7, 1/7, 7/7. Per-layer
  last-token PCC drifts smoothly 0.9999 -> ~0.997-0.998 (bf16 accumulation, no step change).
  Divergences occur at low-margin steps (dog case: HF top-2 margin 0.68 at step 0; HF's own
  bf16 run also flips a token there). Not papered over: the grounded test asserts the first
  token only and reports the full match count.

**Key decisions:** keep `{"action": ...}` response (documented) rather than silently switch
shapes; assert first-token exact match, not 7/7, with the reasoning in the test docstring;
fix the reference bugs rather than regenerate numbers against a broken reference.

**Open:** full 7/7 exact match needs a precision change (fp32 LM head / higher-fidelity
decode), untried. tt_transformers `ModelArgs` still fetches `NousResearch/Llama-2-7b-hf`
config/tokenizer at startup (small, undeclared download).

### 2026-09-27 — PR #1 review fixes (still 0.2.0, unreleased)
- `/act` with a non-string `instruction` returned an unhandled 500; it is now validated inside
  the payload `try` and returns the documented 400.
- The converted-weight cache for a LOCAL `HF_MODEL` dir was keyed on the literal `local`, so
  two local checkpoints (or one updated in place) could silently share a stale cache. It is now
  keyed on `openvla_weights.local_checkpoint_fingerprint()`: resolved path + each shard/index
  file's name, size and mtime.
- `tt/hf_reference.py` duplicated the pinned sha (and treated an exported-but-empty
  `TT_MODEL_WEIGHTS_REVISION` as the revision ""); it now uses `openvla_weights.weights_revision()`.
- `tt/test_demo_grounded_check.py` records `REFERENCE_REVISION` and refuses to compare against a
  different checkpoint instead of reporting a meaningless pass/fail.
- New CPU-only `tt/test_serving_review_fixes.py` (ttnn import-blocked): 9 tests, all seen to
  fail against the previous head, pass now.
