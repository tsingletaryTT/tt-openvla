---
tags:
- blackhole
- p300
- robotics
- tenstorrent
- thin
- tt-dit-server
- tt-model-cache
- ttnn
- vision-language-action
license: other
license_name: llama-2-community-license
license_link: https://github.com/meta-llama/llama/blob/main/LICENSE
pipeline_tag: robotics
base_model:
- openvla/openvla-7b
---

# tt-openvla

A TT-Metal / TTNN bring-up of [OpenVLA](https://github.com/openvla/openvla)-7B, a vision-language-action model for robot manipulation.

- **What it does:** given a camera image and a natural-language instruction, it predicts a 7-DoF end-effector action (dx, dy, dz, droll, dpitch, dyaw, gripper).
- **How:** a fused DINOv2 + SigLIP vision backbone feeds a LLaMA-2-7B language model. The action comes out as extra vocabulary tokens and is decoded back through a bin mapping.
- **Hardware:** a 2-chip Blackhole P300 mesh.
- **Status: Experimental.**
  - On 2 of 4 test inputs, the action tokens match the Hugging Face reference exactly.
  - On the other two, the largest action difference is 0.045, from bf16 numerical drift.
  - No task-success evaluation has been run (see Limitations).
- **Code:** [tsingletaryTT/tt-openvla](https://github.com/tsingletaryTT/tt-openvla).

Packaged and published with [tt-model-manager](https://github.com/tenstorrent/tt-model-manager) as a v6 thin bundle (manifest schema 6): a pip/venv install, not a container.

**No checkpoint is hosted here.** Weights are fetched at install or serve time from [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b) at a pinned revision, unconverted. The two vision towers, the projector and LLaMA-2-7B all load from that checkpoint's own tensors.

## At a glance

| | |
| --- | --- |
| Architecture | About 7B parameters, in this order: <br>1. DINOv2 ViT-L/14 (4 register tokens) + SigLIP ViT-So400M/14, at 224 px <br>2. 3-layer GELU projector (2176 → LLaMA dim) <br>3. 32-layer LLaMA-2-7B, reusing tt-metal `tt_transformers` attention, RoPE and KV cache |
| Hardware | p300: 2 Blackhole chips, tensor-parallel 1×2 mesh. One chip overflows L1 in decode |
| Input / output | In: one RGB image (224 × 224 after preprocessing) and one instruction. Out: one 7-DoF action, from 7 generated tokens (1 prefill + 6 decode) |
| License | Port code MIT · vendored tt-metal code Apache-2.0 · weights MIT per upstream. The LLaMA backbone is a Llama 2 derivative, so the Llama 2 Community License and AUP apply. See Licensing |
| Status | Experimental |
| Model CI v0 | not yet run |

## Intended use

**Direct use:** research and bring-up evaluation of OpenVLA inference on Tenstorrent hardware:
- interactive exploration through the Gradio UI;
- offline or simulated action prediction through `POST /act`, for comparison against GPU OpenVLA.

**Out-of-scope use:**
- **Driving a physical robot without an independent safety layer**, meaning workspace limits, velocity and force clamps, an e-stop and a human supervisor. This port hasn't been evaluated on any robot or simulator task, and its actions aren't bit-identical to the reference model (see Limitations).
- **Robot embodiments or scenes outside the Open X-Embodiment pretraining mix.** Upstream states that OpenVLA doesn't generalize zero-shot to unseen embodiments or setups, and recommends fine-tuning instead.
- **Any use the [Llama 2 Acceptable Use Policy](https://github.com/meta-llama/llama/blob/main/USE_POLICY.md) prohibits.** That includes the §2 activities that present a risk of death or bodily harm, for example "Operation of critical infrastructure, transportation technologies, or heavy machinery".
- **Safety-critical, medical, or unsupervised deployment of any kind.**

## Quickstart

```bash
uv tool install tenstorrent   # once: the Tenstorrent CLI, `tt`
tt model pull episod/tt-openvla
tt serve episod/tt-openvla
```

`tt model pull` installs the v6 thin bundle: a hermetic uv venv with Python 3.12.12, stock `ttnn==0.77.0` from PyPI, the model's own code under `extra_code/`, and one small wheel (`tt-openvla-runtime-deps`) that declares the other Python packages the server imports. Every install step runs under the bundled `constraints.txt`.

- **Weights:** it also fetches `openvla/openvla-7b` at revision `47a0ec7` (about 15 GB) into the bundle's own HF cache (`<install>/.hf`). Weights come down by default; `tt` has no `--with-weights` flag.
- **Port:** `serve` starts uvicorn on port 20000, or the next free port. The Gradio UI is at `/` and the REST endpoint at `/act`.
- **Readiness:** the mesh open and model load run in the FastAPI lifespan, so uvicorn's `Application startup complete.` means the backend is ready.

Measured on a fresh install of the 0.2.1 bundle (one P300c board, 2026-09-27). The 0.3.0 bundle changes how the code is packaged. It does not change what the code computes. These figures have not been re-measured on it:

| step | time |
| --- | --- |
| Launch to ready, weights already pulled | 55 s, which includes building the on-disk tensor cache |
| Launch to ready, weights not pulled | 6.5 min: the server downloads the 15 GB itself, at the same pinned revision |
| First `/act` on a fresh install | 55 s, most of it the one-time kernel compile |
| Later warm calls | about 0.33 s |

No other model repos are fetched, apart from a small NousResearch LLaMA-2 config/tokenizer that tt-metal's `tt_transformers` reads.

Without tt-cli, tt-model alone does the whole job:

```bash
tt-model pull  episod/tt-openvla --with-weights
tt-model serve episod/tt-openvla
```

## Serve profiles

| profile | hardware | mesh |
| --- | --- | --- |
| p300 | P300 (2 Blackhole chips) | 1×2, `MESH_DEVICE=P300`, fabric 1D |

One prediction runs at a time. The Gradio UI is limited to `concurrency_limit=1`. `/act` requests and UI requests share one lock around the backend, so concurrent requests queue rather than run two decode loops on the mesh.

## Using it

**Not an OpenAI-compatible API.** A continuous action vector isn't natural-language output, so there is no `/v1/chat/completions`.

### Gradio UI (`/`)

1. Upload an image, or use the bundled example.
2. Write an instruction.
3. Choose the unnormalization dataset (default `bridge_orig`).
4. Click **Predict action**.

The UI shows the decoded 7-DoF action, the raw generated action-token IDs, and the measured latency.

### REST: `POST /act`

The request is the one [openvla/openvla](https://github.com/openvla/openvla)'s reference server (`vla-scripts/deploy.py`) takes:

- **Plain form:** a json-numpy-encoded `{image, instruction, unnorm_key?}`.
- **Encoded form:** `{"encoded": ...}`.

Both build the same prompt as upstream: `In: What action should the robot take to {instruction.lower()}?\nOut:`, plus the empty token upstream's `predict_action` appends.

It differs from upstream in three ways:

- **Response shape:** `{"action": [...]}`, where upstream returns the bare array. Upstream clients that do `action = resp.json()` need `resp.json()["action"]`.
- **Default `unnorm_key`:** `bridge_orig`. Upstream defaults to `None`, which errors for openvla-7b.
- **Errors:** a malformed payload, including a non-string `instruction`, returns HTTP 400 with a `detail` message. Upstream returns 200 with an `"error"` field.

```python
import json_numpy; json_numpy.patch()  # pip install json-numpy
import requests, numpy as np
from PIL import Image

image = np.array(Image.open("my_photo.jpg").convert("RGB"))  # uint8 HxWx3
resp = requests.post("http://localhost:20000/act", json={
    "image": image,
    "instruction": "pick up the remote control",
    # "unnorm_key": "bridge_orig",   # optional; any key in openvla-7b config.json norm_stats
})
action = resp.json()["action"]  # 7 floats: dx, dy, dz, droll, dpitch, dyaw, gripper
```

- **Shared backend:** the route reuses the backend the UI loaded, so there's no second model load.
- **Determinism:** decoding is greedy. Repeated and concurrent requests for the same input returned identical actions.

### From the GitHub repo

- **`gradio_app/app.py`** runs the same demo directly: on Blackhole by default (hold a 2-chip lease first), or on CPU with `--backend reference`.
- **`tt/hf_reference.py`** runs the upstream HF `AutoModelForVision2Seq` reference that the accuracy numbers below were measured against.

## Expected performance

| stage / metric | vs reference | result |
| --- | --- | --- |
| DINOv2 featurizer (openvla-7b weights) | PCC vs the CPU reference with the same weights | 0.99979 (0.99978-0.99988 across 4 inputs on the served 1×2 mesh) |
| SigLIP fused featurizer (openvla-7b weights) | same | 0.99975 (0.99972-0.99980) |
| Vision backbone + projector | same | 0.99970 (0.99961-0.99976) |
| Prefill logits | PCC vs HF `AutoModelForVision2Seq` (fp32, rev `47a0ec7`) | 0.994-0.995 on 4 inputs |
| 7 greedy action tokens, end to end | exact match vs HF `AutoModelForVision2Seq` + `predict_action` | **7/7 on 2 of 4 inputs**; 5/7 and 1/7 on the other two |
| Decoded action | max \|Δ\| vs HF, per input | 0.0 / 0.0 / 0.0107 / 0.0451 |
| Task success (BridgeData V2 / LIBERO) | none | not measured |
| **`/act` latency, warm, installed bundle** | none | **median 334.2 ms** (min 328.8, max 412.9), N=20; about 3.0 actions/s for one sequential client |
| 4 concurrent `/act` requests | none | all 4 returned HTTP 200 and queued, finishing within 1.35 s in total. The actions were identical to the sequential call |
| First `/act` on a fresh install (kernel JIT) | none | 55 s, once per install |

**Methodology:** the latency and concurrency figures come from HTTP `POST /act` on one P300c board (2 Blackhole chips, 1×2 mesh, `ttnn==0.77.0` from PyPI). The request was the bundled `gradio_app/assets/example.jpg` with "open the drawer" and `unnorm_key=bridge_orig`, json-numpy encoded, measured client-side. There was one untimed first call, then N=20 sequential warm calls and one 4-way concurrent burst. The run used this bundle revision (weights `47a0ec7`) on 2026-09-27.

The accuracy rows come from a separate run on the same board against the upstream HF reference on CPU (timm 0.9.16, transformers 4.40.1, greedy). It used four image/instruction pairs: the repo's example image with "open the drawer", a house in a field with "move forward", a dog with "Pick up the ball", and a synthetic orange image with "move forward". The script is `tt/hf_reference.py` in the GitHub repo.

**Why the remaining token mismatches look like precision, not a bug:**
- **Per layer:** all-token hidden-state PCC against HF is at least 0.99995 at every layer. Last-token PCC decays smoothly, from 0.9999 at layer 0 to about 0.997 at layer 30, with no step change.
- **Where the flips happen:** at low-margin steps. HF's own bf16 run flips a token on one of the same inputs.
- **Not tried yet:** precision options such as an fp32 LM head or HiFi4 decode.

## Limitations

- **Actions are close to the reference, not identical to it.**
  - Two of the four test inputs match HF token for token. On the other two, the first divergent token appears at a low-margin step, and the decoded action differs by at most 0.045.
  - Outputs are deterministic run to run, but a different input can land on a different bin than GPU OpenVLA.
- **No task-level evaluation.** There are no BridgeData V2, LIBERO, or real-robot results. The hardware checks used four still images, not robot scenes or closed-loop control.
- **The server binds `0.0.0.0` with no authentication.**
- **One validated topology:** P300, 2 chips. A single chip doesn't fit (L1 overflow in decode).
- **`/act` is serialized.** Throughput does not scale with concurrent clients: four simultaneous requests finish one after another.
- **Chatty logs.** tt-metal logs a `Failed to discover available ethernet links` warning hundreds of times per `/act` call. It is harmless, but it makes the log grow fast.

## Risks and safety considerations

- **Physical harm.** The model outputs end-effector deltas and a gripper command. If a robot executes them without independent limits, wrong actions can:
  - move it unexpectedly;
  - collide with people or objects;
  - drop or crush what it's holding.

  Nothing in this package bounds, filters, or sanity-checks actions. Keep a safety layer and an e-stop in the loop, and prefer simulation.
- **Silent numerical deviation.** On some inputs this port produces a different action than the reference (see Limitations), and the output looks plausible either way. A wrong action and a right one are indistinguishable from the response alone.
- **Out-of-distribution inputs.** For scenes, cameras, or embodiments outside the training mix, OpenVLA still returns a confident-looking action rather than refusing.
- **Unnormalization choice.** Actions are unnormalized with per-dataset statistics (`unnorm_key`). The wrong key for your robot rescales every motion, and the server defaults silently to `bridge_orig`.
- **Network exposure.** An unauthenticated `/act` on `0.0.0.0` lets anyone on the network issue actions to a connected control loop. Bind to localhost, or put it behind a proxy with auth.
- **Supply chain.** The processor loads with `trust_remote_code=True`, which executes the upstream repo's Python at serve time. That code is pinned to revision `47a0ec7`, the same revision as the weights.
- **Pass these risks on.** The Llama 2 AUP (§4) requires disclosing known dangers of your AI system to end users. If you build on this package, pass these risks along.

## Licensing

| component | license | notes |
| --- | --- | --- |
| Port code (`extra_code/gradio_app`, `extra_code/tt`, GitHub repo) | MIT | matches openvla/openvla |
| Vendored tt-metal `models/common` + `models/tt_transformers/tt`, 28 files under `extra_code/models/` | Apache-2.0 | upstream tenstorrent/tt-metal v0.77.0. Each file keeps its SPDX header |
| `openvla/openvla-7b` weights | MIT (upstream declaration) | The LLaMA-2-7B backbone is a Meta Llama 2 derivative, so the [Llama 2 Community License](https://github.com/meta-llama/llama/blob/main/LICENSE) and [Acceptable Use Policy](https://github.com/meta-llama/llama/blob/main/USE_POLICY.md) also apply: the 700M-MAU clause, the redistribution notice and the prohibited uses. The card's `license: other` reflects those terms |

This repo hosts no weights. Downloading and running `openvla-7b` is between you and its licensors.

## Changelog

| date | change |
| --- | --- |
| 2026-10-16 | 0.3.0: repackaged to the canonical v6 thin shape. The port code and a 28-file subset of the tt-metal closure ship under `extra_code/` in place of two wheels (the second carried 158 files, including gpt_oss and deepseek demos). `constraints.txt` is applied to every install step. Without it the PyTorch CPU index supplied a 2022 `certifi` and `requests 2.28.1`; a fresh 0.1.1 install, which used the same install step, had both. The code and the manifest's serving settings are unchanged. Not yet re-verified on hardware |
| 2026-09-27 | `tt-openvla-serving` 0.2.1 (0.2.0 plus two review fixes: a 400 for a non-string `/act` instruction, and a separate converted-weight cache per local checkpoint). See the list below this table. |
| 2026-09-15 | `tt-openvla-serving` 0.1.1: added the `POST /act` REST endpoint. The 0.1.0 wheel was removed |
| 2026-09-15 | First v6 thin bundle (`tt-model package-thin`), serving on the `tt-dit-server` kind alongside the existing card |
| 2026-09-10 | Model card published (code and weights pointers only) |

The 2026-09-27 release (`tt-openvla-serving` 0.2.1):

- **Vision towers:** now loaded from openvla-7b's own checkpoint tensors, replacing timm's generic pretrained weights.
- **Fresh installs:** weights are downloaded at the pinned revision `47a0ec7`. This fixes the "expected 3 safetensors shards, found []" startup failure.
- **Two RoPE bugs fixed,** which had masked each other:
  - a missing RoPE initialization in the CPU reference;
  - the wrong rotation pairs in TTNN.
- **Prompt:** the empty token upstream's `predict_action` appends is now added.
- **`/act`:**
  - serialized, so a 4-way concurrent burst no longer hangs the server;
  - lowercases the instruction;
  - accepts `{"encoded": ...}`;
  - returns 400 on a bad payload.
- **Manifest:** records the weights revision and `tt_metal_version` 0.77.0.
- **Re-verified on hardware** from a fresh install with an empty HF cache: 334 ms warm `/act`, 4/4 concurrent requests OK.

## Related packages

None.

## Feedback

- **Questions or problems with this package:** open a discussion at https://huggingface.co/episod/tt-openvla/discussions. That is the one channel that reaches the bundle's author.
- **A problem with the `tt` tooling itself:** use `tt report issue`. It files against tenstorrent/tt-cli and doesn't reach this package's author.
- **Product feedback:** support@tenstorrent.com.

## Provenance

| component | built from |
| --- | --- |
| tt-metal | `ttnn==0.77.0` (PyPI), plus 28 files of `models/common` and `models/tt_transformers/tt` from tt-metal v0.77.0 source, shipped under `extra_code/`. The manifest `tt_metal_version` is `0.77.0` |
| runtime | kind `tt-dit-server`: `gradio_app.asgi:app` under uvicorn, with gradio 5.40.0, transformers 4.52.4, torch 2.14.0+cpu |
| weights | `openvla/openvla-7b` @ `47a0ec7fc4ec123775a391911046cf33cf9ed83f`, pinned in the manifest and the code |
| port code | [tsingletaryTT/tt-openvla](https://github.com/tsingletaryTT/tt-openvla), staged by `scripts/package-thin.sh`. The commit and the sha256 of every input are in this bundle's `PROVENANCE.json` |
| runtime-deps wheel | `tt-openvla-runtime-deps` 0.3.0, built by the same script. It holds no code. It declares the packages listed in `runtime-deps.in` in the GitHub repo |
| Model CI v0 | not yet run |
| build | 0.3.0 staged 2026-10-16 · tt-model-manager (producer `tt_kernel_version` 0.1.0) |
