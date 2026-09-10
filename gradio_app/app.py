# SPDX-License-Identifier: MIT
"""OpenVLA demo: a real image + a natural-language instruction, through this repo's
own real-weight pipeline (DINOv2+SigLIP vision backbone -> Projector -> LLaMA-2-7B,
one PREFILL pass + up to 7 real autoregressive DECODE steps), producing an actual
predicted 7-DoF end-effector action -- dx, dy, dz, droll, dpitch, dyaw, gripper.

This is the same "Grounded Check" tt/demo_grounded_check.py validates in its own
correctness test, wrapped as an interactive app: pick (or upload) an image, write an
instruction, and see the real decoded action plus the raw generated action-token ids
and measured latency. Action values are normalized to [-1, 1] except where the
checkpoint's own per-dataset stats mark a dimension unnormalized (e.g. the gripper
dimension in `bridge_orig`'s own stats) -- see tt/action_detokenizer.py.

Run locally against real Blackhole hardware (default): hold a gozer lease covering 2
chips and set TT_VISIBLE_DEVICES + TT_METAL_HOME first. Falls back to the CPU
reference implementation with --backend reference (what an HF Space without
Tenstorrent hardware would run, or a plain sanity check while iterating on this app
without touching hardware)."""

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_IMAGE_PATH = REPO_ROOT / "gradio_app" / "assets" / "example.jpg"
EXAMPLE_PROMPT = "What action should the robot take to open the drawer?"

ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

BACKEND_CONCURRENCY_ID = "model_backend"  # one request through the backend at a time


def _load_example_image() -> Image.Image | None:
    if EXAMPLE_IMAGE_PATH.exists():
        return Image.open(EXAMPLE_IMAGE_PATH).convert("RGB")
    return None


def build_app(backend, unnorm_keys: list[str]) -> gr.Blocks:
    with gr.Blocks(title="OpenVLA on Tenstorrent") as demo:
        gr.Markdown(
            f"""# OpenVLA on Tenstorrent Blackhole

A real image + a real instruction, through a from-scratch TTNN port of
[OpenVLA-7B](https://github.com/openvla/openvla) — DINOv2+SigLIP vision backbone,
LLaMA-2-7B language model — running on real Blackhole hardware
(backend: **{backend.name}**). Every number below comes from an actual forward pass
through all 32 real LLaMA layers with the real fine-tuned `openvla-7b` weights, not a
scripted or precomputed response.

Note: the default example image is a real photograph, not a matched robot scene (see
this repo's README for why) — it's here to exercise the real pipeline end to end, not
to demonstrate a semantically correct drawer-opening prediction."""
        )

        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="Image", type="pil", value=_load_example_image())
                prompt_in = gr.Textbox(
                    label="Instruction", value=EXAMPLE_PROMPT,
                    placeholder="What action should the robot take to ...?",
                )
                unnorm_key_in = gr.Dropdown(
                    label="Unnormalization stats (dataset)", choices=unnorm_keys,
                    value="bridge_orig" if "bridge_orig" in unnorm_keys else unnorm_keys[0],
                )
                run_btn = gr.Button("Predict action", variant="primary")
            with gr.Column():
                action_plot = gr.BarPlot(
                    x="dimension", y="value", title="Decoded 7-DoF action",
                    x_title="", y_title="normalized delta", vertical=False,
                )
                tokens_out = gr.Textbox(label="Generated action-token ids", interactive=False)
                latency_out = gr.Textbox(label="Latency", interactive=False)

        def run(image, prompt, unnorm_key):
            if image is None:
                raise gr.Error("Upload or select an image first.")
            if not prompt.strip():
                raise gr.Error("Enter an instruction.")
            full_prompt = f"In: What action should the robot take to {prompt.strip().rstrip('?')}?\nOut:"
            result = backend.predict_action(image, full_prompt, unnorm_key=unnorm_key)
            action_data = {"dimension": ACTION_LABELS, "value": [float(v) for v in result["action"]]}
            tokens_str = ", ".join(str(t) for t in result["tokens"])
            latency_str = f"{result['latency_ms']:.1f} ms ({backend.name})"
            return action_data, tokens_str, latency_str

        run_btn.click(
            run, inputs=[image_in, prompt_in, unnorm_key_in], outputs=[action_plot, tokens_out, latency_out],
            concurrency_id=BACKEND_CONCURRENCY_ID, concurrency_limit=1,
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ttnn", "reference"], default="ttnn")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    from tt.demo_grounded_check import UNNORM_KEY as _default_unnorm_key  # noqa: F401
    from tt.llama_checkpoint import get_llama2_config

    _ = get_llama2_config()  # fail fast on obvious config issues before loading anything heavy

    import json

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download("openvla/openvla-7b", "config.json")
    openvla_config = json.load(open(config_path))
    unnorm_keys = sorted(openvla_config["norm_stats"].keys())

    if args.backend == "ttnn":
        from backends import TTNNBackend

        backend = TTNNBackend()
    else:
        from backends import ReferenceBackend

        backend = ReferenceBackend()

    demo = build_app(backend, unnorm_keys)
    try:
        demo.launch(server_name="0.0.0.0", share=args.share)
    finally:
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()
