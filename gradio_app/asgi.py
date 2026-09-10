# SPDX-License-Identifier: MIT
"""ASGI entry point for tt-model-manager's `tt-dit-server` kind (`runtime.app:
gradio_app.asgi:app`) -- mounts the same Gradio demo app.py builds onto a plain
FastAPI app, since `tt-dit-server` launches a `module:attribute` ASGI callable with
uvicorn directly rather than calling `demo.launch()` (which would start its own
server). Backend selection is fixed to TTNNBackend here: an image served this way is
assumed to actually have Blackhole hardware attached, unlike app.py's CLI, which
defaults to it but allows --backend reference for CPU-only iteration."""

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from huggingface_hub import hf_hub_download

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import build_app  # noqa: E402
from backends import TTNNBackend  # noqa: E402

_backend = TTNNBackend()

_config_path = hf_hub_download("openvla/openvla-7b", "config.json")
_unnorm_keys = sorted(json.load(open(_config_path))["norm_stats"].keys())

_demo = build_app(_backend, _unnorm_keys)

app = FastAPI()
app = gr.mount_gradio_app(app, _demo, path="/")
