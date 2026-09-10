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


class _LazyTTNNBackend:
    """Defers `TTNNBackend.__init__` (opens a real 2-device Blackhole mesh) until the
    first actual prediction request. tt-dit-server's own verify.sh imports this module
    at image BUILD time, with no device passthrough or gozer lease -- constructing the
    real backend eagerly at import time would try to open hardware during `docker
    build` itself, not at `tt-model serve` time."""

    name = TTNNBackend.name

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._real: TTNNBackend | None = None

    def _get(self) -> TTNNBackend:
        if self._real is None:
            self._real = TTNNBackend(*self._args, **self._kwargs)
        return self._real

    def predict_action(self, *args, **kwargs):
        return self._get().predict_action(*args, **kwargs)

    def close(self):
        if self._real is not None:
            self._real.close()


_backend = _LazyTTNNBackend()

_config_path = hf_hub_download("openvla/openvla-7b", "config.json")
_unnorm_keys = sorted(json.load(open(_config_path))["norm_stats"].keys())

_demo = build_app(_backend, _unnorm_keys)

app = FastAPI()
app = gr.mount_gradio_app(app, _demo, path="/")
