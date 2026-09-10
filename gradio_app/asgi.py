# SPDX-License-Identifier: MIT
"""ASGI entry point for tt-model-manager's `tt-dit-server` kind (`runtime.app:
gradio_app.asgi:app`) -- mounts the same Gradio demo app.py builds onto a plain
FastAPI app, since `tt-dit-server` launches a `module:attribute` ASGI callable with
uvicorn directly rather than calling `demo.launch()` (which would start its own
server). Backend selection is fixed to TTNNBackend here: an image served this way is
assumed to actually have Blackhole hardware attached, unlike app.py's CLI, which
defaults to it but allows --backend reference for CPU-only iteration.

Follows the same two contracts tt-animatediff's own server/app.py documents for this
kind (read that docstring for the full rationale):

1. **Readiness is the lifespan.** The mesh open + model load happen in the FastAPI
   lifespan below, awaited to completion, so uvicorn's "Application startup complete"
   actually means the backend can serve a request -- not lazily on first request, which
   would report ready while still loading and time out that first caller.
2. **Importing this module must not touch hardware (or require a writable HF cache /
   network).** `verify_lines` imports this module at image-BUILD time, on a machine
   with no card and no writable $HF_HOME, purely to prove `runtime.app` resolves. So
   `TTNNBackend()` is never constructed at module scope, and the one HF metadata fetch
   this module needs (the demo's dropdown of `unnorm_keys`) degrades to a static
   fallback instead of failing import if the cache/network isn't available."""

import asyncio
import contextlib
import json
import sys
from pathlib import Path

from fastapi import FastAPI

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import build_app  # noqa: E402
from backends import TTNNBackend  # noqa: E402

# The only unnorm_key this demo actually defaults to (see app.py's build_app) -- used
# as-is if the real list can't be fetched at import time (no writable HF cache, or no
# network, in the image-build sandbox verify.sh runs in).
_FALLBACK_UNNORM_KEYS = ["bridge_orig"]


def _resolve_unnorm_keys() -> list[str]:
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download("openvla/openvla-7b", "config.json")
        return sorted(json.load(open(config_path))["norm_stats"].keys())
    except Exception:
        return _FALLBACK_UNNORM_KEYS


class _BackendHandle:
    """Stands in for a real TTNNBackend while the Gradio Blocks graph is being built
    (import time -- before any device is open). The lifespan below constructs the real
    backend and installs it via `set()` before uvicorn reports startup complete."""

    name = TTNNBackend.name

    def __init__(self):
        self._real: TTNNBackend | None = None

    def set(self, real: TTNNBackend) -> None:
        self._real = real

    def predict_action(self, *args, **kwargs):
        if self._real is None:
            raise RuntimeError("backend not ready -- called before lifespan startup completed")
        return self._real.predict_action(*args, **kwargs)

    def close(self) -> None:
        if self._real is not None:
            self._real.close()


_backend = _BackendHandle()
_demo = build_app(_backend, _resolve_unnorm_keys())


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    real_backend = await asyncio.to_thread(TTNNBackend)
    _backend.set(real_backend)
    try:
        yield
    finally:
        await asyncio.to_thread(_backend.close)


app = FastAPI(lifespan=_lifespan)
# path="/" (not "") makes Starlette's Mount 307-redirect "/" -> "//" -- reproduced in
# isolation outside this container with a minimal gr.Blocks() + mount_gradio_app, so
# it's a real gradio/Starlette root-mount quirk, not something specific to this app.
# The client bundle then throws `Invalid URL` trying to parse a config value built
# from the doubled path, and the UI never gets past "Loading...".
app = gr.mount_gradio_app(app, _demo, path="")
