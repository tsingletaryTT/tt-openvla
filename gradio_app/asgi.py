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
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from PIL import Image
from starlette.responses import JSONResponse

import gradio as gr
import json_numpy

# Must run before any json.dumps/loads touches a request or response body carrying a
# numpy array (the /act route below): this monkey-patches the stdlib json module in
# place, so ordinary json.loads/json.dumps -- including the ones Starlette's own
# Request.json()/JSONResponse.render() call internally -- become numpy-aware. Patching
# at import time, not lazily inside the route, so there's no window where a request
# could be served before it's active.
json_numpy.patch()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import build_app  # noqa: E402
from backends import DEFAULT_UNNORM_KEY, TTNNBackend, build_openvla_prompt  # noqa: E402
from tt import openvla_weights  # noqa: E402  (backends put the repo root on sys.path)

# The only unnorm_key this demo actually defaults to (see app.py's build_app) -- used
# as-is if the real list can't be fetched at import time (no writable HF cache, or no
# network, in the image-build sandbox verify.sh runs in).
_FALLBACK_UNNORM_KEYS = ["bridge_orig"]


def _resolve_unnorm_keys() -> list[str]:
    """unnorm_key choices for the dropdown, from config.json at the PINNED revision
    (tt/openvla_weights.py). Only config.json is fetched here -- never the shards -- and
    any failure (no network, read-only cache at build-time verify) degrades to the
    static fallback instead of failing the import."""
    config = openvla_weights.try_load_config_without_download()
    if config is None:
        return _FALLBACK_UNNORM_KEYS
    return sorted(config["norm_stats"].keys())


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


def _physical_device_ids_from_env() -> list[int] | None:
    """Optional escape hatch for a shared/dev box: a dedicated production deployment
    (this kind's actual target) has no reason to set this, since `--device
    /dev/tenstorrent` there exposes only the chips that box actually has. On a shared
    box with other leases active, restricting `--device` to just the leased nodes
    instead breaks UMD's cluster/ethernet-topology discovery (confirmed: it hung
    indefinitely past 'Failed to discover available ethernet links' with no forward
    progress) -- discovery needs to see the whole board. This env var lets the launcher
    keep full device visibility for discovery while still pinning the actual mesh open
    to specific chip ids, so it can't collide with another lease-holder's chip."""
    raw = os.environ.get("OPENVLA_PHYSICAL_DEVICE_IDS")
    if not raw:
        return None
    return [int(x) for x in raw.split(",") if x.strip()]


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    real_backend = await asyncio.to_thread(
        TTNNBackend, mesh_physical_device_ids=_physical_device_ids_from_env()
    )
    _backend.set(real_backend)
    try:
        yield
    finally:
        await asyncio.to_thread(_backend.close)


app = FastAPI(lifespan=_lifespan)


def _parse_act_payload(payload: dict) -> tuple[dict, bool]:
    """Accept both request shapes upstream `vla-scripts/deploy.py` accepts.

    - Plain: {"image": <uint8 HxWx3 array>, "instruction": str, "unnorm_key": str?},
      numpy-aware via json_numpy (patched at import time above).
    - Double-encoded: {"encoded": "<json_numpy string of the plain payload>"}, for
      clients where json_numpy can't patch the HTTP layer. Upstream requires it to be the
      ONLY key; so do we.
    Returns (payload, was_encoded)."""
    if "encoded" in payload:
        if len(payload) != 1:
            raise ValueError("an 'encoded' payload must be the only key (same rule as upstream deploy.py)")
        return json.loads(payload["encoded"]), True
    return payload, False


@app.post("/act")
async def act(request: Request) -> JSONResponse:
    """OpenVLA's REST contract, modelled on upstream `vla-scripts/deploy.py`. NOT a
    byte-for-byte clone -- the differences below are deliberate and documented rather
    than hidden (0.1.1's docstring claimed exact parity; it wasn't).

    Request -- same as upstream: plain or {"encoded": ...} payload (see
    `_parse_act_payload`), same prompt template including `instruction.lower()` (`backends.build_openvla_prompt`), same
    greedy decoding, same token-29871 fix-up (in the backend).

    Response -- DIFFERS from upstream:
      * Always `{"action": [7 floats]}` (a JSON object with a plain list). Upstream returns
        the BARE array (json_numpy-encoded ndarray; for `encoded` requests, a json_numpy
        string). An unmodified upstream client that does `action = r.json()` and indexes it
        as an array must read `r.json()["action"]` here instead.
      * `unnorm_key` defaults to "bridge_orig". Upstream defaults to None, which for
        openvla-7b (trained on many datasets) fails its own assertion and returns "error".
      * Errors are HTTP 4xx/5xx with a JSON `detail`; upstream returns HTTP 200 with the
        string "error".

    Concurrency: the backend serializes predictions with a lock (see
    `TTNNBackend.predict_action`), so concurrent requests queue and are answered one at a
    time instead of racing on the single mesh (0.1.1 hung on a 4-way burst). Reuses the
    exact backend instance the Gradio UI calls (same weights, same mesh, same lock)."""
    try:
        payload, _was_encoded = _parse_act_payload(await request.json())
        image = Image.fromarray(np.asarray(payload["image"], dtype=np.uint8)).convert("RGB")
        instruction = payload["instruction"]
        unnorm_key = payload.get("unnorm_key") or DEFAULT_UNNORM_KEY
    except (KeyError, ValueError, TypeError) as e:
        return JSONResponse(status_code=400, content={"detail": f"bad /act payload: {e!r}"})
    result = await asyncio.to_thread(
        _backend.predict_action, image, build_openvla_prompt(instruction), unnorm_key=unnorm_key
    )
    return JSONResponse(content={"action": [float(x) for x in result["action"]]})


# path="/" (not "") makes Starlette's Mount 307-redirect "/" -> "//" -- reproduced in
# isolation outside this container with a minimal gr.Blocks() + mount_gradio_app, so
# it's a real gradio/Starlette root-mount quirk, not something specific to this app.
# The client bundle then throws `Invalid URL` trying to parse a config value built
# from the doubled path, and the UI never gets past "Loading...". The /act route
# above is registered on `app` before this mount, so Starlette matches it first --
# mount_gradio_app's root-path mount would otherwise shadow it.
app = gr.mount_gradio_app(app, _demo, path="")
