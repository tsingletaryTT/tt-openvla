# SPDX-License-Identifier: MIT
"""Entry point `tt-model package-thin --model-py` requires.

This bundle serves with the `tt-dit-server` kind. `run.sh` starts uvicorn on
`gradio_app.asgi:app`, which resolves from `extra_code/` (first on PYTHONPATH), so nothing
reads this file at serve time. It re-exports the same ASGI app so `import model` and
`gradio_app.asgi:app` name one object.

There is no `import <ops package>` here because the model has no custom kernels: every
op is stock `ttnn`. If a `generic_op` wheel is added later, import it in this file;
`deps.custom_ops` is recorded from this file's imports.

Importing it does not open a device. `gradio_app/asgi.py` builds the Gradio graph at
import time and constructs the real backend only in the FastAPI lifespan.
"""

from gradio_app.asgi import app  # noqa: F401

__all__ = ["app"]
