# SPDX-License-Identifier: MIT
"""Real end-to-end latency for the full pipeline (vision backbone -> Projector ->
LLaMA-2-7B prefill + 6 decode steps), on real Blackhole hardware, at the real
operating shape this repo's Grounded Check demo uses (a real image + OpenVLA's own
prompt format).

Unlike tt-vjepa2's tt/benchmark.py, this does NOT use traced replay: the shapes here
change between calls in ways traced replay doesn't tolerate well (PREFILL's
padded_seq_len depends on the actual prompt length, and the decode loop's KV-cache
position advances every step), so this measures ordinary (non-traced) device
execution end to end -- a real, honest number for what the demo you'd actually click
through experiences, at the cost of including host-dispatch overhead that traced
replay would hide. See gradio_app/backends.py's own module docstring for the two real
hardware bugs (a single-chip L1 overflow in decode mode, a fabric-handshake hang from
mixing device contexts) that a naive first attempt at this pipeline runs into.

Reports COLD (first call -- includes kernel JIT compilation, cached to disk
afterward by tt-metal's own build cache) separately from WARM (every call after),
since the two differ by roughly 20x and conflating them misrepresents both."""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gradio_app"))

WARMUP_ITERS = 1  # the first call IS the cold-start measurement; nothing to discard
TIMED_ITERS = 5


def main():
    from backends import TTNNBackend
    from PIL import Image

    backend = TTNNBackend()
    image = Image.open(Path(__file__).resolve().parent.parent / "gradio_app" / "assets" / "example.jpg").convert("RGB")
    prompt = "In: What action should the robot take to open the drawer?\nOut:"

    print("Running COLD call (includes kernel JIT compilation)...")
    t0 = time.perf_counter()
    result = backend.predict_action(image, prompt, unnorm_key="bridge_orig")
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"cold: {cold_ms:.1f} ms (backend-reported: {result['latency_ms']:.1f} ms)")

    print(f"Running {TIMED_ITERS} WARM calls...")
    warm_latencies = []
    for i in range(TIMED_ITERS):
        result = backend.predict_action(image, prompt, unnorm_key="bridge_orig")
        warm_latencies.append(result["latency_ms"])
        print(f"  warm[{i}]: {result['latency_ms']:.1f} ms")

    backend.close()

    print()
    print(f"COLD (first call):  {cold_ms:.1f} ms")
    print(
        f"WARM (mean of {TIMED_ITERS}): {statistics.mean(warm_latencies):.1f} ms "
        f"(min {min(warm_latencies):.1f}, max {max(warm_latencies):.1f})"
    )


if __name__ == "__main__":
    main()
