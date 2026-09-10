# SPDX-License-Identifier: MIT
"""CPU reference latency for the same real end-to-end pipeline tt/benchmark.py
measures on Blackhole -- what an HF Space without Tenstorrent hardware would
experience. No cold/warm distinction here: there's no device-side kernel JIT
compilation to amortize, so every call costs about the same."""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gradio_app"))

TIMED_ITERS = 3


def main():
    from backends import ReferenceBackend
    from PIL import Image

    backend = ReferenceBackend()
    image = Image.open(Path(__file__).resolve().parent.parent / "gradio_app" / "assets" / "example.jpg").convert("RGB")
    prompt = "In: What action should the robot take to open the drawer?\nOut:"

    print(f"Running {TIMED_ITERS} calls...")
    latencies = []
    for i in range(TIMED_ITERS):
        result = backend.predict_action(image, prompt, unnorm_key="bridge_orig")
        latencies.append(result["latency_ms"])
        print(f"  [{i}]: {result['latency_ms']:.1f} ms")

    print()
    print(f"mean: {statistics.mean(latencies):.1f} ms (min {min(latencies):.1f}, max {max(latencies):.1f})")


if __name__ == "__main__":
    main()
