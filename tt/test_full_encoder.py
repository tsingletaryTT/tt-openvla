# SPDX-License-Identifier: MIT
"""Full-model correctness check: all 24 DINOv2 layers + final LayerNorm, TTNN vs the
real `transformers.Dinov2Model` reference, on the real `facebook/dinov2-large`
checkpoint, at OpenVLA's actual 224px input size. This is where depth-compounding
precision issues would show up if they exist (tt-vjepa2's own encoder found exactly
this shape of bug once, at 40 layers) -- block_0 alone passing (see
test_functional_encoder.py) doesn't guarantee the full stack does.
"""

import os
import sys
from pathlib import Path

import torch

TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise RuntimeError("Set TT_METAL_HOME to a tt-metal checkout.")
sys.path.insert(0, TT_METAL_HOME)

import ttnn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tt.functional_encoder import Dinov2Config, Model  # noqa: E402
from tt.test_functional_encoder import pcc  # noqa: E402


def main():
    from transformers import Dinov2Model as RefDinov2Model

    cfg = Dinov2Config()

    print("Loading real facebook/dinov2-large checkpoint...")
    ref_model = RefDinov2Model.from_pretrained("facebook/dinov2-large")
    ref_model.eval()
    sd = {k: v.float() for k, v in ref_model.state_dict().items()}

    torch.manual_seed(0)
    pixel_values = torch.randn(1, cfg.in_chans, cfg.image_size, cfg.image_size)

    print("Running reference (transformers.Dinov2Model, full 24-layer forward)...")
    with torch.no_grad():
        ref_out = ref_model(pixel_values).last_hidden_state

    print("Running TTNN port (full 24-layer forward)...")
    device = ttnn.open_device(device_id=0)
    try:
        tt_model = Model.from_state_dict(sd, cfg=cfg, device=device)
        tt_out = tt_model(pixel_values)

        full_pcc = pcc(ttnn.to_torch(tt_out).reshape(ref_out.shape), ref_out)
        print(f"full 24-layer model output PCC: {full_pcc:.6f}")
        assert full_pcc >= 0.995, f"full model PCC {full_pcc} < 0.995"
        print("PASS")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
