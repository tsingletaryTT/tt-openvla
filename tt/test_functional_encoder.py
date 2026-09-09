# SPDX-License-Identifier: MIT
"""First correctness checkpoint: DINOv2 patch embed + one encoder block, TTNN vs the
real `transformers.Dinov2Model` reference, on the real `facebook/dinov2-large`
checkpoint, at OpenVLA's actual 224px input size (not DINOv2's own 518px default).

PCC >= 0.995 is the acceptance bar (ttm-functional-decoder's default, same one
tt-vjepa2 uses), even though this is an encoder not a decoder -- the bar itself
doesn't depend on causal-vs-bidirectional.
"""

import os
import sys
from pathlib import Path

import torch

TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise RuntimeError("Set TT_METAL_HOME to a tt-metal checkout.")
sys.path.insert(0, TT_METAL_HOME)  # for `models.common.lightweightmodule` -- tt-metal's own package, not this
# repo's or tt-vjepa2's. This repo depends on tt-metal only, never on tt-vjepa2 (no shared code between them --
# see README).

import ttnn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repo's own root, for `tt.*`
from tt.functional_encoder import Dinov2Config, Embeddings, EncoderLayer  # noqa: E402


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    from transformers import Dinov2Model

    cfg = Dinov2Config()  # image_size=224 (OpenVLA's actual size), patch_size=14 -> 16x16=256 patches + CLS = 257

    print("Loading real facebook/dinov2-large checkpoint...")
    ref_model = Dinov2Model.from_pretrained("facebook/dinov2-large")
    ref_model.eval()
    sd = {k: v.float() for k, v in ref_model.state_dict().items()}

    torch.manual_seed(0)
    pixel_values = torch.randn(1, cfg.in_chans, cfg.image_size, cfg.image_size)

    print("Running reference (transformers.Dinov2Model, real 224px forward)...")
    with torch.no_grad():
        ref_embeddings = ref_model.embeddings(pixel_values, bool_masked_pos=None)
        ref_block_out = ref_model.encoder.layer[0](ref_embeddings)[0]

    print("Running TTNN port...")
    device = ttnn.open_device(device_id=0)
    try:
        tt_embeddings_module = Embeddings.from_state_dict(sd, cfg=cfg, device=device)
        tt_layer0 = EncoderLayer.from_state_dict(sd, layer_idx=0, cfg=cfg, device=device)

        tt_embeddings = tt_embeddings_module(pixel_values)
        seq_len = cfg.grid_size * cfg.grid_size + 1
        tt_out = tt_layer0(tt_embeddings, batch=1, seq_len=seq_len)

        embeddings_pcc = pcc(ttnn.to_torch(tt_embeddings).reshape(ref_embeddings.shape), ref_embeddings)
        block_out_pcc = pcc(ttnn.to_torch(tt_out).reshape(ref_block_out.shape), ref_block_out)
        print(f"embeddings (patch embed + pos embed) PCC: {embeddings_pcc:.6f}")
        print(f"layer_0 output PCC: {block_out_pcc:.6f}")
        assert embeddings_pcc >= 0.995, f"embeddings PCC {embeddings_pcc} < 0.995"
        assert block_out_pcc >= 0.995, f"layer_0 PCC {block_out_pcc} < 0.995"
        print("PASS")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
