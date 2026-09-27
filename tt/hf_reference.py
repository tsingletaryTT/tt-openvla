"""Ground-truth OpenVLA reference: the REAL upstream model code, not this port's composition.

Runs `AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", trust_remote_code=True,
revision=<pinned>)` -- i.e. upstream's own modeling_prismatic.py, its own timm featurizers
loaded from the checkpoint's own vision_backbone.* tensors, its own predict_action() (which
appends token 29871 after "Out:") -- on CPU, in the pinned env upstream demands (timm 0.9.16,
transformers 4.40.1, tokenizers 0.19.1). Runs in its own venv (tt/hf_reference.requirements.txt)
because this port's serving stack pins timm 1.x, which modeling_prismatic.py rejects outright.

Usage:
  hfref/bin/python tt/hf_reference.py --cases cases.json --out ref/
where cases.json is [{"name": ..., "image": <path or "synthetic_orange">, "instruction": ...}].
The recorded REFERENCE_GENERATED_IDS in tt/test_demo_grounded_check.py came from this script
(case: gradio_app/assets/example.jpg + "open the drawer"; fp32 and bf16 agree).

For each (image, instruction) case it records, per dtype:
  - predict_action tokens + unnormalized action (bridge_orig), exactly as deploy.py would
  - intermediate tensors for per-stage PCC against the TTNN port:
      featurizer (DINOv2) patches, fused_featurizer (SigLIP) patches, projector output,
      every LLM hidden state of the prefill pass, and the prefill's last-position logits.
Output: <out_dir>/<case>_<dtype>.pt
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

REV = os.environ.get("TT_MODEL_WEIGHTS_REVISION", "47a0ec7fc4ec123775a391911046cf33cf9ed83f")


def load_cases(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtypes", default="float32,bfloat16")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True, revision=REV)
    cases = load_cases(args.cases)

    for dtype_name in args.dtypes.split(","):
        dtype = getattr(torch, dtype_name)
        t0 = time.time()
        vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True, revision=REV, torch_dtype=dtype,
            low_cpu_mem_usage=True, attn_implementation="sdpa",
        ).eval()
        print(f"[{dtype_name}] loaded in {time.time() - t0:.0f}s", flush=True)

        # Hooks on the two towers + projector: capture exactly what upstream's own forward
        # computes, rather than re-deriving it by hand.
        cap = {}
        vla.vision_backbone.featurizer.register_forward_hook(lambda m, i, o: cap.__setitem__("dinov2", o.detach().float().clone()))
        vla.vision_backbone.fused_featurizer.register_forward_hook(lambda m, i, o: cap.__setitem__("siglip", o.detach().float().clone()))
        vla.projector.register_forward_hook(lambda m, i, o: cap.__setitem__("projector", o.detach().float().clone()))

        for case in cases:
            image = Image.open(case["image"]).convert("RGB") if case["image"] != "synthetic_orange" else Image.new("RGB", (224, 224), color=(255, 100, 50))
            # deploy.py's exact prompt construction (instruction.lower()).
            prompt = f"In: What action should the robot take to {case['instruction'].lower()}?\nOut:"
            inputs = processor(prompt, image)
            pixel_values = inputs["pixel_values"].to(dtype)
            input_ids = inputs["input_ids"]

            t1 = time.time()
            with torch.no_grad():
                action = vla.predict_action(input_ids=input_ids, pixel_values=pixel_values,
                                            attention_mask=inputs["attention_mask"],
                                            unnorm_key="bridge_orig", do_sample=False)
            dt = time.time() - t1
            # Recover the tokens predict_action generated (it returns only the action): rerun
            # its own recipe via generate on the same augmented ids. (Checked: detokenizing
            # these reproduces predict_action's returned action exactly.)
            ids = input_ids
            if not torch.all(ids[:, -1] == 29871):
                ids = torch.cat((ids, torch.tensor([[29871]])), dim=1)
            with torch.no_grad():
                gen = vla.generate(input_ids=ids, pixel_values=pixel_values,
                                   attention_mask=torch.ones_like(ids), max_new_tokens=7, do_sample=False)
            tokens = gen[0, -7:].tolist()

            # One explicit prefill forward for per-stage captures (hidden states, logits).
            cap.clear()
            with torch.no_grad():
                out = vla(input_ids=ids, pixel_values=pixel_values, attention_mask=torch.ones_like(ids),
                          output_hidden_states=True)
            rec = {
                "case": case, "dtype": dtype_name, "revision": REV, "prompt": prompt,
                "input_ids_augmented": ids[0].tolist(), "pixel_values": inputs["pixel_values"].float(),
                "tokens": tokens, "action": np.asarray(action, dtype=np.float64).tolist(),
                "predict_action_seconds": dt,
                "dinov2": cap["dinov2"], "siglip": cap["siglip"], "projector": cap["projector"],
                "hidden_states": [h[0].float().clone() for h in out.hidden_states],
                "last_logits": out.logits[0, -1].float().clone(),
            }
            torch.save(rec, os.path.join(args.out, f"{case['name']}_{dtype_name}.pt"))
            print(f"[{dtype_name}] {case['name']}: tokens={tokens} action={np.round(action, 5).tolist()} ({dt:.1f}s)", flush=True)
        del vla


if __name__ == "__main__":
    main()
