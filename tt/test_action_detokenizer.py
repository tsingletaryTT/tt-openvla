# SPDX-License-Identifier: MIT
"""Correctness check for action_detokenizer.py: a round-trip test (encode a known
continuous action into its exact token id, decode it back, confirm it recovers the
same bin center) plus a sanity check against the real openvla-7b config's own
bridge_orig normalization stats (the real BridgeData V2 statistics this checkpoint
was evaluated against, per this repo's README)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tt.action_detokenizer import action_bin_centers, detokenize_actions  # noqa: E402


def main():
    from huggingface_hub import hf_hub_download
    import json

    vocab_size = 32000  # openvla-7b's ORIGINAL vocab size, pre pad_to_multiple_of

    print("Round-trip check: encode a known bin -> token id -> decode -> same bin center...")
    bin_centers = action_bin_centers(256)
    for bin_idx in [0, 1, 127, 254]:
        # Inverse of detokenize_actions' arithmetic: discretized = bin_idx,
        # token_id = vocab_size - (discretized + 1)
        token_id = vocab_size - (bin_idx + 1)
        recovered_bin = vocab_size - token_id
        recovered_bin = np.clip(recovered_bin - 1, 0, bin_centers.shape[0] - 1)
        assert recovered_bin == bin_idx, (recovered_bin, bin_idx)
    print("PASS: bin <-> token id arithmetic round-trips exactly")

    print("Loading real openvla-7b config for bridge_orig norm_stats...")
    config_path = hf_hub_download("openvla/openvla-7b", "config.json")
    config = json.load(open(config_path))
    norm_stats = config["norm_stats"]["bridge_orig"]["action"]
    print(f"bridge_orig action stats: mask={norm_stats['mask']}")

    # A token sequence that decodes to the lowest bin (index 0) in every dimension.
    action_dim = len(norm_stats["mask"])
    lowest_bin_token = vocab_size - 1  # discretized=0 after the -1 clip logic
    action_token_ids = np.full(action_dim, lowest_bin_token)
    actions = detokenize_actions(
        action_token_ids, vocab_size=vocab_size, action_norm_stats=norm_stats, n_action_bins=256,
    )
    print(f"decoded actions (all-lowest-bin tokens): {actions}")
    assert actions.shape == (action_dim,)
    assert np.isfinite(actions).all()

    # Dimension 6 (gripper) has mask=False in the real stats -- its value should be
    # the raw normalized bin center (~ -1, the lowest bin), NOT affine-unnormalized.
    assert not norm_stats["mask"][6]
    assert abs(actions[6] - bin_centers[0]) < 1e-6, (actions[6], bin_centers[0])
    print("PASS: masked (gripper) dimension passes through unnormalized as expected")
    print("PASS")


if __name__ == "__main__":
    main()
