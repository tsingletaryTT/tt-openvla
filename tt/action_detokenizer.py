# SPDX-License-Identifier: MIT
"""Decodes OpenVLA's generated action tokens back into continuous 7-DoF actions.

Pure host-side post-processing (numpy only, no ttnn) -- this runs on whatever the LLM
backbone generates as its last `action_dim` token ids, after autoregressive decoding
finishes; nothing here touches hardware. Reference: tt-metal's own (unfinished)
`models/experimental/openvla/tt/open_vla.py`'s `get_final_action` -- reused as
knowledge (rewritten standalone, not imported) since it's simple, correct, and
independent of the rest of that framework.

OpenVLA represents each of the 7 action dimensions as one extra vocabulary token,
appended at the TOP of the tokenizer's id range (highest ids = action bins, in
reverse order): `discretized_actions = vocab_size - token_id`. `vocab_size` here is
deliberately the ORIGINAL 32000, not the padded 32064 (`pad_to_multiple_of=64`
adds 64 unused filler ids above the real range, per config.json's own
`text_config.vocab_size` / `pad_to_multiple_of` fields) -- confirmed directly against
tt-metal's own reference rather than assumed.

Each of 256 bins spanning [-1, 1] gets a token; the bin *center* (not edge) is the
recovered normalized value, which then gets affine-unnormalized per-dimension using
the real checkpoint's own `norm_stats[<dataset>]["action"]` quantile statistics
(q01/q99), with the `mask` field selecting which dimensions get unnormalized at all
(the gripper's mask is False in openvla-7b's own bridge_orig stats, for example --
its "normalized" value is used as-is)."""

from __future__ import annotations

import numpy as np


def action_bin_centers(n_action_bins: int = 256) -> np.ndarray:
    bins = np.linspace(-1, 1, n_action_bins)
    return (bins[:-1] + bins[1:]) / 2.0


def detokenize_actions(
    action_token_ids: np.ndarray,
    *,
    vocab_size: int,
    action_norm_stats: dict,
    n_action_bins: int = 256,
) -> np.ndarray:
    """`action_token_ids`: the last `action_dim` generated token ids (int array).
    `vocab_size`: the tokenizer's ORIGINAL vocab size (32000 for openvla-7b, not the
    padded 32064). `action_norm_stats`: one dataset's `norm_stats[<key>]["action"]`
    dict from the real checkpoint's config.json (has q01/q99/mask, each length
    action_dim)."""
    bin_centers = action_bin_centers(n_action_bins)

    discretized = vocab_size - np.asarray(action_token_ids)
    discretized = np.clip(discretized - 1, a_min=0, a_max=bin_centers.shape[0] - 1)
    normalized_actions = bin_centers[discretized]

    mask = np.array(action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool)))
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
