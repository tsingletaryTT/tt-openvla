# SPDX-License-Identifier: MIT
"""Runtime monkeypatch for a real bug in tt-metal's own models/tt_transformers/tt/
common.py: `Mode` is a plain `Enum` (not a `str`-mixin), so `Mode.DECODE == "decode"`
is always False -- yet a handful of call sites in tt_transformers compare it directly
against raw strings expecting that to work. Found by direct source comparison while
debugging a container-only cross-chip fabric hang (this repo's own README/memory has
the full story); this bug isn't the hang's root cause, but it's real and independently
worth fixing, since it silently changes behavior every time it's hit.

Known buggy call sites at the time of writing (structurally the same bug this project
already found once before, in the abandoned models/experimental/openvla attempt --
see llama_checkpoint.py's own notes on that):
  - distributed_norm.py: three `mode == "decode"` checks that gate whether an
    all_gather_async call uses the model's pre-tuned num_links/chunks_per_sync/
    num_workers_per_link, or falls back to auto-discovering the link count at
    runtime. Since the comparison is always False, EVERY all_gather_async call --
    decode steps included, not just prefill -- takes the auto-discover branch.
  - decoder.py: two spots (`mode == "prefill"`, `mode == "decode"`).
  - model_config.py: one spot (logits vs. hidden_states branch).

Not our bug to fix in a shared tt-metal checkout, and not safe to edit a checkout
other sessions actively use -- patched at runtime here instead. Scoped narrowly:
only affects Mode-vs-string comparisons (currently always False, so this is a pure
correctness fix, not a behavior change to anything that depends on that being
False), and does nothing to Mode-vs-Mode or Mode-vs-other-type comparisons."""


def apply() -> None:
    from models.tt_transformers.tt.common import Mode

    if getattr(Mode, "_tt_openvla_str_eq_patched", False):
        return  # idempotent -- safe to call from more than one entry point

    original_eq = Mode.__eq__

    def _eq_with_value(self, other):
        if isinstance(other, str):
            return self.value == other
        return original_eq(self, other)

    Mode.__eq__ = _eq_with_value
    Mode._tt_openvla_str_eq_patched = True
