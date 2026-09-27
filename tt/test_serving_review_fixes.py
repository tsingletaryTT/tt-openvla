"""CPU-only regression tests for the PR #1 review fixes. No device, and no ttnn at all:
`ttnn` is blocked in sys.modules before anything is imported, so a test that reached for
the device would fail with ImportError instead of quietly opening a chip.

    python -m pytest tt/test_serving_review_fixes.py -q
"""

import os
import sys
import time
from pathlib import Path

sys.modules["ttnn"] = None  # any `import ttnn` below raises ImportError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gradio_app"))

import pytest  # noqa: E402


# ---- /act: a bad instruction is a 400, not an unhandled 500 --------------------------------


@pytest.fixture
def act_client():
    from fastapi.testclient import TestClient

    import asgi

    # No `with`: the lifespan (which builds the real backend) never runs, so a request
    # that got past validation would fail loudly rather than touch a device.
    return TestClient(asgi.app, raise_server_exceptions=False)


@pytest.mark.parametrize("instruction", [123, None, ["open", "the", "drawer"], {"x": 1}])
def test_act_rejects_a_non_string_instruction_with_400(act_client, instruction):
    image = [[[0, 0, 0]] * 4] * 4  # a valid 4x4 RGB image, so only the instruction is bad
    r = act_client.post("/act", json={"image": image, "instruction": instruction})
    assert r.status_code == 400, r.text
    assert "instruction must be a string" in r.json()["detail"]


# ---- the converted-weight cache key for a local checkpoint dir -------------------------------


def _checkpoint(root: Path, payload: bytes) -> Path:
    root.mkdir(parents=True)
    (root / "model-00001-of-00001.safetensors").write_bytes(payload)
    (root / "model.safetensors.index.json").write_text("{}")
    return root


def test_two_local_checkpoints_get_different_cache_keys(tmp_path):
    from tt.openvla_weights import local_checkpoint_fingerprint

    a = _checkpoint(tmp_path / "a", b"weights-a")
    b = _checkpoint(tmp_path / "b", b"weights-a")  # same bytes, different checkpoint dir
    assert local_checkpoint_fingerprint(str(a)) != local_checkpoint_fingerprint(str(b))


def test_updating_a_local_checkpoint_in_place_changes_its_cache_key(tmp_path):
    from tt.openvla_weights import local_checkpoint_fingerprint

    ckpt = _checkpoint(tmp_path / "c", b"weights-v1")
    before = local_checkpoint_fingerprint(str(ckpt))
    shard = ckpt / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weights-v2-longer")
    os.utime(shard, ns=(time.time_ns(), time.time_ns() + 1_000_000_000))
    assert local_checkpoint_fingerprint(str(ckpt)) != before


def test_an_unchanged_local_checkpoint_keeps_its_cache_key(tmp_path):
    from tt.openvla_weights import local_checkpoint_fingerprint

    ckpt = _checkpoint(tmp_path / "d", b"weights")
    assert local_checkpoint_fingerprint(str(ckpt)) == local_checkpoint_fingerprint(str(ckpt))


# ---- one source of truth for the pinned revision ---------------------------------------------


def test_hf_reference_uses_the_canonical_pin(monkeypatch):
    import importlib

    from tt import openvla_weights

    monkeypatch.delenv("TT_MODEL_WEIGHTS_REVISION", raising=False)
    src = (REPO_ROOT / "tt" / "hf_reference.py").read_text()
    assert openvla_weights.PINNED_REVISION not in src  # no second copy of the sha
    import tt.hf_reference as ref

    assert importlib.reload(ref).REV == openvla_weights.PINNED_REVISION


def test_grounded_fixtures_are_recorded_at_the_pinned_revision():
    from tt import openvla_weights
    from tt.test_demo_grounded_check import REFERENCE_REVISION

    assert REFERENCE_REVISION == openvla_weights.PINNED_REVISION
