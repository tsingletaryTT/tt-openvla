# SPDX-License-Identifier: MIT
"""Offline tests for the thin-bundle recipe (scripts/package-thin.sh and scripts/thin/*).

No hardware, no ttnn, no network (a test that needs the tt-metal closure wheel or the
`tt-model` CLI skips when it is absent). Each test asserts on the part that can break the
bundle, and was seen to fail when that part was broken (see tt-openvla CLAUDE.md, 2026-10-16).

    pytest tt/test_package_recipe.py
"""

import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
THIN = REPO / "scripts" / "thin"
SCRIPT = REPO / "scripts" / "package-thin.sh"


def _load(name: str):
    """Import a scripts/thin module by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(f"thin_{name}", THIN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


closure = _load("closure")
inspect_stage = _load("inspect_stage")


def _lines(p: Path) -> list[str]:
    return closure.read_list(p)


# ---------------------------------------------------------------------------------------
# closure.py: the tt-metal subset that ships in extra_code/
# ---------------------------------------------------------------------------------------
def _write(root: Path, rel: str, text: str = "") -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text)


def test_closure_follows_imports_relative_imports_and_data_files(tmp_path):
    """Fixture tree: absolute, relative and from-import edges, an unused module, a data file."""
    _write(tmp_path, "models/a.py", "from models.b import x\nfrom . import c\n")
    _write(tmp_path, "models/b.py", "import models.pkg.d\n")
    _write(tmp_path, "models/c.py", "")
    _write(tmp_path, "models/pkg/__init__.py", "")
    _write(tmp_path, "models/pkg/d.py", "from pathlib import Path\nP = Path(__file__).parent / 'sub/conf.yaml'\n")
    _write(tmp_path, "models/pkg/sub/conf.yaml", "k: v\n")
    _write(tmp_path, "models/unused.py", "")
    files, dyn = closure.derive(tmp_path, {"models.a"})
    assert files == ["models/a.py", "models/b.py", "models/c.py", "models/pkg/__init__.py",
                     "models/pkg/d.py", "models/pkg/sub/conf.yaml"]
    assert dyn == []


def test_closure_reports_dynamic_imports(tmp_path):
    _write(tmp_path, "models/a.py", "import importlib\nimportlib.import_module('models.b')\n")
    _write(tmp_path, "models/b.py", "")
    _, dyn = closure.derive(tmp_path, {"models.a"})
    assert dyn == [("models/a.py", 2)], "a dynamic import must be reported, not silently skipped"


def _closure_wheel() -> Path | None:
    """The pinned tt-metal wheel, if this machine has it (script sha256 is checked by the recipe)."""
    pins = SCRIPT.read_text()
    sha = re.search(r'^CLOSURE_WHEEL_SHA="([0-9a-f]{64})"', pins, re.M).group(1)
    import hashlib
    cands = [Path.home() / ".cache/tt-model/models/episod/tt-openvla/wheels/tt_openvla_models_closure-0.77.0-py3-none-any.whl"]
    for c in cands:
        if c.is_file() and hashlib.sha256(c.read_bytes()).hexdigest() == sha:
            return c
    return None


def test_committed_closure_equals_what_served_code_reaches(tmp_path):
    """The wiring check: ttt_closure.txt is what the real AST walk of the real wheel produces."""
    wheel = _closure_wheel()
    if wheel is None:
        pytest.skip("pinned tt-metal closure wheel not on this machine")
    with zipfile.ZipFile(wheel) as z:
        z.extractall(tmp_path)
    served = _lines(THIN / "served_files.txt")
    files, dyn = closure.derive(tmp_path, closure.served_entry_modules(REPO, served))
    assert dyn == []
    assert files == _lines(THIN / "ttt_closure.txt")
    # Every file must exist in the wheel, so the recipe's copy step cannot miss one.
    assert all((tmp_path / f).is_file() for f in files)


# ---------------------------------------------------------------------------------------
# served_files.txt: the model's own code that ships in extra_code/
# ---------------------------------------------------------------------------------------
def test_served_files_exist_and_are_closed_under_local_imports():
    """Every tt.* / gradio_app.* module the served files import is itself in the list.

    asgi.py puts its own directory on sys.path and imports `app` and `backends` by bare name,
    so those two resolve to gradio_app/ files.
    """
    served = set(_lines(THIN / "served_files.txt"))
    assert all((REPO / f).is_file() for f in served), "a served file is missing from the repo"
    bare = {"app": "gradio_app/app.py", "backends": "gradio_app/backends.py"}
    missing = set()
    for rel in sorted(f for f in served if f.endswith(".py")):
        tree = ast.parse((REPO / rel).read_text())
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and not n.level and n.module:
                mods = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            for m in mods:
                top = m.split(".")[0]
                if top in ("tt", "gradio_app"):
                    path = m.replace(".", "/")
                    if (REPO / (path + ".py")).is_file() and path + ".py" not in served:
                        missing.add(path + ".py")
                elif rel == "gradio_app/asgi.py" and m in bare and bare[m] not in served:
                    missing.add(bare[m])
    assert not missing, f"imported by served code but not in served_files.txt: {sorted(missing)}"


def test_served_files_hold_no_tests_or_bring_up_scripts():
    bad = [f for f in _lines(THIN / "served_files.txt") if "/test_" in f or "benchmark" in f or "hf_reference" in f]
    assert not bad


# ---------------------------------------------------------------------------------------
# requirements.txt / constraints.txt / runtime-deps.in
# ---------------------------------------------------------------------------------------
def _active(path: Path) -> list[str]:
    out = [ln.split("#", 1)[0].strip() for ln in path.read_text().splitlines()]
    return [ln for ln in out if ln]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.lower())


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for ln in _active(path):
        m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", ln)
        if m:
            pins[_norm(m.group(1))] = m.group(2)
    return pins


def test_requirements_have_no_index_options_and_a_stock_ttnn():
    lines = _active(THIN / "requirements.txt")
    assert not [ln for ln in lines if ln.startswith("-")], "option lines (index options) are not allowed"
    ttnn = [ln for ln in lines if ln.startswith("ttnn")]
    assert len(ttnn) == 1 and re.fullmatch(r"ttnn==\d+\.\d+\.\d+", ttnn[0]), ttnn


def test_tt_model_does_not_append_the_http_stack_to_requirements():
    """tt-model appends fastapi/uvicorn/pydantic/pillow unless each name is listed; step 1 must not resolve them.

    Asserts on the real function (the wiring), then on the marker that keeps uv from
    resolving the names (the effect).
    """
    pkg = pytest.importorskip("tt_kernel.packaging")
    from tt_kernel.launchers import TtDitServerLauncher

    text = (THIN / "requirements.txt").read_text()
    assert pkg._merge_default_packages(text, TtDitServerLauncher.DEFAULT_PACKAGES) == text
    for name in ("fastapi", "uvicorn", "pydantic", "pillow"):
        lines = [ln for ln in text.splitlines() if re.match(rf"{name}\b", ln)]
        assert len(lines) == 1 and '; python_version < "0"' in lines[0], lines


def test_constraints_cover_every_runtime_dep_and_carry_no_stale_pin():
    cons = _pins(THIN / "constraints.txt")
    deps = {_norm(re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", ln).group(0)) for ln in _active(THIN / "runtime-deps.in")}
    reqs = {_norm(re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", ln).group(0)) for ln in _active(THIN / "requirements.txt")
            if "python_version" not in ln}
    assert not ((deps | reqs) - set(cons)), f"not pinned in constraints.txt: {sorted((deps | reqs) - set(cons))}"
    # The versions the PyTorch CPU index serves. Any of them in constraints.txt means the resolve
    # was done against that index for a package it must not supply.
    stale = {"certifi": "2022.12.7", "requests": "2.28.1", "urllib3": "1.26.13", "idna": "3.4", "charset-normalizer": "2.1.1"}
    assert {k: v for k, v in stale.items() if cons.get(k) == v} == {}
    # constraints.txt and requirements.txt must not disagree about the engine.
    for k, v in _pins(THIN / "requirements.txt").items():
        assert cons[k] == v, f"{k}: requirements {v} vs constraints {cons[k]}"


# ---------------------------------------------------------------------------------------
# --verify guards (install.sh runs these with the venv's python)
# ---------------------------------------------------------------------------------------
GUARD_NAMES = ["VERIFY_TTNN", "VERIFY_NO_MODELS_TREE", "VERIFY_EXTRA_CODE", "VERIFY_NOT_STALE", "VERIFY_PINS"]


def _guards() -> dict[str, str]:
    """The guard statements exactly as package-thin.sh defines them."""
    script = ('TTNN_VERSION=0.77.0; eval "$(grep -E \'^VERIFY_[A-Z_]+=\' "$1")"; '
              + "".join(f'printf "%s\\0" "${n}"; ' for n in GUARD_NAMES))
    out = subprocess.run(["bash", "-c", script, "_", str(SCRIPT)], capture_output=True, text=True, check=True).stdout
    return dict(zip(GUARD_NAMES, out.split("\0")))


GOOD = {"ttnn": "0.77.0", "torch": "2.14.0+cpu", "gradio": "5.40.0", "transformers": "4.52.4", "tokenizers": "0.21.4",
        "certifi": "2026.7.22", "requests": "2.34.2", "urllib3": "2.8.0", "idna": "3.20", "charset-normalizer": "3.5.2"}


def _fake_bundle(tmp: Path, dists: dict[str, str], extra_files: list[str], site_models: bool = False) -> Path:
    """A bundle dir with a real (pip-less) venv, fake dist-info for each package, and extra_code/ files."""
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(tmp / "venv")], check=True)
    site = next((tmp / "venv" / "lib").glob("python*/site-packages"))
    for name, ver in dists.items():
        d = site / f"{name.replace('-', '_')}-{ver}.dist-info"
        d.mkdir()
        (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\n")
    if site_models:
        _write(site, "models/__init__.py")  # a regular `models` package in site-packages
    for rel in extra_files:
        _write(tmp / "extra_code", rel)
    return tmp


GOOD_FILES = ["gradio_app/asgi.py", "tt/functional_llama.py", "models/tt_transformers/tt/model.py",
              "models/common/lightweightmodule.py", "models/tt_transformers/tt/prefetcher/prefetcher_config.yaml"]


def _run_guard(tmp: Path, stmt: str) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(tmp / "extra_code"), "PATH": "/usr/bin:/bin"}
    return subprocess.run([str(tmp / "venv/bin/python"), "-c", stmt], capture_output=True, text=True, env=env)


@pytest.fixture(scope="module")
def guards():
    return _guards()


def test_guards_pass_on_the_canonical_layout(tmp_path, guards):
    b = _fake_bundle(tmp_path, GOOD, GOOD_FILES)
    for name, stmt in guards.items():
        r = _run_guard(b, stmt)
        assert r.returncode == 0, f"{name} rejected a good layout: {r.stderr[-300:]}"


@pytest.mark.parametrize("guard,dists,files,site_models", [
    ("VERIFY_TTNN", {**GOOD, "ttnn": "0.77.0+local"}, GOOD_FILES, False),
    ("VERIFY_TTNN", {**GOOD, "ttnn": "0.78.0"}, GOOD_FILES, False),
    ("VERIFY_NO_MODELS_TREE", GOOD, GOOD_FILES, True),
    ("VERIFY_NO_MODELS_TREE", {**GOOD, "tt-openvla-models-closure": "0.77.0"}, GOOD_FILES, False),
    ("VERIFY_EXTRA_CODE", GOOD, [f for f in GOOD_FILES if "model.py" not in f], False),
    ("VERIFY_EXTRA_CODE", GOOD, [f for f in GOOD_FILES if not f.endswith(".yaml")], False),
    ("VERIFY_NOT_STALE", {**GOOD, "certifi": "2022.12.7"}, GOOD_FILES, False),
    ("VERIFY_NOT_STALE", {**GOOD, "requests": "2.28.1"}, GOOD_FILES, False),
    ("VERIFY_PINS", {**GOOD, "gradio": "6.0.0"}, GOOD_FILES, False),
])
def test_each_guard_rejects_the_layout_it_exists_for(tmp_path, guards, guard, dists, files, site_models):
    b = _fake_bundle(tmp_path, dists, files, site_models)
    if "tt-openvla-models-closure" in dists:  # the leftover wheel's import name differs from its dist name
        _write(next((b / "venv" / "lib").glob("python*/site-packages")), "tt_openvla_models_closure/__init__.py")
    r = _run_guard(b, guards[guard])
    assert r.returncode != 0, f"{guard} accepted a layout it must reject"


# ---------------------------------------------------------------------------------------
# inspect_stage.py against a real `tt-model package-thin` staging of a small fixture
# ---------------------------------------------------------------------------------------
def _fixture_wheel(path: Path, deps: list[str]) -> Path:
    name = "tt_openvla_runtime_deps-0.3.0-py3-none-any.whl"
    meta = "Metadata-Version: 2.1\nName: tt-openvla-runtime-deps\nVersion: 0.3.0\n" + "".join(f"Requires-Dist: {d}\n" for d in deps)
    whl = path / name
    with zipfile.ZipFile(whl, "w") as z:
        z.writestr("tt_openvla_runtime_deps/__init__.py", "")
        z.writestr("tt_openvla_runtime_deps-0.3.0.dist-info/METADATA", meta)
        z.writestr("tt_openvla_runtime_deps-0.3.0.dist-info/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        z.writestr("tt_openvla_runtime_deps-0.3.0.dist-info/RECORD", "")
    return whl


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """(bundle dir, fixture thin dir) staged by the real CLI, with the recipe's flags."""
    if not shutil.which("tt-model"):
        pytest.skip("tt-model CLI not installed")
    root = tmp_path_factory.mktemp("stage")
    thin = root / "thin"
    thin.mkdir()
    (thin / "served_files.txt").write_text("gradio_app/asgi.py\n")
    (thin / "ttt_closure.txt").write_text("models/common/lightweightmodule.py\n")
    (thin / "runtime-deps.in").write_text("gradio==5.40.0   # comment\nfastapi\n")
    _write(root, "ec/gradio_app/asgi.py")
    _write(root, "ec/models/common/lightweightmodule.py")
    (root / "model.py").write_text("")
    # Reuse the real requirements and constraints files, so the option-line check sees the real text.
    wheel = _fixture_wheel(root, ["gradio==5.40.0", "fastapi"])
    guard = "import sys; sys.exit(0)"
    out = root / "bundle"
    r = subprocess.run(
        ["tt-model", "package-thin", "--out", str(out), "--name", "tt-openvla", "--kind", "tt-dit-server",
         "--app", "gradio_app.asgi:app", "--no-vllm", "--model-py", str(root / "model.py"),
         "--extra-code", str(root / "ec"), "--requirements", str(THIN / "requirements.txt"),
         "--constraints", str(THIN / "constraints.txt"), "--python", "3.12.12", "--ops-wheel", str(wheel),
         "--arch", "blackhole", "--mesh", "P300", "--device-count", "2",
         "--weights", "openvla/openvla-7b", "--weights-revision", "47a0ec7fc4ec123775a391911046cf33cf9ed83f"]
        + sum((["--verify", guard] for _ in range(5)), []),
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    return out, thin


def _failed(bundle: Path, thin: Path) -> list[str]:
    return [m for ok, m in inspect_stage.inspect(bundle, thin=thin) if not ok]


def test_inspect_accepts_the_canonical_staging(staged):
    bundle, thin = staged
    assert _failed(bundle, thin) == []


@pytest.mark.parametrize("name", ["models_wheel", "index_option", "missing_extra_code_file", "pythonpath_order",
                                  "unsplit_cmd_token", "ttnn_local_tag", "extra_file", "no_constraints"])
def test_inspect_rejects_each_departure_from_the_canonical_shape(staged, tmp_path, name):
    bundle, thin = staged
    b = tmp_path / "b"
    shutil.copytree(bundle, b, symlinks=True)
    man = json.loads((b / "tt_kernel_manifest.json").read_text())
    run = (b / "run.sh").read_text()
    if name == "models_wheel":      # the old shape: the code-bearing wheel as a models wheel
        man["deps"]["models_wheels"] = ["wheels/tt_openvla_models_closure-0.77.0-py3-none-any.whl"]
        (b / "tt_kernel_manifest.json").write_text(json.dumps(man))
    elif name == "index_option":
        (b / "requirements.txt").write_text("--extra-index-url https://download.pytorch.org/whl/cpu\n" + (b / "requirements.txt").read_text())
    elif name == "missing_extra_code_file":
        (b / "extra_code/models/common/lightweightmodule.py").unlink()
    elif name == "pythonpath_order":
        (b / "run.sh").write_text(run.replace('"$HERE/extra_code:$HERE:', '"$HERE:$HERE/extra_code:'))
    elif name == "unsplit_cmd_token":
        (b / "run.sh").write_text(run.replace("--lifespan on gradio_app.asgi:app", '"--lifespan on" gradio_app.asgi:app'))
    elif name == "ttnn_local_tag":
        (b / "requirements.txt").write_text((b / "requirements.txt").read_text().replace("ttnn==0.77.0", "ttnn==0.77.0+local"))
    elif name == "extra_file":      # a stray whole-tree module riding along in extra_code
        _write(b, "extra_code/models/demos/gpt_oss/x.py")
    elif name == "no_constraints":
        (b / "constraints.txt").unlink()
    assert _failed(b, thin), f"inspect_stage accepted the '{name}' departure"
