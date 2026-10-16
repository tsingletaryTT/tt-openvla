# SPDX-License-Identifier: MIT
"""Check a staged thin bundle against the canonical v6 shape. Standard library plus bash.

    python3 -I inspect_stage.py <staged-bundle-dir> [--wheel-name-prefix tt_openvla_runtime_deps]

Exits 0 and prints one line per check when the bundle is right. Exits 1 and lists every
failed check otherwise. It reads files only: it imports nothing from the bundle and
never touches a device.

What it covers (the Step 3 inspection list of docs/E2E_RECIPE_V6.md):
  * run.sh: the CMD array is expanded through bash and must come out as the exact uvicorn
    argv, one token per argument; PYTHONPATH starts with extra_code/; the weights revision
    and mesh settings are exported; VLLM_PLUGINS is not set.
  * manifest: kind tt-dit-server, stock ttnn pin, extra_code_dir, only the runtime-deps
    wheel under deps.wheels, no models wheels, no vLLM, deps.custom_ops lists no tooling or
    models wheel, a weights pointer pinned to a revision.
  * requirements.txt carries no index options. constraints.txt is staged.
  * install.sh passes --constraint to every uv step, runs `uv pip check`, and runs the
    --verify guards after the venv is built.
  * extra_code/ holds exactly the served files plus the tt-metal closure, nothing else.
"""

import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

THIN = Path(__file__).resolve().parent


def read_list(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def inspect(bundle: Path, wheel_prefix: str = "tt_openvla_runtime_deps", thin: Path = THIN) -> list[tuple[bool, str]]:
    """`thin` holds served_files.txt, ttt_closure.txt and runtime-deps.in. Tests point it at a small fixture."""
    results: list[tuple[bool, str]] = []

    def check(ok: bool, msg: str) -> None:
        results.append((bool(ok), msg))

    man = json.loads((bundle / "tt_kernel_manifest.json").read_text())
    deps = man.get("deps") or {}
    run_sh = (bundle / "run.sh").read_text()
    install_sh = (bundle / "install.sh").read_text()
    reqs = (bundle / "requirements.txt").read_text()

    # --- manifest ---------------------------------------------------------------------
    check(man.get("schema_version") == "6", "manifest schema_version is 6")
    check(deps.get("kind") == "tt-dit-server" and deps.get("app") == "gradio_app.asgi:app",
          "deps.kind is tt-dit-server and deps.app is gradio_app.asgi:app")
    check(deps.get("extra_code_dir") == "extra_code", "deps.extra_code_dir is extra_code")
    wheels = deps.get("wheels") or []
    check(len(wheels) == 1 and Path(wheels[0]).name.startswith(wheel_prefix),
          f"deps.wheels is exactly the runtime-deps wheel (got {wheels})")
    check(not deps.get("models_wheels"), f"deps.models_wheels is empty (got {deps.get('models_wheels')})")
    check(not deps.get("vllm"), "deps.vllm is empty (no vLLM step for this kind)")
    custom = deps.get("custom_ops") or []
    check(all("tt_transformers" not in c and "models" not in Path(c).name for c in custom),
          f"deps.custom_ops lists no tooling or models wheel (got {custom})")
    check(man.get("tt_metal_version") == "0.77.0", f"tt_metal_version is 0.77.0 (got {man.get('tt_metal_version')})")
    w = man.get("weights") or {}
    check(w.get("repo_id") == "openvla/openvla-7b" and re.fullmatch(r"[0-9a-f]{40}", w.get("revision") or ""),
          "weights pointer is openvla/openvla-7b at a 40-character revision")
    check((man.get("mesh") or {}).get("topology") == "P300" and man.get("device_count") == 2,
          "mesh is P300 with 2 devices")
    check(man.get("entrypoint") in (None, {}), "no vLLM entrypoint")
    check(not (bundle / "vllm_models").exists(), "no vllm_models/ directory")

    # --- requirements / constraints ---------------------------------------------------
    active = [ln.split("#", 1)[0].strip() for ln in reqs.splitlines()]
    active = [ln for ln in active if ln]
    check(not [ln for ln in active if ln.startswith("-")], "requirements.txt has no option lines (no index options)")
    check(any(re.fullmatch(r"ttnn==\d+(\.\d+)*", ln) for ln in active), "requirements.txt pins a stock ttnn (no +local tag)")
    check(not any("+" in ln and ln.startswith("ttnn") for ln in active), "ttnn pin carries no local version")
    check((bundle / "constraints.txt").is_file(), "constraints.txt is staged")
    check("--constraint" in install_sh or "--constraints" in install_sh or "TT_MODEL_CONSTRAINTS_ARGS" in install_sh,
          "install.sh applies the constraints file")

    # --- install.sh -------------------------------------------------------------------
    check("uv pip check" in install_sh, "install.sh runs uv pip check")
    n_guards = len(re.findall(r'"\$VENV/bin/python" -c ', install_sh))
    check(n_guards >= 4, f"install.sh runs the --verify guards (found {n_guards})")
    check(install_sh.rfind("uv pip check") < install_sh.find("-c ", install_sh.rfind("uv pip check")),
          "the guards run after uv pip check, so after the venv is complete")

    # --- run.sh -----------------------------------------------------------------------
    cmd_line = next((ln for ln in run_sh.splitlines() if ln.startswith("CMD=(")), "")
    argv = subprocess.run(
        ["bash", "-c", f'PYBIN=PY PORT=PORT; set --; {cmd_line}; printf "%s\\n" "${{CMD[@]}}"'],
        capture_output=True, text=True, check=False,
    ).stdout.splitlines()
    want = ["PY", "-m", "uvicorn", "--host", "0.0.0.0", "--port", "PORT", "--lifespan", "on", "gradio_app.asgi:app"]
    check(argv == want, f"run.sh CMD expands to the uvicorn argv, one token per argument (got {argv})")
    pp = next((ln for ln in run_sh.splitlines() if ln.startswith("export PYTHONPATH=")), "")
    check(pp.startswith('export PYTHONPATH="$HERE/extra_code:'), f"PYTHONPATH starts with extra_code/ (got {pp!r})")
    check("export TT_MODEL_WEIGHTS_REVISION=" in run_sh and "export HF_MODEL=" in run_sh,
          "run.sh exports HF_MODEL and TT_MODEL_WEIGHTS_REVISION")
    check("VLLM_PLUGINS" not in "\n".join(ln for ln in run_sh.splitlines() if not ln.lstrip().startswith("#")),
          "run.sh does not set VLLM_PLUGINS")
    check('MESH_DEVICE="${MESH_DEVICE:-P300}"' in run_sh and "NCHIPS=2" in run_sh, "run.sh sets MESH_DEVICE=P300 and NCHIPS=2")

    # --- wheels -----------------------------------------------------------------------
    found = sorted(p.name for p in (bundle / "wheels").glob("*")) if (bundle / "wheels").is_dir() else []
    check(len(found) == 1 and found[0].startswith(wheel_prefix) and found[0].endswith(".whl"),
          f"wheels/ holds only the runtime-deps wheel (got {found})")
    if found:
        with zipfile.ZipFile(bundle / "wheels" / found[0]) as z:
            names = z.namelist()
            meta = z.read(next(n for n in names if n.endswith("METADATA"))).decode()
        check(not [n for n in names if n.endswith(".py") and "dist-info" not in n and n.count("/") > 1],
              "the runtime-deps wheel carries no model code")
        norm = lambda spec: re.sub(r"[-_.]+", "-", re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", spec.strip()).group(0).lower())
        got = {norm(m) for m in re.findall(r"^Requires-Dist: (.+)$", meta, re.M)}
        want_deps = {norm(ln.split("#", 1)[0]) for ln in (thin / "runtime-deps.in").read_text().splitlines()
                     if ln.split("#", 1)[0].strip()}
        check(got == want_deps, f"wheel Requires-Dist matches runtime-deps.in (diff {sorted(got ^ want_deps)})")

    # --- extra_code -------------------------------------------------------------------
    have = {str(p.relative_to(bundle / "extra_code")) for p in (bundle / "extra_code").rglob("*") if p.is_file()}
    expect = set(read_list(thin / "served_files.txt")) | set(read_list(thin / "ttt_closure.txt"))
    check(have == expect, f"extra_code/ holds exactly the served files plus the closure (diff {sorted(have ^ expect)[:6]})")
    check(not (bundle / "extra_code" / "models" / "__init__.py").exists(),
          "extra_code/models is a namespace package, as in tt-metal")
    return results


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    prefix = sys.argv[sys.argv.index("--wheel-name-prefix") + 1] if "--wheel-name-prefix" in sys.argv else "tt_openvla_runtime_deps"
    res = inspect(Path(sys.argv[1]), prefix)
    for ok, msg in res:
        print(("ok    " if ok else "FAIL  ") + msg)
    bad = [m for ok, m in res if not ok]
    print(f"\n{len(res) - len(bad)}/{len(res)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
