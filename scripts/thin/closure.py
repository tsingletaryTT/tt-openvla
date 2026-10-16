# SPDX-License-Identifier: MIT
"""Static `models.*` import closure for the thin bundle. Standard library only.

The LLaMA backbone reuses tt-metal's `models.tt_transformers.tt` (Transformer, ModelArgs)
and `models.common`. The bundle ships only the files that the served code can reach,
read from the AST of the served files and then of each file they pull in. Importing
nothing here executes any of that code.

    python3 -I closure.py derive <tt-metal-src-root> <served-files.txt>   # prints the list
    python3 -I closure.py check  <tt-metal-src-root> <served-files.txt> <closure.txt>

A `models` package without `__init__.py` is a namespace package, which is how tt-metal
ships it; `__init__.py` files that do exist along a path are included. Data files a
closure module names by a relative string literal (a YAML beside it) are included too.
"""

import ast
import sys
from pathlib import Path


def _pkg_of(rel: str) -> list[str]:
    return rel.split("/")[:-1]


def _imported_modules(tree: ast.AST, pkg: list[str]) -> set[str]:
    """Absolute dotted module names (and `from m import name` candidates) imported anywhere in `tree`."""
    mods: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                base = pkg[: len(pkg) - (n.level - 1)] if n.level > 1 else list(pkg)
                m = ".".join(base + ([n.module] if n.module else []))
            else:
                m = n.module or ""
            if m:
                mods.add(m)
                mods.update(f"{m}.{a.name}" for a in n.names)
    return mods


def dynamic_imports(tree: ast.AST) -> list[int]:
    """Line numbers of importlib.import_module / __import__ calls: the AST walk cannot follow them."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func
            if getattr(fn, "attr", getattr(fn, "id", "")) in ("import_module", "__import__"):
                out.append(n.lineno)
    return out


def served_entry_modules(repo: Path, served_files: list[str]) -> set[str]:
    """Every `models.*` module the served files import."""
    out: set[str] = set()
    for rel in served_files:
        if not rel.endswith(".py"):
            continue
        tree = ast.parse((repo / rel).read_text())
        out |= {m for m in _imported_modules(tree, _pkg_of(rel)) if m == "models" or m.startswith("models.")}
    return out


def derive(src: Path, entries: set[str]) -> tuple[list[str], list[tuple[str, int]]]:
    """(sorted file list, dynamic-import sites) for the closure of `entries` under `src`."""
    files = {str(p.relative_to(src)) for p in src.rglob("*.py")}

    def files_for(mod: str) -> list[str]:
        p = mod.replace(".", "/")
        hits = [f for f in (p + ".py", p + "/__init__.py") if f in files]
        parts = p.split("/")
        hits += [q for i in range(1, len(parts)) if (q := "/".join(parts[:i]) + "/__init__.py") in files]
        return hits

    seen: set[str] = set()
    stack: list[str] = []

    def add(mod: str) -> None:
        for f in files_for(mod):
            if f not in seen:
                seen.add(f)
                stack.append(f)

    for e in sorted(entries):
        add(e)
    dyn: list[tuple[str, int]] = []
    data: set[str] = set()
    while stack:
        f = stack.pop()
        tree = ast.parse((src / f).read_text())
        dyn += [(f, ln) for ln in dynamic_imports(tree)]
        data |= data_files(src, f, tree)
        for m in _imported_modules(tree, _pkg_of(f)):
            if m == "models" or m.startswith("models."):
                add(m)
    return sorted(seen | data), dyn


def data_files(src: Path, rel: str, tree: ast.AST) -> set[str]:
    """Data files a module reads next to itself.

    tt-metal modules open files with `Path(__file__).parent / "sub/name.yaml"`. An import walk
    cannot see those, and a bundle without them fails at import time. A string literal that
    ends in a data extension and names an existing file beside the module counts.
    """
    here = (src / rel).parent
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith((".yaml", ".yml", ".json", ".txt")):
            cand = here / n.value
            if cand.is_file():
                out.add(str(cand.relative_to(src)))
    return out


def read_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[1] not in ("derive", "check"):
        print(__doc__)
        return 2
    src = Path(argv[2]).resolve()
    served = read_list(Path(argv[3]))
    repo = Path(__file__).resolve().parents[2]
    files, dyn = derive(src, served_entry_modules(repo, served))
    if dyn:
        print(f"closure.py: dynamic imports the AST cannot follow: {dyn}", file=sys.stderr)
        return 1
    if argv[1] == "derive":
        print("\n".join(files))
        return 0
    want = read_list(Path(argv[4]))
    if want != files:
        print("closure.py: committed closure differs from the derived one.", file=sys.stderr)
        print(f"  only committed: {sorted(set(want) - set(files))}", file=sys.stderr)
        print(f"  only derived:   {sorted(set(files) - set(want))}", file=sys.stderr)
        return 1
    print(f"closure OK: {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
