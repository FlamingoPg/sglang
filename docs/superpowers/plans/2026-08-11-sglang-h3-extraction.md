# SGLang-H3 Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone `sglang-h3` repository that serves MiniMax-H3 without runtime dependence on the current `sglang` repo.

**Architecture:** Bootstrap a new repo from `python/sglang/multimodal_gen` history, prune to the H3 transitive closure, rewrite imports to `sglang_h3`, rewrite the small pure-Python runtime surface, extract only H3-needed kernel wrappers, and verify H20 parity for 4x/8x serving.

**Tech Stack:** Python 3.12, PyTorch cu129/cu130 wheels, `sglang-kernel` native wheel, FastAPI/Uvicorn, git filter-repo, pytest, ruff/import-linter, ffmpeg/ffprobe, ninja.

## Global Constraints

- New repo path: `/Users/bytedance/dev/git/sglang-h3`. It is the H3 source of truth; current SGLang is read-only reference after extraction.
- Package/import root: `sglang_h3`. No `sglang.srt` or `sglang.multimodal_gen` imports are allowed anywhere in the new repo.
- Native kernels stay external: dependency is `sglang-kernel` (module `sgl_kernel`), verified baseline `sglang-kernel==0.4.5+cu129` for H20.
- Kernel policy: extract only H3-needed Python kernel wrappers/JIT into `sglang_h3/kernels/`; rewrite other call sites instead of preserving unrelated kernel abstraction.
- v1 supported matrix: H20/H100-class NVIDIA, resident placement, 8x Ulysses8 and 4x TP2+Ulysses2, async `/v1/videos`, T2VA smoke, `target.short_edge == 768` only.
- System runtime deps: `ffmpeg` + `ffprobe` on `PATH`, `ninja` on `PATH` for JIT, HDFS/fuse model path readable at `/mnt/hdfs/_BYTE_DATA_SEED_WL_/user/yinfan.1024/os-model/MiniMax-H3`.
- Parity baselines to record: 8x H20 about 161 s, 4x H20 about 321 s for the 5 s / 50-step T2VA request; ffprobe must show H.264 `1344x768@24fps` and AAC stereo `32kHz`.
- Rename and behavior changes are separate commits. Never mix mechanical rename with feature edits.

---

## File Structure

- `pyproject.toml` — package metadata for `sglang_h3`; declares deps and import-linter/ruff config.
- `README.md` — runbook: install, system deps, HDFS model path, serve commands, smoke request, supported matrix.
- `sglang_h3/__init__.py` — version and public API surface (`__version__`).
- `sglang_h3/runtime_core/` — rewritten pure-Python runtime surface: `environ.py`, `trace.py`, `distributed.py`, `http.py`, `platforms.py`, `utils.py`.
- `sglang_h3/kernels/` — extracted H3-needed Python kernel wrappers only.
- `sglang_h3/multimodal_gen/` — H3-pruned serving code copied from SGLang `python/sglang/multimodal_gen` and renamed.
- `tools/list_h3_closure.py` — AST import-closure inventory used to decide what is copied.
- `tools/rewrite_imports.py` — deterministic prefix rewriter for `sglang.multimodal_gen`, `sglang.srt`, `sglang.kernels`.
- `tools/prune_to_h3.py` — copies closure files from a source SGLang checkout into the new repo layout.
- `tests/test_import_boundary.py` — fails on forbidden `sglang.srt`/`sglang.multimodal_gen` imports.
- `tests/test_request_validation.py` — H3 target validation: accepts 768, rejects 1440 and unsupported upscale flags.
- `tests/test_runtime_core.py` — unit tests for rewritten runtime helpers.
- `tests/test_rewrite_imports.py` — unit tests for the rewrite mapping.
- `scripts/serve_h3_4gpu.sh`, `scripts/serve_h3_8gpu.sh` — H20 serve launchers used for parity.
- `scripts/smoke_t2va.py` — posts a 5 s / 50-step T2VA request and validates completion.

---

### Task 1: Create the standalone repo from filtered history

**Files:**
- Create: `/Users/bytedance/dev/git/sglang-h3/` (new git repo)
- Modify: none in current SGLang
- Test: `git -C /Users/bytedance/dev/git/sglang-h3 log --oneline | head`

**Interfaces:**
- Consumes: current SGLang checkout at `/Users/bytedance/dev/git/sglang`.
- Produces: a git repo whose tree contains only `python/sglang/multimodal_gen` history at repo root under `upstream/multimodal_gen/`.

- [ ] **Step 1: Clone a local source for filtering**

```bash
git clone --no-local /Users/bytedance/dev/git/sglang /tmp/sglang-h3-source
cd /tmp/sglang-h3-source
git switch -c h3-extract-baseline
```

- [ ] **Step 2: Install git-filter-repo if missing**

Run: `python3 -m pip show git-filter-repo || python3 -m pip install --user git-filter-repo`
Expected: `git filter-repo --version` prints a version.

- [ ] **Step 3: Filter history to multimodal_gen only**

```bash
cd /tmp/sglang-h3-source
git filter-repo --path python/sglang/multimodal_gen --path-rename python/sglang/multimodal_gen:upstream/multimodal_gen --force
```

- [ ] **Step 4: Create the new repo and import filtered history**

```bash
git init /Users/bytedance/dev/git/sglang-h3
cd /Users/bytedance/dev/git/sglang-h3
git remote add source /tmp/sglang-h3-source
git fetch source
git switch -c main FETCH_HEAD
```

- [ ] **Step 5: Verify history exists and repo is standalone**

Run: `git log --oneline -- upstream/multimodal_gen | head -5` and `git remote -v`
Expected: at least one historical commit touching `upstream/multimodal_gen`; remote `source` points to `/tmp/sglang-h3-source`.

- [ ] **Step 6: Commit the import marker**

```bash
cd /Users/bytedance/dev/git/sglang-h3
git commit --allow-empty -m "chore: mark filtered multimodal_gen history import"
```

---

### Task 2: Add package skeleton and forbidden-import gate

**Files:**
- Create: `pyproject.toml`
- Create: `sglang_h3/__init__.py`
- Create: `tests/test_import_boundary.py`
- Test: `tests/test_import_boundary.py`

**Interfaces:**
- Consumes: repo from Task 1.
- Produces: `sglang_h3.__version__ == "0.1.0"` and a boundary test that scans tracked `*.py` for forbidden imports.

- [ ] **Step 1: Write package metadata**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=80", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sglang-h3"
version = "0.1.0"
description = "Standalone MiniMax-H3 diffusion serving extracted from SGLang."
requires-python = ">=3.10"
dependencies = [
  "fastapi",
  "uvicorn",
  "pydantic",
  "pillow",
  "numpy",
  "torch",
  "diffusers",
  "transformers",
  "runai-model-streamer[s3,gcs,azure]>=0.15.7",
  "sglang-kernel",
]

[tool.setuptools.packages.find]
include = ["sglang_h3*"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Write version module**

```python
# sglang_h3/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 3: Write the failing boundary test**

```python
# tests/test_import_boundary.py
import ast
import subprocess
from pathlib import Path

FORBIDDEN = ("sglang.srt", "sglang.multimodal_gen")

def tracked_py_files():
    out = subprocess.check_output(["git", "ls-files", "*.py"], text=True)
    return [Path(p) for p in out.splitlines() if p.strip()]

def test_no_forbidden_sglang_imports():
    bad = []
    for path in tracked_py_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.startswith(FORBIDDEN):
                    bad.append((str(path), name))
    assert bad == []
```

- [ ] **Step 4: Run boundary test to verify current failure**

Run: `pytest tests/test_import_boundary.py -v`
Expected: FAIL because `upstream/multimodal_gen` still contains `sglang.srt` / `sglang.multimodal_gen` imports.

- [ ] **Step 5: Install package editable and verify import**

Run: `python3 -m pip install -e . && python3 -c "import sglang_h3; print(sglang_h3.__version__)"`
Expected: prints `0.1.0`.

- [ ] **Step 6: Commit skeleton**

```bash
git add pyproject.toml sglang_h3/__init__.py tests/test_import_boundary.py
git commit -m "feat: add sglang_h3 package skeleton and import boundary test"
```

---

### Task 3: Build the H3 import-closure inventory

**Files:**
- Create: `tools/list_h3_closure.py`
- Create: `tests/test_h3_closure.py`
- Test: `tests/test_h3_closure.py`

**Interfaces:**
- Consumes: source tree `upstream/multimodal_gen` from Task 1.
- Produces: `build/h3_closure.txt`, a newline list of repo-relative source files that must be copied by Task 5.

- [ ] **Step 1: Write closure tool**

```python
# tools/list_h3_closure.py
import ast
import sys
from pathlib import Path

ROOT = Path("upstream/multimodal_gen")
SRC_ROOT = Path("/tmp/sglang-h3-source")
ENTRY = [
  ROOT / "runtime/entrypoints/cli/serve.py",
  ROOT / "runtime/entrypoints/cli/generate.py",
  ROOT / "registry.py",
  ROOT / "runtime/entrypoints/http_server.py",
]
PREFIXES = ("sglang.multimodal_gen", "sglang.kernels")

def module_to_path(module: str) -> Path | None:
    if module.startswith("sglang.multimodal_gen"):
        rel = module.removeprefix("sglang.multimodal_gen").replace(".", "/")
        return ROOT / (rel + ".py")
    if module.startswith("sglang.kernels"):
        rel = module.removeprefix("sglang.kernels").replace(".", "/")
        return SRC_ROOT / "python/sglang/kernels" / (rel + ".py")
    return None

def imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        for mod in mods:
            if not mod.startswith(PREFIXES):
                continue
            target = module_to_path(mod)
            if target and target.exists():
                out.add(target)
    return out

def main() -> int:
    seen = set()
    stack = [p for p in ENTRY if p.exists()]
    while stack:
        path = stack.pop()
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        stack.extend(imports(path))
    Path("build").mkdir(exist_ok=True)
    Path("build/h3_closure.txt").write_text("\n".join(sorted(str(p) for p in seen)) + "\n")
    print(f"closure files: {len(seen)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write closure test**

```python
# tests/test_h3_closure.py
from pathlib import Path

def test_h3_closure_contains_h3_and_excludes_other_models():
    text = Path("build/h3_closure.txt").read_text()
    assert "model_specific_stages/minimax_h3" in text
    assert "runtime/entrypoints/cli/serve.py" in text
    forbidden = ["model_specific_stages/cosmos3", "model_specific_stages/ideogram", "apps/webui"]
    for item in forbidden:
        assert item not in text
```

- [ ] **Step 3: Run closure tool**

Run: `python3 tools/list_h3_closure.py`
Expected: prints `closure files: N` and writes `build/h3_closure.txt`.

- [ ] **Step 4: Run closure test**

Run: `pytest tests/test_h3_closure.py -v`
Expected: PASS. If a forbidden path appears, remove it from `ENTRY` or add a targeted exclude rule in `module_to_path` and rerun Step 3.

- [ ] **Step 5: Commit closure tooling**

```bash
git add tools/list_h3_closure.py tests/test_h3_closure.py build/h3_closure.txt
git commit -m "feat: compute MiniMax-H3 import closure"
```

---

### Task 4: Add deterministic import rewrite tooling

**Files:**
- Create: `tools/rewrite_imports.py`
- Create: `tests/test_rewrite_imports.py`
- Test: `tests/test_rewrite_imports.py`

**Interfaces:**
- Consumes: file list from Task 3.
- Produces: rewrite mappings used by Task 5:
  - `sglang.multimodal_gen` -> `sglang_h3.multimodal_gen`
  - `sglang.kernels` -> `sglang_h3.kernels`
  - `sglang.srt` -> `sglang_h3.runtime_core`

- [ ] **Step 1: Write rewrite tool**

```python
# tools/rewrite_imports.py
import argparse
from pathlib import Path

MAPPING = (
    ("sglang.multimodal_gen", "sglang_h3.multimodal_gen"),
    ("sglang.kernels", "sglang_h3.kernels"),
    ("sglang.srt", "sglang_h3.runtime_core"),
)

def rewrite_text(text: str) -> str:
    for old, new in MAPPING:
        text = text.replace(old, new)
    return text

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = 0
    for raw in args.paths:
        path = Path(raw)
        old = path.read_text()
        new = rewrite_text(old)
        if new != old:
            changed += 1
            if args.check:
                print(f"would rewrite: {path}")
            else:
                path.write_text(new)
    if args.check and changed:
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write rewrite tests**

```python
# tests/test_rewrite_imports.py
from tools.rewrite_imports import rewrite_text

def test_rewrite_prefixes():
    src = "from sglang.srt.environ import envs\nimport sglang.kernels.ops.x as x\nfrom sglang.multimodal_gen.registry import r\n"
    out = rewrite_text(src)
    assert "sglang_h3.runtime_core.environ" in out
    assert "sglang_h3.kernels.ops.x" in out
    assert "sglang_h3.multimodal_gen.registry" in out
    assert "sglang.srt" not in out
```

- [ ] **Step 3: Run rewrite tests**

Run: `pytest tests/test_rewrite_imports.py -v`
Expected: PASS.

- [ ] **Step 4: Commit rewrite tooling**

```bash
git add tools/rewrite_imports.py tests/test_rewrite_imports.py
git commit -m "feat: add deterministic sglang_h3 import rewriter"
```

---

### Task 5: Copy the H3 closure into the new package and prune everything else

**Files:**
- Create: `tools/prune_to_h3.py`
- Modify: repo tree — creates `sglang_h3/multimodal_gen/**` and `sglang_h3/kernels/**`
- Test: `tests/test_import_boundary.py`, `tests/test_pruned_tree.py`

**Interfaces:**
- Consumes: `build/h3_closure.txt` from Task 3 and `tools/rewrite_imports.py` from Task 4.
- Produces: a pruned package where every tracked `.py` passes the boundary test and H3 paths exist.

- [ ] **Step 1: Write prune/copy tool**

```python
# tools/prune_to_h3.py
import shutil
from pathlib import Path

SRC_ROOT = Path("/tmp/sglang-h3-source")
DST_ROOT = Path("/Users/bytedance/dev/git/sglang-h3")

def target_for(src: Path) -> Path:
    s = str(src)
    if "upstream/multimodal_gen" in s:
        rel = s.split("upstream/multimodal_gen/", 1)[1]
        return DST_ROOT / "sglang_h3/multimodal_gen" / rel
    if "python/sglang/kernels" in s:
        rel = s.split("python/sglang/kernels/", 1)[1]
        return DST_ROOT / "sglang_h3/kernels" / rel
    raise ValueError(f"unmapped source: {src}")

def main() -> int:
    kept = [Path(line) for line in Path("build/h3_closure.txt").read_text().splitlines() if line.strip()]
    for src in kept:
        dst = target_for(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"copied {len(kept)} files")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write pruned-tree test**

```python
# tests/test_pruned_tree.py
from pathlib import Path

def test_h3_tree_exists_and_unrelated_models_absent():
    assert Path("sglang_h3/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3").is_dir()
    assert Path("sglang_h3/multimodal_gen/runtime/entrypoints/cli/serve.py").is_file()
    assert not Path("sglang_h3/multimodal_gen/apps").exists()
    assert not Path("sglang_h3/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3.py").exists()
```

- [ ] **Step 3: Run prune/copy tool**

Run: `python3 tools/prune_to_h3.py`
Expected: prints `copied N files` and creates `sglang_h3/multimodal_gen` plus `sglang_h3/kernels`.

- [ ] **Step 4: Apply import rewrite to copied files**

```bash
git ls-files 'sglang_h3/*.py' | tr '\n' ' ' > build/h3_files.txt
python3 tools/rewrite_imports.py $(cat build/h3_files.txt)
python3 tools/rewrite_imports.py --check $(cat build/h3_files.txt) || true
```

Expected: second command prints no `would rewrite` lines after the first rewrite.

- [ ] **Step 5: Run boundary and pruned-tree tests**

Run: `pytest tests/test_import_boundary.py tests/test_pruned_tree.py -v`
Expected: boundary FAIL is now limited to modules that belong in `runtime_core`; pruned-tree PASS. Record the failing import list for Task 6.

- [ ] **Step 6: Commit copied tree before runtime rewrite**

```bash
git add sglang_h3 tools/prune_to_h3.py tests/test_pruned_tree.py
git commit -m "feat: copy H3 closure into sglang_h3 package"
```

---

### Task 6: Rewrite the minimal runtime_core surface

**Files:**
- Create: `sglang_h3/runtime_core/__init__.py`
- Create: `sglang_h3/runtime_core/environ.py`
- Create: `sglang_h3/runtime_core/trace.py`
- Create: `sglang_h3/runtime_core/distributed.py`
- Create: `sglang_h3/runtime_core/http.py`
- Create: `sglang_h3/runtime_core/platforms.py`
- Create: `sglang_h3/runtime_core/utils.py`
- Test: `tests/test_runtime_core.py`, `tests/test_import_boundary.py`

**Interfaces:**
- Consumes: failing import list from Task 5 Step 5.
- Produces these exact names used by copied H3 code:
  - `runtime_core.environ.envs` with `.get()` returning env-backed values.
  - `runtime_core.trace.TraceReqContext` and `extract_trace_headers(headers)`.
  - `runtime_core.distributed.init_world_group(world_size, rank)` returning a process-group-like object with `rank()`, `world_size()`, `barrier()`.
  - `runtime_core.http.orjson_response(data, status_code=200)`.
  - `runtime_core.platforms.is_cuda()`, `get_cuda_version()`, `get_device_sm()`.
  - `runtime_core.utils.json_response`, `network.NetworkAddress`, `logging_utils.init_logger`, `precision` helpers only if present in the failing import list.

- [ ] **Step 1: Write failing runtime_core tests**

```python
# tests/test_runtime_core.py
import os
from sglang_h3.runtime_core.environ import envs
from sglang_h3.runtime_core.distributed import init_world_group
from sglang_h3.runtime_core.http import orjson_response

def test_envs_reads_process_env(monkeypatch):
    monkeypatch.setenv("SGLANG_H3_TEST_FLAG", "1")
    assert envs.SGLANG_H3_TEST_FLAG.get() is True

def test_world_group_single_rank():
    group = init_world_group(world_size=1, rank=0)
    assert group.rank() == 0
    assert group.world_size() == 1
    group.barrier()

def test_orjson_response_bytes():
    resp = orjson_response({"ok": True})
    assert resp.status_code == 200
    assert b'"ok":true' in resp.body
```

- [ ] **Step 2: Run runtime_core tests to verify they fail**

Run: `pytest tests/test_runtime_core.py -v`
Expected: FAIL with `ModuleNotFoundError: sglang_h3.runtime_core`.

- [ ] **Step 3: Implement runtime_core modules**

Write these modules with only the behavior asserted above plus the names from the Task 5 failing import list:

```python
# sglang_h3/runtime_core/environ.py
import os
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class EnvVar:
    name: str
    default: Any = None
    def get(self):
        value = os.environ.get(self.name)
        if value is None:
            return self.default
        if isinstance(self.default, bool):
            return value.lower() in ("1", "true", "yes", "on")
        return value

class _Envs:
    def __getattr__(self, name: str) -> EnvVar:
        return EnvVar(name)

envs = _Envs()
```

```python
# sglang_h3/runtime_core/distributed.py
from dataclasses import dataclass

@dataclass
class WorldGroup:
    _world_size: int
    _rank: int
    def rank(self) -> int: return self._rank
    def world_size(self) -> int: return self._world_size
    def barrier(self) -> None: return None

def init_world_group(world_size: int, rank: int) -> WorldGroup:
    return WorldGroup(world_size, rank)
```

```python
# sglang_h3/runtime_core/http.py
import json
from fastapi.responses import Response

def orjson_response(data, status_code: int = 200) -> Response:
    return Response(
        content=json.dumps(data, separators=(",", ":"), ensure_ascii=False),
        status_code=status_code,
        media_type="application/json",
    )
```

Implement `trace.py`, `platforms.py`, and `utils.py` with the smallest functions/classes that satisfy the Task 5 import errors; every added name must be referenced by an import error or a test.

- [ ] **Step 4: Run runtime_core and boundary tests**

Run: `pytest tests/test_runtime_core.py tests/test_import_boundary.py -v`
Expected: runtime_core PASS; boundary PASS or a shorter concrete missing-name list. If a copied H3 file imports a name not needed at runtime for the v1 matrix, delete that import and the unreachable code path in the same commit.

- [ ] **Step 5: Run H3 import smoke**

Run: `python3 -c "import sglang_h3.multimodal_gen.registry as r; print('registry import ok')"`
Expected: prints `registry import ok`.

- [ ] **Step 6: Commit runtime rewrite**

```bash
git add sglang_h3/runtime_core tests/test_runtime_core.py sglang_h3/multimodal_gen sglang_h3/kernels
git commit -m "feat: rewrite standalone runtime_core for H3 serving"
```

---

### Task 7: Make CLI and request validation pass without SGLang

**Files:**
- Modify: `sglang_h3/multimodal_gen/runtime/entrypoints/cli/serve.py`
- Modify: `sglang_h3/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py`
- Test: `tests/test_request_validation.py`, `tests/test_cli_help.py`

**Interfaces:**
- Consumes: Tasks 5-6 package tree.
- Produces:
  - `sglang_h3 serve --help` exits 0 and prints `--model-variant`.
  - `validate_request(target={"short_edge": 768, ...})` accepts; `short_edge=1440` raises `ValueError("target.short_edge must be 768")`.

- [ ] **Step 1: Write validation and CLI tests**

```python
# tests/test_request_validation.py
import pytest
from sglang_h3.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import _validate_target

class Profile:
    task = "t2va"
    aspect_ratio_forced_auto = False

def test_target_768_accepted():
    out = _validate_target({"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0}, profile=Profile())
    assert out["short_edge"] == 768

def test_target_1440_rejected():
    with pytest.raises(ValueError, match="target.short_edge must be 768"):
        _validate_target({"short_edge": 1440, "aspect_ratio": "16:9", "duration_seconds": 5.0}, profile=Profile())
```

```python
# tests/test_cli_help.py
import subprocess

def test_serve_help_lists_model_variant():
    out = subprocess.run(["sglang_h3", "serve", "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert out.returncode == 0
    assert "--model-variant" in out.stdout
```

- [ ] **Step 2: Run tests to verify current failures**

Run: `pytest tests/test_request_validation.py tests/test_cli_help.py -v`
Expected: at least one FAIL from missing CLI entrypoint or import rewrite leftovers.

- [ ] **Step 3: Add CLI entrypoint and fix validation imports**

Add to `pyproject.toml`:

```toml
[project.scripts]
sglang_h3 = "sglang_h3.multimodal_gen.runtime.entrypoints.cli.main:main"
```

In `request_validation.py`, keep this exact check and remove imports that resolve to deleted non-H3 profiles:

```python
if short_edge != 768:
    raise ValueError(f"target.short_edge must be 768 for minimax_h3, got {short_edge}")
```

- [ ] **Step 4: Reinstall and rerun tests**

Run: `python3 -m pip install -e . && pytest tests/test_request_validation.py tests/test_cli_help.py -v`
Expected: PASS.

- [ ] **Step 5: Commit CLI/validation parity**

```bash
git add pyproject.toml sglang_h3/multimodal_gen tests/test_request_validation.py tests/test_cli_help.py
git commit -m "feat: standalone H3 CLI and 768 target validation"
```

---

### Task 8: Add H20 serve scripts and offline smoke client

**Files:**
- Create: `scripts/serve_h3_4gpu.sh`
- Create: `scripts/serve_h3_8gpu.sh`
- Create: `scripts/smoke_t2va.py`
- Test: `tests/test_smoke_script.py`

**Interfaces:**
- Consumes: Task 7 CLI.
- Produces: launchers that bind `0.0.0.0:30000`, set `PATH` for `ffprobe`/`ninja`/venv, set `HF_HUB_OFFLINE=1`, and use the HDFS model path from Global Constraints.

- [ ] **Step 1: Write serve scripts**

```bash
# scripts/serve_h3_4gpu.sh
set -euo pipefail
export PATH="/opt/tiger/ss_bin:/tmp/h3-venv/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL=/mnt/hdfs/_BYTE_DATA_SEED_WL_/user/yinfan.1024/os-model/MiniMax-H3
exec sglang_h3 serve --model-path "$MODEL" --model-variant fl2va \
  --num-gpus 4 --tp-size 2 --ulysses-degree 2 --performance-mode speed \
  --host 0.0.0.0 --port 30000
```

```bash
# scripts/serve_h3_8gpu.sh
set -euo pipefail
export PATH="/opt/tiger/ss_bin:/tmp/h3-venv/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL=/mnt/hdfs/_BYTE_DATA_SEED_WL_/user/yinfan.1024/os-model/MiniMax-H3
exec sglang_h3 serve --model-path "$MODEL" --model-variant fl2va \
  --num-gpus 8 --ulysses-degree 8 --performance-mode speed \
  --host 0.0.0.0 --port 30000
```

- [ ] **Step 2: Write smoke client**

```python
# scripts/smoke_t2va.py
import json, os, sys, time, urllib.request
BASE = os.environ.get("H3_BASE", "http://127.0.0.1:30000")
MODEL = os.environ.get("H3_MODEL", "/mnt/hdfs/_BYTE_DATA_SEED_WL_/user/yinfan.1024/os-model/MiniMax-H3")
payload = {
  "model": MODEL,
  "prompt": "A small red cube rotates on a plain table while a soft beep plays.",
  "seconds": 5,
  "task": "t2va",
  "conditions": [],
  "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
  "num_outputs_per_prompt": 1,
  "num_inference_steps": 50,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": 1101,
}
def req(method, path, data=None):
    body = None if data is None else json.dumps(data).encode()
    r = urllib.request.Request(BASE + path, data=body, method=method)
    if body is not None: r.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.loads(resp.read())
resp = req("POST", "/v1/videos", payload)
vid = resp["id"]
deadline = time.time() + 1200
while time.time() < deadline:
    st = req("GET", f"/v1/videos/{vid}")
    if st["status"] == "completed":
        print("COMPLETED", st["file_path"], st.get("inference_time_s"), st.get("peak_memory_mb"))
        sys.exit(0)
    if st["status"] == "failed":
        print("FAILED", st)
        sys.exit(1)
    time.sleep(5)
print("TIMEOUT")
sys.exit(1)
```

- [ ] **Step 3: Write script lint test**

```python
# tests/test_smoke_script.py
from pathlib import Path

def test_serve_scripts_use_h3_flags():
    four = Path("scripts/serve_h3_4gpu.sh").read_text()
    eight = Path("scripts/serve_h3_8gpu.sh").read_text()
    assert "--num-gpus 4" in four and "--tp-size 2" in four and "--ulysses-degree 2" in four
    assert "--num-gpus 8" in eight and "--ulysses-degree 8" in eight
    assert "target.short_edge" not in four
```

- [ ] **Step 4: Run script lint test**

Run: `pytest tests/test_smoke_script.py -v`
Expected: PASS.

- [ ] **Step 5: Commit scripts**

```bash
chmod +x scripts/serve_h3_4gpu.sh scripts/serve_h3_8gpu.sh scripts/smoke_t2va.py
git add scripts tests/test_smoke_script.py
git commit -m "feat: add H20 serve launchers and T2VA smoke client"
```

---

### Task 9: Verify H20 4-GPU parity and record evidence

**Files:**
- Modify: `README.md` (parity evidence section only after run)
- Test: manual H20 run recorded in `build/parity_4gpu.md`

**Interfaces:**
- Consumes: Task 8 scripts on host `llmserver-devbox-lab`.
- Produces: `build/parity_4gpu.md` containing server PID, `/v1/models` output, smoke status, ffprobe output, latency, peak memory.

- [ ] **Step 1: Install package on H20**

Run on `llmserver-devbox-lab`: `python3 -m pip install -e /Users/bytedance/dev/git/sglang-h3`
Expected: `sglang_h3 --help` exits 0.

- [ ] **Step 2: Start 4-GPU server**

Run on H20: `nohup scripts/serve_h3_4gpu.sh > /tmp/sglang_h3_4gpu.log 2>&1 & echo $!`
Expected: log eventually contains `The server is fired up and ready to roll!`; port `30000` listens.

- [ ] **Step 3: Run smoke request**

Run on H20: `python3 scripts/smoke_t2va.py`
Expected: prints `COMPLETED <file_path> <inference_time_s> <peak_memory_mb>`.

- [ ] **Step 4: Validate output media**

Run on H20: `/opt/tiger/ss_bin/ffprobe -v error -show_entries stream=codec_type,codec_name,width,height,duration,sample_rate,channels,avg_frame_rate -of json <file_path from Step 3>`
Expected: one video stream `h264`, `1344x768`, `avg_frame_rate=24/1`; one audio stream `aac`, `32000`, `channels=2`.

- [ ] **Step 5: Write parity evidence file**

Create `build/parity_4gpu.md` with: server PID, exact serve command, `/v1/models` response, smoke `inference_time_s`, `peak_memory_mb`, ffprobe JSON, and whether latency is within 15% of the 321 s baseline.

- [ ] **Step 6: Commit evidence**

```bash
git add build/parity_4gpu.md README.md
git commit -m "test: record H20 4-GPU H3 parity evidence"
```

---

### Task 10: Verify H20 8-GPU parity and finalize repository hygiene

**Files:**
- Modify: `README.md`
- Create: `build/parity_8gpu.md`
- Test: `pytest`, boundary test, H20 manual run

**Interfaces:**
- Consumes: Tasks 8-9.
- Produces: final v1 repo state with green local tests and recorded 8x parity.

- [ ] **Step 1: Start 8-GPU server**

Run on H20: stop the 4-GPU PID from Task 9, then `nohup scripts/serve_h3_8gpu.sh > /tmp/sglang_h3_8gpu.log 2>&1 & echo $!`
Expected: log contains `The server is fired up and ready to roll!`; port `30000` listens.

- [ ] **Step 2: Run smoke request**

Run on H20: `python3 scripts/smoke_t2va.py`
Expected: `COMPLETED` with latency near the 161 s baseline; record any deviation over 15% as a bug, not a note.

- [ ] **Step 3: Validate output media**

Run the same ffprobe command from Task 9 Step 4 on the 8-GPU output file.
Expected: H.264 `1344x768@24fps` and AAC stereo `32kHz`.

- [ ] **Step 4: Write 8-GPU evidence and README runbook**

Create `build/parity_8gpu.md` with the same fields as Task 9 Step 5. Update `README.md` with: install, system deps, HDFS model path, 4-GPU and 8-GPU commands, smoke command, supported matrix, and explicit non-goals from the spec.

- [ ] **Step 5: Run full local gate**

Run: `pytest && python3 tools/rewrite_imports.py --check $(git ls-files 'sglang_h3/*.py')`
Expected: pytest PASS; rewrite check prints nothing and exits 0.

- [ ] **Step 6: Commit final v1 hygiene**

```bash
git add README.md build/parity_8gpu.md
git commit -m "test: record H20 8-GPU parity and finalize v1 runbook"
```

---

## Post-v1 backlog (separate plans, not this implementation)

- Cache-DiT: reproduce the current no-speedup result on `sglang_h3`, inspect whether SCM mask reaches `minimax_h3` denoise loop, and add a latency regression test before enabling presets.
- 2K unlock: change shape policy from hardcoded 768 to an explicit validated 2K profile; expect a new request-validation task, memory benchmark, and quality comparison.
- FP8: add a BF16-vs-FP8 pairwise acceptance harness before any FP8 serving claim on H20.
- Kernel tuning: profile the 3.22 s/it H20 denoise step and decide whether H3 needs a custom attention/norm kernel or a different TP/Ulysses split.
