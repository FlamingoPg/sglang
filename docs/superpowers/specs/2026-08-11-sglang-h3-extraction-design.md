# SGLang MiniMax-H3 Extraction Design

Date: 2026-08-11
Status: approved-for-spec-review

## 1. Goal

Create a **new, standalone repository** for MiniMax-H3 diffusion serving. The new repository is the source of truth for H3 deployment and future H3 optimization. It is bootstrapped from SGLang history, but after extraction it must **not** depend on, import from, or reuse the current `sglang` repository at runtime.

The first product is an installable package that can run the current verified MiniMax-H3 serving path on our H20/H100-class machines, with a clean boundary for continued optimization.

## 2. Non-goals for v1

- No full `sglang` diffusion ecosystem migration. Only MiniMax-H3 is kept.
- No runtime dependency on `sglang`, `sglang.srt`, or `sglang.multimodal_gen`.
- No production guarantee for FSDP, breakable CUDA graph, `torch.compile`, FP8, or Cache-DiT. These are optimization backlog items, not v1 commitments.
- No 2K output in v1. Current SGLang H3 release policy is locked to `target.short_edge == 768`; 2K requires a separate shape-policy change and validation.
- No wholesale migration of `python/sglang/kernels` or `python/sglang/srt`. Only the transitive closure required by H3 is vendored.

## 3. Key decisions

### 3.1 New repository, not continued reuse of SGLang

- Bootstrap: use `git filter-repo` to extract history for `python/sglang/multimodal_gen` into a new repo, then make an H3-only pruning commit.
- After the pruning commit, the current SGLang repo becomes a **read-only reference**. Upstream fixes are imported only by manual cherry-pick/review.
- Hard rule: the new package must not import `sglang.*` or `sglang.multimodal_gen.*`. CI enforces this with an import-linter.

### 3.2 Package and import boundary

- Package name: `sglang_h3`.
- Import rewrite:
  - `sglang.multimodal_gen.*` -> `sglang_h3.*`
  - `sglang.srt.*` used by H3 -> `sglang_h3.runtime_core.*`
  - `sglang.kernels.*` used by H3 -> `sglang_h3.kernels.*`
  - native ops stay external via the `sgl_kernel` wheel (`sglang-kernel`), e.g. the verified `0.4.5+cu129` build for H20.
- `runtime_core/` vendors only pure-Python utilities required by H3: environment flags, trace hooks, minimal distributed world-group state, JSON/network response helpers, small utils. It does **not** vendor LLM scheduler, model executor, FSDP/deep_gemm, or compiled kernels.
- `kernels/` vendors only the Python kernel wrappers/JIT reachable from the H3 closure, e.g. qknorm, flash/varlen attention helpers, diffusion triton scale/shift, and required quantization helpers.

### 3.3 Kept serving surface

Keep only what is needed to serve MiniMax-H3 end to end:

- CLI: `serve` / `generate` H3 path.
- HTTP: async OpenAI-compatible `/v1/videos` plus common health/model listing pieces needed by clients.
- Runtime: scheduler, managers, gpu worker, pipeline executor required by H3.
- Stages: shared pipeline stages plus `model_specific_stages/minimax_h3/`.
- Models/components: MiniMax-H3 DiT, Qwen text encoder path, video/audio VAE path, media IO, request validation, target geometry, warmup, ffprobe final-output validation.
- Native dependency: `sglang-kernel` wheel; system tools: `ffmpeg/ffprobe`, `ninja` for JIT.

### 3.4 v1 supported matrix

- Hardware: H20/H100-class NVIDIA GPUs, resident placement.
- Topology: 8x Ulysses8 and 4x TP2+Ulysses2, matching the two configurations already verified on the H20 devbox.
- API: T2VA first; FL2VA/Ref2VA kept only if they remain inside the same closure and pass smoke.
- Output: 768 short-edge policy only; H.264 24 fps + AAC 32 kHz stereo MP4 validation.

## 4. Migration plan

1. **Baseline freeze**: choose the SGLang 0.5.17 H3 code point that was verified on H20. Create `h3-extract` branch.
2. **History extraction**: `git filter-repo` extracts `python/sglang/multimodal_gen` history into the new repository.
3. **Closure computation**: from the H3 entrypoints (`serve`, `/v1/videos`, H3 registry), compute the transitive import closure. Keep only closure modules from `multimodal_gen`, plus the needed subset of `python/sglang/kernels` wrappers.
4. **H3-only prune**: delete all non-closure models, apps, benchmarks, and tests. Keep H3 unit tests and a small smoke suite.
5. **Import rewrite**: apply the package rewrite in section 3.2 and vendor `runtime_core`.
6. **Boundary CI**: add an import-linter rule that fails on `sglang.srt`, `sglang.multimodal_gen`, or any unvendored `sglang.kernels` import.
7. **Packaging**: add `pyproject.toml`, package data, README with system dependencies and HDFS/fuse model-path conventions.
8. **Parity verification**: run the H20 parity suite and compare against recorded baselines.

## 5. Verification

Required before calling v1 done:

- `pip install .` succeeds in a clean venv.
- Import-linter passes: no `sglang.srt`, no `sglang.multimodal_gen`.
- On the H20 devbox:
  - 4x TP2+Ulysses2 server starts and reports `/v1/models`.
  - 8x Ulysses8 server starts and reports `/v1/models`.
  - A 5s / 50-step T2VA request returns `status=completed`.
  - ffprobe validates H.264 `1344x768@24fps` and AAC stereo `32kHz`.
- Record latency and peak memory for both topologies and compare to the current baselines: 8x H20 about 161 s, 4x H20 about 321 s for the same request shape.

## 6. Risks and mitigations

- **Kernel wrapper closure is larger than expected.** `python/sglang/kernels` is large and coupled to `srt`. Mitigate by computing the closure from entrypoints and deleting JIT/wrapper paths not used by H3 resident serving.
- **Distributed state sync with LLM `srt` world group.** Remove `_sync_srt_*` behavior; use a standalone H3 world group.
- **Model loading environment.** H3 weights live on HDFS/fuse paths; document required mounts and keep RunAI streamer behavior inside the vendored loader path.
- **Rename collisions.** Do a single mechanical rewrite commit, then a separate behavior commit; never mix rename and feature changes.
- **Upstream drift.** After extraction, upstream SGLang is reference-only. Fixes come in through reviewed cherry-picks, not implicit reuse.

## 7. Milestones

- M0: extraction + rename + clean install.
- M1: H3-only prune + `runtime_core` + import-linter green.
- M2: H20 parity for 4x/8x serving and smoke request.
- M3: optimization backlog: prove/fix Cache-DiT on the H3 native path, evaluate 2K shape-policy unlock, FP8 validation against BF16, H20 kernel tuning.
