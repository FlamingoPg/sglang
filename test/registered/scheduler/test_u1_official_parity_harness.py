# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.sample.ug import UGSamplingParams
from sglang.srt.ug.context import (
    UGContextBundle,
    UGContextHandle,
    UGSessionHandle,
    UGSRTKVTokenBinding,
)
from sglang.srt.ug.parity import (
    UGParityArtifact,
    UGParityCase,
    UGParityDiff,
    UGParityReport,
    UGTensorSummary,
    compare_ug_parity_artifacts,
    summarize_ug_image,
    write_ug_parity_bundle,
)
from sglang.srt.ug.interleaved import UGInterleavedRequest
from sglang.srt.ug.runtime import (
    UGDecodeResult,
    UGInterleavedMessage,
    UGSegmentState,
    UGSRTPreparedInput,
)
from sglang.srt.ug.u1 import (
    U1_IMG_START_TOKEN,
    U1SubprocessVLMBackend,
    U1UGModelAdapter,
    U1SRTBackedUGMiddleBridge,
    _u1_precompute_generated_image_commit_embeddings,
    build_u1_native_edit_img_condition_prepared_input,
    build_u1_native_edit_prepared_input,
    build_u1_native_edit_uncondition_prepared_input,
    build_u1_native_t2i_cfg_uncondition_prepared_input,
    build_u1_native_t2i_prepared_input,
    build_u1_vlm_input_ids_and_offsets,
    load_u1_native_image,
)


class TestU1OfficialParityHarness(unittest.TestCase):
    def test_u1_official_parity_harness(self):
        run_u1_official_parity_from_env(os.environ)


def run_u1_official_parity_from_env(env) -> Path:
    if env.get("SGLANG_TEST_U1_PARITY_DRY_RUN") == "1":
        output_dir = Path(
            env.get("SGLANG_TEST_U1_PARITY_OUTPUT")
            or tempfile.mkdtemp(prefix="u1-parity-")
        )
        return _write_dry_run_bundle(output_dir)

    if env.get("SGLANG_TEST_U1_PARITY_MODE") == "vlm_official_reference":
        return _run_vlm_official_reference_mode(env)
    if env.get("SGLANG_TEST_U1_PARITY_MODE") == "t2i_official_reference":
        return _run_t2i_official_reference_mode(env)
    if env.get("SGLANG_TEST_U1_PARITY_MODE") == "edit_official_reference":
        return _run_edit_official_reference_mode(env)
    if env.get("SGLANG_TEST_U1_PARITY_MODE") == "interleave_official_reference":
        return _run_interleave_official_reference_mode(env)

    reference_path = env.get("SGLANG_TEST_U1_PARITY_REFERENCE_ARTIFACT")
    candidate_path = env.get("SGLANG_TEST_U1_PARITY_CANDIDATE_ARTIFACT")
    output_dir = env.get("SGLANG_TEST_U1_PARITY_OUTPUT")
    missing = [
        name
        for name, value in (
            ("SGLANG_TEST_U1_PARITY_REFERENCE_ARTIFACT", reference_path),
            ("SGLANG_TEST_U1_PARITY_CANDIDATE_ARTIFACT", candidate_path),
            ("SGLANG_TEST_U1_PARITY_OUTPUT", output_dir),
        )
        if not value
    ]
    if missing:
        raise unittest.SkipTest(
            "U1 official parity harness is opt-in; missing env: " + ", ".join(missing)
        )

    reference = UGParityArtifact.from_json(Path(reference_path).read_text())
    candidate = UGParityArtifact.from_json(Path(candidate_path).read_text())
    case = _case_from_artifact(reference)
    report = compare_ug_parity_artifacts(reference, candidate)
    bundle = write_ug_parity_bundle(
        output_dir=output_dir,
        case=case,
        reference=reference,
        candidate=candidate,
        report=report,
    )
    if not report.passed:
        raise AssertionError(f"U1 parity failed; report: {bundle / 'report.json'}")
    return bundle


def _run_vlm_official_reference_mode(env) -> Path:
    output_dir = Path(
        env.get("SGLANG_TEST_U1_PARITY_OUTPUT")
        or tempfile.mkdtemp(prefix="u1-vlm-parity-")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    official_py = Path(
        env.get("SGLANG_TEST_U1_OFFICIAL_PY")
        or "/data/venvs/sensenova_u1_official/bin/python"
    )
    official_repo = Path(
        env.get("SGLANG_TEST_U1_OFFICIAL_REPO") or "/data/repos/SenseNova-U1"
    )
    model_path = env.get("SGLANG_TEST_U1_MODEL_PATH") or (
        "/data/models/SenseNova-U1-8B-MoT"
    )
    image_path = Path(
        env.get("SGLANG_TEST_U1_VLM_IMAGE")
        or official_repo / "examples/vqa/data/images/image1.jpg"
    )
    question = env.get("SGLANG_TEST_U1_VLM_QUESTION") or "What is in this image?"
    max_new_tokens = int(env.get("SGLANG_TEST_U1_VLM_MAX_NEW_TOKENS") or "4")
    case_id = env.get("SGLANG_TEST_U1_VLM_CASE_ID") or "u1-vlm-official-reference"
    device = env.get("SGLANG_TEST_U1_VLM_DEVICE") or "cuda"
    dtype = env.get("SGLANG_TEST_U1_VLM_DTYPE") or "bfloat16"
    attn_backend = env.get("SGLANG_TEST_U1_VLM_ATTN_BACKEND") or "sdpa"
    timeout = int(env.get("SGLANG_TEST_U1_PARITY_TIMEOUT") or "600")

    case = UGParityCase(
        case_id=case_id,
        model="sensenova-u1",
        task="vlm",
        prompt=question,
        image_path=str(image_path),
        sampling_params={
            "max_new_tokens": max_new_tokens,
            "device": device,
            "dtype": dtype,
            "attn_backend": attn_backend,
        },
        dump_points=("text", "input_image"),
        metadata={
            "mode": "vlm_official_reference",
            "official_python": str(official_py),
            "official_repo": str(official_repo),
            "model_path": model_path,
        },
    )

    reference = _run_official_vlm_reference(
        case=case,
        official_py=official_py,
        official_repo=official_repo,
        model_path=model_path,
        image_path=image_path,
        question=question,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype,
        attn_backend=attn_backend,
        timeout=timeout,
        output_dir=output_dir,
        env=env,
    )
    candidate_path = env.get("SGLANG_TEST_U1_PARITY_CANDIDATE_ARTIFACT")
    if candidate_path:
        candidate = UGParityArtifact.from_json(Path(candidate_path).read_text())
    elif env.get("SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE") == "1":
        candidate = _run_sglang_native_vlm_candidate(
            case=case,
            model_path=env.get("SGLANG_TEST_U1_CANDIDATE_MODEL_PATH") or model_path,
            image_path=image_path,
            question=question,
            max_new_tokens=max_new_tokens,
            cuda_visible_devices=env.get("SGLANG_TEST_U1_CUDA_VISIBLE_DEVICES"),
        )
    elif env.get("SGLANG_TEST_U1_PARITY_RUN_SGLANG_CANDIDATE") == "1":
        candidate = _run_sglang_vlm_candidate(
            case=case,
            official_py=Path(
                env.get("SGLANG_TEST_U1_CANDIDATE_PY") or str(official_py)
            ),
            official_repo=Path(
                env.get("SGLANG_TEST_U1_CANDIDATE_REPO") or str(official_repo)
            ),
            model_path=env.get("SGLANG_TEST_U1_CANDIDATE_MODEL_PATH") or model_path,
            image_path=image_path,
            question=question,
            max_new_tokens=max_new_tokens,
            device=env.get("SGLANG_TEST_U1_CANDIDATE_DEVICE") or device,
            dtype=env.get("SGLANG_TEST_U1_CANDIDATE_DTYPE") or dtype,
            attn_backend=(
                env.get("SGLANG_TEST_U1_CANDIDATE_ATTN_BACKEND") or attn_backend
            ),
            timeout=timeout,
            cuda_visible_devices=env.get("SGLANG_TEST_U1_CUDA_VISIBLE_DEVICES"),
            output_dir=output_dir / "candidate-work",
        )
    else:
        candidate = _candidate_unavailable_artifact(case, image_path=image_path)

    report = compare_ug_parity_artifacts(reference, candidate)
    bundle = write_ug_parity_bundle(
        output_dir=output_dir,
        case=case,
        reference=reference,
        candidate=candidate,
        report=report,
    )
    if reference.error:
        raise AssertionError(
            f"U1 official VLM reference failed; report: {bundle / 'report.json'}"
        )
    if candidate_path and not report.passed:
        raise AssertionError(f"U1 VLM parity failed; report: {bundle / 'report.json'}")
    return bundle


def _run_t2i_official_reference_mode(env) -> Path:
    output_dir = Path(
        env.get("SGLANG_TEST_U1_PARITY_OUTPUT")
        or tempfile.mkdtemp(prefix="u1-t2i-parity-")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    official_py = Path(
        env.get("SGLANG_TEST_U1_OFFICIAL_PY")
        or "/data/venvs/sensenova_u1_official/bin/python"
    )
    official_repo = Path(
        env.get("SGLANG_TEST_U1_OFFICIAL_REPO") or "/data/repos/SenseNova-U1"
    )
    model_path = env.get("SGLANG_TEST_U1_MODEL_PATH") or (
        "/data/models/SenseNova-U1-8B-MoT"
    )
    prompt = env.get("SGLANG_TEST_U1_T2I_PROMPT") or (
        "A red ceramic cup on a wooden table, soft studio lighting."
    )
    width = int(env.get("SGLANG_TEST_U1_T2I_WIDTH") or "512")
    height = int(env.get("SGLANG_TEST_U1_T2I_HEIGHT") or "512")
    cfg_scale = float(env.get("SGLANG_TEST_U1_T2I_CFG_SCALE") or "4.0")
    cfg_norm = env.get("SGLANG_TEST_U1_T2I_CFG_NORM") or "none"
    cfg_interval = _parse_float_pair(
        env.get("SGLANG_TEST_U1_T2I_CFG_INTERVAL"),
        default=(0.0, 1.0),
    )
    timestep_shift = float(env.get("SGLANG_TEST_U1_T2I_TIMESTEP_SHIFT") or "3.0")
    num_steps = int(env.get("SGLANG_TEST_U1_T2I_NUM_STEPS") or "50")
    seed = int(env.get("SGLANG_TEST_U1_T2I_SEED") or "42")
    device = env.get("SGLANG_TEST_U1_T2I_DEVICE") or "cuda"
    dtype = env.get("SGLANG_TEST_U1_T2I_DTYPE") or "bfloat16"
    attn_backend = env.get("SGLANG_TEST_U1_T2I_ATTN_BACKEND") or "sdpa"
    timeout = int(env.get("SGLANG_TEST_U1_PARITY_TIMEOUT") or "1800")
    case_id = env.get("SGLANG_TEST_U1_T2I_CASE_ID") or "u1-t2i-official-reference"

    case = UGParityCase(
        case_id=case_id,
        model="sensenova-u1",
        task="t2i",
        prompt=prompt,
        seed=seed,
        sampling_params={
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "cfg_norm": cfg_norm,
            "cfg_interval": list(cfg_interval),
            "timestep_shift": timestep_shift,
            "num_steps": num_steps,
            "device": device,
            "dtype": dtype,
            "attn_backend": attn_backend,
        },
        dump_points=("image",),
        metadata={
            "mode": "t2i_official_reference",
            "official_python": str(official_py),
            "official_repo": str(official_repo),
            "model_path": model_path,
        },
    )
    reference = _run_official_t2i_reference(
        case=case,
        official_py=official_py,
        official_repo=official_repo,
        model_path=model_path,
        prompt=prompt,
        width=width,
        height=height,
        cfg_scale=cfg_scale,
        cfg_norm=cfg_norm,
        cfg_interval=cfg_interval,
        timestep_shift=timestep_shift,
        num_steps=num_steps,
        seed=seed,
        device=device,
        dtype=dtype,
        attn_backend=attn_backend,
        timeout=timeout,
        output_dir=output_dir,
        env=env,
    )

    candidate_path = env.get("SGLANG_TEST_U1_PARITY_CANDIDATE_ARTIFACT")
    if candidate_path:
        candidate = UGParityArtifact.from_json(Path(candidate_path).read_text())
    elif env.get("SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE") == "1":
        candidate = _run_sglang_native_t2i_candidate(
            case=case,
            model_path=env.get("SGLANG_TEST_U1_CANDIDATE_MODEL_PATH") or model_path,
            prompt=prompt,
            width=width,
            height=height,
            cfg_scale=cfg_scale,
            cfg_norm=cfg_norm,
            cfg_interval=cfg_interval,
            timestep_shift=timestep_shift,
            num_steps=num_steps,
            seed=seed,
            dtype=env.get("SGLANG_TEST_U1_CANDIDATE_DTYPE") or dtype,
            mem_fraction_static=float(
                env.get("SGLANG_TEST_U1_CANDIDATE_MEM_FRACTION_STATIC") or "0.80"
            ),
            enable_fp32_lm_head=_parse_bool(
                env.get("SGLANG_TEST_U1_CANDIDATE_ENABLE_FP32_LM_HEAD"),
                False,
            ),
            attention_backend=(
                env.get("SGLANG_TEST_U1_CANDIDATE_ATTENTION_BACKEND") or "torch_native"
            ),
            cuda_visible_devices=env.get("SGLANG_TEST_U1_CUDA_VISIBLE_DEVICES"),
            output_dir=output_dir,
        )
    else:
        candidate = _candidate_t2i_unavailable_artifact(case)

    report = _compare_image_artifacts_with_tolerance(
        reference,
        candidate,
        mean_threshold=float(env.get("SGLANG_TEST_U1_T2I_MEAN_THRESHOLD") or "8.0"),
        max_threshold=float(env.get("SGLANG_TEST_U1_T2I_MAX_THRESHOLD") or "192.0"),
        p99_threshold=float(env.get("SGLANG_TEST_U1_T2I_P99_THRESHOLD") or "32.0"),
        psnr_threshold=float(env.get("SGLANG_TEST_U1_T2I_PSNR_THRESHOLD") or "30.0"),
        ssim_threshold=float(env.get("SGLANG_TEST_U1_T2I_SSIM_THRESHOLD") or "0.995"),
    )
    bundle = write_ug_parity_bundle(
        output_dir=output_dir,
        case=case,
        reference=reference,
        candidate=candidate,
        report=report,
    )
    if reference.error:
        raise AssertionError(
            f"U1 official T2I reference failed; report: {bundle / 'report.json'}"
        )
    if (
        env.get("SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE") == "1"
        and not report.passed
    ):
        raise AssertionError(f"U1 T2I parity failed; report: {bundle / 'report.json'}")
    return bundle


def _run_edit_official_reference_mode(env) -> Path:
    output_dir = Path(
        env.get("SGLANG_TEST_U1_PARITY_OUTPUT")
        or tempfile.mkdtemp(prefix="u1-edit-parity-")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    official_py = Path(
        env.get("SGLANG_TEST_U1_OFFICIAL_PY")
        or "/data/venvs/sensenova_u1_official/bin/python"
    )
    official_repo = Path(
        env.get("SGLANG_TEST_U1_OFFICIAL_REPO") or "/data/repos/SenseNova-U1"
    )
    model_path = env.get("SGLANG_TEST_U1_MODEL_PATH") or (
        "/data/models/SenseNova-U1-8B-MoT"
    )
    image_path = Path(
        env.get("SGLANG_TEST_U1_EDIT_IMAGE")
        or official_repo / "examples/editing/data/images/1.webp"
    )
    prompt = env.get("SGLANG_TEST_U1_EDIT_PROMPT") or (
        "Change the jacket of the person on the left to bright yellow."
    )
    width = int(env.get("SGLANG_TEST_U1_EDIT_WIDTH") or "512")
    height = int(env.get("SGLANG_TEST_U1_EDIT_HEIGHT") or "512")
    cfg_scale = float(env.get("SGLANG_TEST_U1_EDIT_CFG_SCALE") or "1.0")
    img_cfg_scale = float(env.get("SGLANG_TEST_U1_EDIT_IMG_CFG_SCALE") or "1.0")
    cfg_norm = env.get("SGLANG_TEST_U1_EDIT_CFG_NORM") or "none"
    cfg_interval = _parse_float_pair(
        env.get("SGLANG_TEST_U1_EDIT_CFG_INTERVAL"),
        default=(0.0, 1.0),
    )
    timestep_shift = float(env.get("SGLANG_TEST_U1_EDIT_TIMESTEP_SHIFT") or "3.0")
    num_steps = int(env.get("SGLANG_TEST_U1_EDIT_NUM_STEPS") or "50")
    seed = int(env.get("SGLANG_TEST_U1_EDIT_SEED") or "42")
    device = env.get("SGLANG_TEST_U1_EDIT_DEVICE") or "cuda"
    dtype = env.get("SGLANG_TEST_U1_EDIT_DTYPE") or "bfloat16"
    attn_backend = env.get("SGLANG_TEST_U1_EDIT_ATTN_BACKEND") or "sdpa"
    timeout = int(env.get("SGLANG_TEST_U1_PARITY_TIMEOUT") or "1800")
    case_id = env.get("SGLANG_TEST_U1_EDIT_CASE_ID") or "u1-edit-official-reference"

    case = UGParityCase(
        case_id=case_id,
        model="sensenova-u1",
        task="edit",
        prompt=prompt,
        image_path=str(image_path),
        seed=seed,
        sampling_params={
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "img_cfg_scale": img_cfg_scale,
            "cfg_norm": cfg_norm,
            "cfg_interval": list(cfg_interval),
            "timestep_shift": timestep_shift,
            "num_steps": num_steps,
            "device": device,
            "dtype": dtype,
            "attn_backend": attn_backend,
        },
        dump_points=("image",),
        metadata={
            "mode": "edit_official_reference",
            "official_python": str(official_py),
            "official_repo": str(official_repo),
            "model_path": model_path,
        },
    )
    reference = _run_official_edit_reference(
        case=case,
        official_py=official_py,
        official_repo=official_repo,
        model_path=model_path,
        image_path=image_path,
        prompt=prompt,
        width=width,
        height=height,
        cfg_scale=cfg_scale,
        img_cfg_scale=img_cfg_scale,
        cfg_norm=cfg_norm,
        cfg_interval=cfg_interval,
        timestep_shift=timestep_shift,
        num_steps=num_steps,
        seed=seed,
        device=device,
        dtype=dtype,
        attn_backend=attn_backend,
        timeout=timeout,
        output_dir=output_dir,
        env=env,
    )

    candidate_path = env.get("SGLANG_TEST_U1_PARITY_CANDIDATE_ARTIFACT")
    if candidate_path:
        candidate = UGParityArtifact.from_json(Path(candidate_path).read_text())
    elif env.get("SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE") == "1":
        candidate = _run_sglang_native_edit_candidate(
            case=case,
            model_path=env.get("SGLANG_TEST_U1_CANDIDATE_MODEL_PATH") or model_path,
            image_path=image_path,
            prompt=prompt,
            width=width,
            height=height,
            cfg_scale=cfg_scale,
            img_cfg_scale=img_cfg_scale,
            cfg_norm=cfg_norm,
            cfg_interval=cfg_interval,
            timestep_shift=timestep_shift,
            num_steps=num_steps,
            seed=seed,
            dtype=env.get("SGLANG_TEST_U1_CANDIDATE_DTYPE") or dtype,
            mem_fraction_static=float(
                env.get("SGLANG_TEST_U1_CANDIDATE_MEM_FRACTION_STATIC") or "0.80"
            ),
            attention_backend=(
                env.get("SGLANG_TEST_U1_CANDIDATE_ATTENTION_BACKEND") or "torch_native"
            ),
            cuda_visible_devices=env.get("SGLANG_TEST_U1_CUDA_VISIBLE_DEVICES"),
            output_dir=output_dir,
        )
    else:
        candidate = _candidate_image_unavailable_artifact(case)

    report = _compare_image_artifacts_with_tolerance(
        reference,
        candidate,
        mean_threshold=float(env.get("SGLANG_TEST_U1_EDIT_MEAN_THRESHOLD") or "8.0"),
        max_threshold=float(env.get("SGLANG_TEST_U1_EDIT_MAX_THRESHOLD") or "192.0"),
        p99_threshold=float(env.get("SGLANG_TEST_U1_EDIT_P99_THRESHOLD") or "32.0"),
        psnr_threshold=float(env.get("SGLANG_TEST_U1_EDIT_PSNR_THRESHOLD") or "30.0"),
        ssim_threshold=float(env.get("SGLANG_TEST_U1_EDIT_SSIM_THRESHOLD") or "0.995"),
    )
    bundle = write_ug_parity_bundle(
        output_dir=output_dir,
        case=case,
        reference=reference,
        candidate=candidate,
        report=report,
    )
    if reference.error:
        raise AssertionError(
            f"U1 official edit reference failed; report: {bundle / 'report.json'}"
        )
    if (
        env.get("SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE") == "1"
        and not report.passed
    ):
        raise AssertionError(f"U1 edit parity failed; report: {bundle / 'report.json'}")
    return bundle


def _run_interleave_official_reference_mode(env) -> Path:
    output_dir = Path(
        env.get("SGLANG_TEST_U1_PARITY_OUTPUT")
        or tempfile.mkdtemp(prefix="u1-interleave-parity-")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    official_py = Path(
        env.get("SGLANG_TEST_U1_OFFICIAL_PY")
        or "/data/venvs/sensenova_u1_official/bin/python"
    )
    official_repo = Path(
        env.get("SGLANG_TEST_U1_OFFICIAL_REPO") or "/data/repos/SenseNova-U1"
    )
    model_path = env.get("SGLANG_TEST_U1_MODEL_PATH") or (
        "/data/models/SenseNova-U1-8B-MoT"
    )
    prompt = env.get("SGLANG_TEST_U1_INTERLEAVE_PROMPT") or (
        "Create an image of a red ceramic cup, then briefly describe the result."
    )
    image_paths = _parse_path_list(env.get("SGLANG_TEST_U1_INTERLEAVE_IMAGES"))
    width = int(env.get("SGLANG_TEST_U1_INTERLEAVE_WIDTH") or "512")
    height = int(env.get("SGLANG_TEST_U1_INTERLEAVE_HEIGHT") or "512")
    cfg_scale = float(env.get("SGLANG_TEST_U1_INTERLEAVE_CFG_SCALE") or "4.0")
    img_cfg_scale = float(env.get("SGLANG_TEST_U1_INTERLEAVE_IMG_CFG_SCALE") or "1.0")
    cfg_interval = _parse_float_pair(
        env.get("SGLANG_TEST_U1_INTERLEAVE_CFG_INTERVAL"),
        default=(0.0, 1.0),
    )
    timestep_shift = float(env.get("SGLANG_TEST_U1_INTERLEAVE_TIMESTEP_SHIFT") or "3.0")
    num_steps = int(env.get("SGLANG_TEST_U1_INTERLEAVE_NUM_STEPS") or "50")
    seed = int(env.get("SGLANG_TEST_U1_INTERLEAVE_SEED") or "42")
    think_mode = _parse_bool(env.get("SGLANG_TEST_U1_INTERLEAVE_THINK"), False)
    post_text_max_new_tokens = int(
        env.get("SGLANG_TEST_U1_INTERLEAVE_POST_TEXT_MAX_NEW_TOKENS") or "8"
    )
    max_interleave_images = int(env.get("SGLANG_TEST_U1_INTERLEAVE_MAX_IMAGES") or "10")
    max_interleave_text_segments = int(
        env.get("SGLANG_TEST_U1_INTERLEAVE_MAX_TEXT_SEGMENTS") or "16"
    )
    device = env.get("SGLANG_TEST_U1_INTERLEAVE_DEVICE") or "cuda"
    dtype = env.get("SGLANG_TEST_U1_INTERLEAVE_DTYPE") or "bfloat16"
    attn_backend = env.get("SGLANG_TEST_U1_INTERLEAVE_ATTN_BACKEND") or "sdpa"
    timeout = int(env.get("SGLANG_TEST_U1_PARITY_TIMEOUT") or "1800")
    case_id = (
        env.get("SGLANG_TEST_U1_INTERLEAVE_CASE_ID")
        or "u1-interleave-official-reference"
    )
    alignment_gate = _normalize_u1_interleave_alignment_gate(
        env.get("SGLANG_TEST_U1_INTERLEAVE_ALIGNMENT_GATE")
    )
    teacher_force_mode = env.get(
        "SGLANG_TEST_U1_INTERLEAVE_TEACHER_FORCE_REFERENCE_IMAGES"
    )
    teacher_force_text = _parse_bool(
        env.get("SGLANG_TEST_U1_INTERLEAVE_TEACHER_FORCE_REFERENCE_TEXT"),
        False,
    )
    if alignment_gate == "teacher_force_text":
        teacher_force_text = True
    elif alignment_gate == "official_raw_image_commit" and not teacher_force_mode:
        teacher_force_mode = "raw"
    fail_on_report = _parse_bool(
        env.get("SGLANG_TEST_U1_INTERLEAVE_FAIL_ON_REPORT"),
        alignment_gate != "full_free_run_observation",
    )

    messages = tuple(
        [{"type": "image", "image": str(path)} for path in image_paths]
        + [{"type": "text", "text": prompt}]
    )
    case = UGParityCase(
        case_id=case_id,
        model="sensenova-u1",
        task="interleave",
        prompt=prompt,
        image_path=str(image_paths[0]) if image_paths else None,
        messages=messages,
        seed=seed,
        sampling_params={
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "img_cfg_scale": img_cfg_scale,
            "cfg_interval": list(cfg_interval),
            "timestep_shift": timestep_shift,
            "num_steps": num_steps,
            "think_mode": think_mode,
            "post_text_max_new_tokens": post_text_max_new_tokens,
            "max_interleave_images": max_interleave_images,
            "max_interleave_text_segments": max_interleave_text_segments,
            "device": device,
            "dtype": dtype,
            "attn_backend": attn_backend,
        },
        dump_points=("segments", "image", "text"),
        metadata={
            "mode": "interleave_official_reference",
            "alignment_gate": alignment_gate,
            "official_python": str(official_py),
            "official_repo": str(official_repo),
            "model_path": model_path,
        },
    )
    dump_debug_tensors = env.get("SGLANG_TEST_U1_INTERLEAVE_DUMP_DEBUG_TENSORS") == "1"
    reference = _run_official_interleave_reference(
        case=case,
        official_py=official_py,
        official_repo=official_repo,
        model_path=model_path,
        prompt=prompt,
        image_paths=image_paths,
        width=width,
        height=height,
        cfg_scale=cfg_scale,
        img_cfg_scale=img_cfg_scale,
        cfg_interval=cfg_interval,
        timestep_shift=timestep_shift,
        num_steps=num_steps,
        seed=seed,
        think_mode=think_mode,
        device=device,
        dtype=dtype,
        attn_backend=attn_backend,
        timeout=timeout,
        output_dir=output_dir,
        env=env,
        dump_raw_images=(
            env.get("SGLANG_TEST_U1_INTERLEAVE_DUMP_RAW_IMAGES") == "1"
            or env.get("SGLANG_TEST_U1_INTERLEAVE_DUMP_TOKEN_TRACE") == "1"
            or str(teacher_force_mode or "").lower() in ("raw", "tensor", "pt")
            or dump_debug_tensors
        ),
        dump_debug_tensors=dump_debug_tensors,
    )

    candidate_path = env.get("SGLANG_TEST_U1_PARITY_CANDIDATE_ARTIFACT")
    if candidate_path:
        candidate = UGParityArtifact.from_json(Path(candidate_path).read_text())
    elif env.get("SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE") == "1":
        candidate = _run_sglang_native_interleave_candidate(
            case=case,
            model_path=env.get("SGLANG_TEST_U1_CANDIDATE_MODEL_PATH") or model_path,
            prompt=prompt,
            image_paths=image_paths,
            width=width,
            height=height,
            cfg_scale=cfg_scale,
            img_cfg_scale=img_cfg_scale,
            cfg_interval=cfg_interval,
            timestep_shift=timestep_shift,
            num_steps=num_steps,
            seed=seed,
            think_mode=think_mode,
            post_text_max_new_tokens=post_text_max_new_tokens,
            max_interleave_images=max_interleave_images,
            max_interleave_text_segments=max_interleave_text_segments,
            dtype=env.get("SGLANG_TEST_U1_CANDIDATE_DTYPE") or dtype,
            mem_fraction_static=float(
                env.get("SGLANG_TEST_U1_CANDIDATE_MEM_FRACTION_STATIC") or "0.80"
            ),
            attention_backend=(
                env.get("SGLANG_TEST_U1_CANDIDATE_ATTENTION_BACKEND") or "torch_native"
            ),
            cuda_visible_devices=env.get("SGLANG_TEST_U1_CUDA_VISIBLE_DEVICES"),
            output_dir=output_dir,
            teacher_force_commit_paths=(
                _interleave_teacher_force_commit_paths(
                    reference=reference,
                    mode=teacher_force_mode,
                )
            ),
            teacher_force_text_parts=(
                _interleave_teacher_force_text_parts(reference)
                if teacher_force_text
                else None
            ),
            teacher_force_token_parts=(
                _interleave_teacher_force_token_parts(reference)
                if teacher_force_text
                else None
            ),
            force_u_decode_token_ids=(
                _interleave_reference_u_decode_token_ids(
                    reference=reference,
                    tokenizer_model_path=env.get("SGLANG_TEST_U1_CANDIDATE_MODEL_PATH")
                    or model_path,
                )
                if _parse_bool(
                    env.get("SGLANG_TEST_U1_INTERLEAVE_FORCE_REFERENCE_U_DECODE"),
                    False,
                )
                else None
            ),
            dump_debug_tensors=dump_debug_tensors,
            alignment_gate=alignment_gate,
        )
    else:
        candidate = _candidate_interleave_unavailable_artifact(case)

    min_candidate_images_env = env.get("SGLANG_TEST_U1_INTERLEAVE_MIN_IMAGES")
    if min_candidate_images_env is None:
        min_candidate_images = min(
            max_interleave_images,
            len(_artifact_image_paths(reference)),
        )
    else:
        min_candidate_images = int(min_candidate_images_env)
    require_text_after_each_image = _parse_bool(
        env.get("SGLANG_TEST_U1_INTERLEAVE_REQUIRE_TEXT_AFTER_EACH_IMAGE"),
        True,
    )

    report = _compare_interleave_artifacts_with_tolerance(
        reference,
        candidate,
        mean_threshold=float(
            env.get("SGLANG_TEST_U1_INTERLEAVE_MEAN_THRESHOLD") or "8.0"
        ),
        max_threshold=float(
            env.get("SGLANG_TEST_U1_INTERLEAVE_MAX_THRESHOLD") or "192.0"
        ),
        p99_threshold=float(
            env.get("SGLANG_TEST_U1_INTERLEAVE_P99_THRESHOLD") or "32.0"
        ),
        psnr_threshold=float(
            env.get("SGLANG_TEST_U1_INTERLEAVE_PSNR_THRESHOLD") or "30.0"
        ),
        ssim_threshold=float(
            env.get("SGLANG_TEST_U1_INTERLEAVE_SSIM_THRESHOLD") or "0.995"
        ),
        require_text=(env.get("SGLANG_TEST_U1_INTERLEAVE_REQUIRE_TEXT", "1") != "0"),
        require_text_exact=(
            env.get("SGLANG_TEST_U1_INTERLEAVE_REQUIRE_TEXT_EXACT", "0") == "1"
        ),
        min_candidate_images=min_candidate_images,
        require_text_after_each_image=require_text_after_each_image,
        teacher_force_text=teacher_force_text,
    )
    report = _annotate_u1_interleave_alignment_report(
        report,
        alignment_gate=alignment_gate,
        teacher_force_mode=teacher_force_mode,
        teacher_force_text=teacher_force_text,
        fail_on_report=fail_on_report,
    )
    bundle = write_ug_parity_bundle(
        output_dir=output_dir,
        case=case,
        reference=reference,
        candidate=candidate,
        report=report,
    )
    if reference.error:
        raise AssertionError(
            f"U1 official interleave reference failed; report: {bundle / 'report.json'}"
        )
    if (
        env.get("SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE") == "1"
        and not report.passed
        and fail_on_report
    ):
        raise AssertionError(
            f"U1 interleave parity failed; report: {bundle / 'report.json'}"
        )
    return bundle


def _normalize_u1_interleave_alignment_gate(value: str | None) -> str:
    normalized = str(value or "free_run").strip().lower().replace("-", "_")
    aliases = {
        "": "free_run",
        "0": "free_run",
        "none": "free_run",
        "free": "free_run",
        "free_run": "free_run",
        "full_free_run": "free_run",
        "teacher_force": "teacher_force_text",
        "teacher_forced_text": "teacher_force_text",
        "teacher_force_text": "teacher_force_text",
        "text_teacher_force": "teacher_force_text",
        "official_commit": "official_raw_image_commit",
        "raw_commit": "official_raw_image_commit",
        "official_raw_commit": "official_raw_image_commit",
        "official_raw_image_commit": "official_raw_image_commit",
        "commit_semantics": "official_raw_image_commit",
        "observation": "full_free_run_observation",
        "observe": "full_free_run_observation",
        "full_free_run_observation": "full_free_run_observation",
        "free_run_observation": "full_free_run_observation",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported SGLANG_TEST_U1_INTERLEAVE_ALIGNMENT_GATE value: " f"{value!r}"
        )
    return aliases[normalized]


def _annotate_u1_interleave_alignment_report(
    report: UGParityReport,
    *,
    alignment_gate: str,
    teacher_force_mode: str | None,
    teacher_force_text: bool,
    fail_on_report: bool,
) -> UGParityReport:
    metadata = dict(report.metadata)
    metadata.update(
        {
            "alignment_gate": alignment_gate,
            "alignment_gate_policy": _u1_interleave_alignment_gate_policy(
                alignment_gate
            ),
            "teacher_force_reference_images": teacher_force_mode,
            "teacher_force_reference_text": bool(teacher_force_text),
            "fail_on_report": bool(fail_on_report),
        }
    )
    return UGParityReport(
        case_id=report.case_id,
        model=report.model,
        passed=report.passed,
        diffs=report.diffs,
        metadata=metadata,
    )


def _u1_interleave_alignment_gate_policy(alignment_gate: str) -> str:
    if alignment_gate == "teacher_force_text":
        return "hard gate: official text and image markers drive candidate G"
    if alignment_gate == "official_raw_image_commit":
        return "hard gate: official raw generated images are committed back to U"
    if alignment_gate == "full_free_run_observation":
        return "observation: closed-loop free-run report does not fail by default"
    return "hard gate: closed-loop free-run candidate must pass configured tolerances"


def _run_sglang_native_vlm_candidate(
    *,
    case: UGParityCase,
    model_path: str,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    cuda_visible_devices: str | None,
) -> UGParityArtifact:
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    from transformers import AutoTokenizer

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.mm_utils import init_mm_embedding_cache
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import PortArgs, ServerArgs

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pixel_values, grid_hw = load_u1_native_image(image_path)
    input_ids, image_offsets, _ = build_u1_vlm_input_ids_and_offsets(
        tokenizer=tokenizer,
        grid_hw=grid_hw,
        question=question,
    )

    server_args = ServerArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        trust_remote_code=False,
        disable_cuda_graph=True,
        disable_hybrid_swa_memory=True,
        mem_fraction_static=0.80,
        chunked_prefill_size=-1,
    )
    port_args = PortArgs.init_new(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
        moe_ep_rank=0,
        moe_ep_size=1,
    )
    model = runner.model
    init_mm_embedding_cache()

    generated = []
    with torch.no_grad():
        batch = None
        try:
            if max_new_tokens > 0:
                batch, token = _run_native_u1_vlm_prefill(
                    runner=runner,
                    model=model,
                    model_config=model_config,
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    grid_hw=grid_hw,
                    image_offsets=image_offsets,
                )
                generated.append(token)
            for _ in range(1, max_new_tokens):
                token = _run_native_u1_vlm_decode_step(
                    runner=runner,
                    model_config=model_config,
                    batch=batch,
                    input_token=generated[-1],
                )
                generated.append(token)
        finally:
            runner.req_to_token_pool.clear()
            runner.token_to_kv_pool_allocator.clear()

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="sglang",
        text=text,
        image=summarize_ug_image(image_path),
        metadata={
            "candidate_backend": "u1_native_srt_vlm_kv_decode",
            "native_srt_model_runner": True,
            "model_cls": type(model).__module__ + "." + type(model).__name__,
            "language_cls": (
                type(model.language_model).__module__
                + "."
                + type(model.language_model).__name__
            ),
            "token_ids": generated,
            "kv_decode": True,
            "prefill_forwards": 1 if max_new_tokens > 0 else 0,
            "decode_forwards": max(0, max_new_tokens - 1),
        },
    )


def _run_native_u1_vlm_prefill(
    *,
    runner,
    model,
    model_config,
    input_ids: list[int],
    pixel_values: torch.Tensor,
    grid_hw: torch.Tensor,
    image_offsets: list[tuple[int, int]],
):
    from sglang.bench_one_batch import TreeCacheNamespace
    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
        Req,
        ScheduleBatch,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    runner.req_to_token_pool.clear()
    runner.token_to_kv_pool_allocator.clear()
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=pixel_values.clone(),
        model_specific_data={"image_grid_hws": grid_hw.clone()},
        offsets=image_offsets,
    )
    item.set_pad_value()
    mm_inputs = MultimodalInputs(mm_items=[item])
    padded_ids = model.pad_input_ids(list(input_ids), mm_inputs)
    req = Req(
        rid="u1-native-vlm-kv-decode",
        origin_input_text="",
        origin_input_ids=padded_ids,
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=1),
    )
    req.fill_ids = list(padded_ids)
    req.multimodal_inputs = mm_inputs
    req.logprob_start_len = -1
    req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
    tree_cache = TreeCacheNamespace(
        page_size=1,
        device=runner.device,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=[req],
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(worker_batch, runner)
    output, _ = runner.forward_extend(forward_batch)
    return batch, _greedy_next_token(output.next_token_logits)


def _run_native_u1_vlm_decode_step(
    *,
    runner,
    model_config,
    batch,
    input_token: int,
) -> int:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    del model_config
    batch.output_ids = torch.tensor(
        [input_token],
        dtype=torch.int64,
        device=runner.device,
    )
    batch.prepare_for_decode()
    worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(worker_batch, runner)
    output = runner.forward_decode(forward_batch)
    return _greedy_next_token(output.next_token_logits)


def _greedy_next_token(next_token_logits: torch.Tensor) -> int:
    return int(torch.argmax(next_token_logits[0]).item())


def _debug_topk_payload(logits: torch.Tensor, *, k: int = 8) -> dict[str, object]:
    logits = logits.detach()
    topk = torch.topk(logits[0].float(), k=min(k, logits.shape[-1]))
    return {
        "selected_token": int(torch.argmax(logits[0]).item()),
        "topk_ids": topk.indices.detach().cpu(),
        "topk_values": topk.values.detach().cpu(),
    }


def _run_sglang_native_t2i_candidate(
    *,
    case: UGParityCase,
    model_path: str,
    prompt: str,
    width: int,
    height: int,
    cfg_scale: float,
    cfg_norm: str,
    cfg_interval: tuple[float, float],
    timestep_shift: float,
    num_steps: int,
    seed: int,
    dtype: str,
    mem_fraction_static: float,
    enable_fp32_lm_head: bool,
    attention_backend: str | None,
    cuda_visible_devices: str | None,
    output_dir: Path,
) -> UGParityArtifact:
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    from transformers import AutoTokenizer

    from sglang.bench_one_batch import TreeCacheNamespace
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.mm_utils import init_mm_embedding_cache
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.ug.srt_executor import UGSRTSchedulerExecutor

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    server_args = ServerArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        trust_remote_code=False,
        disable_cuda_graph=True,
        disable_hybrid_swa_memory=True,
        mem_fraction_static=mem_fraction_static,
        chunked_prefill_size=-1,
        dtype=dtype,
        attention_backend=attention_backend,
        enable_fp32_lm_head=enable_fp32_lm_head,
    )
    port_args = PortArgs.init_new(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
        moe_ep_rank=0,
        moe_ep_size=1,
    )
    model = runner.model
    init_mm_embedding_cache()

    session_id = "u1-native-t2i"
    session_ref = SimpleNamespace(handle=SimpleNamespace(session_id=session_id))
    condition_prepared = build_u1_native_t2i_prepared_input(
        tokenizer=tokenizer,
        messages=[UGInterleavedMessage(type="text", content=prompt)],
        session=session_ref,
    )
    cfg_prepared = None
    if cfg_scale > 1.0:
        cfg_prepared = build_u1_native_t2i_cfg_uncondition_prepared_input(
            tokenizer=tokenizer,
            session=session_ref,
        )

    try:
        with torch.no_grad():
            condition_binding = _run_native_u1_prefill_binding(
                runner=runner,
                model=model,
                model_config=model_config,
                prepared=condition_prepared,
                session_id=session_id,
                request_id="u1-native-t2i-prefill",
            )
            cfg_binding = None
            if cfg_prepared is not None:
                cfg_binding = _run_native_u1_prefill_binding(
                    runner=runner,
                    model=model,
                    model_config=model_config,
                    prepared=cfg_prepared,
                    session_id=f"{session_id}:u1_t2i_cfg_uncondition",
                    request_id="u1-native-t2i-cfg-prefill",
                )

            tree_cache = TreeCacheNamespace(
                page_size=1,
                device=runner.device,
                token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            )
            scheduler = SimpleNamespace(
                model_worker=SimpleNamespace(model_runner=runner),
                tree_cache=tree_cache,
                session_controller=SimpleNamespace(),
                is_fully_idle=lambda: True,
            )
            srt_executor = UGSRTSchedulerExecutor(scheduler)
            native_executor = srt_executor.create_u1_native_srt_pixel_flow_executor()
            sampling_params = UGSamplingParams(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                cfg_text_scale=cfg_scale,
                cfg_img_scale=1.0,
                cfg_interval=list(cfg_interval),
                cfg_renorm_type=cfg_norm,
                timestep_shift=timestep_shift,
            )
            batch = SimpleNamespace(
                sampling_params=sampling_params,
                seed=seed,
                height=height,
                width=width,
            )
            handle = UGSessionHandle(
                session_id=session_id,
                anchor_request_id=condition_binding.request_id,
                context_length=condition_binding.token_count,
                context_version=1,
            )
            contexts = UGContextBundle(
                full=UGContextHandle(
                    request_id=condition_binding.request_id,
                    token_count=condition_binding.token_count,
                    session=handle,
                ),
                text_cfg=UGContextHandle(
                    request_id=cfg_binding.request_id if cfg_binding else "",
                    token_count=cfg_binding.token_count if cfg_binding else 0,
                ),
                image_cfg=UGContextHandle(request_id="", token_count=0),
            )
            segment = native_executor.generate(
                contexts=contexts,
                batch=batch,
                server_args=server_args,
                srt_kv_token_binding=condition_binding,
                cfg_img_condition_srt_kv_token_binding=cfg_binding,
            )
    finally:
        runner.req_to_token_pool.clear()
        runner.token_to_kv_pool_allocator.clear()

    image_path = output_dir / f"{case.case_id}.candidate.png"
    segment.image.save(image_path)
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="sglang",
        image=summarize_ug_image(image_path),
        metadata={
            "candidate_backend": "u1_native_srt_t2i_pixel_flow",
            "native_srt_model_runner": True,
            "attention_backend": attention_backend,
            "dtype": dtype,
            "enable_fp32_lm_head": enable_fp32_lm_head,
            "mem_fraction_static": mem_fraction_static,
            "model_cls": type(model).__module__ + "." + type(model).__name__,
            "language_cls": (
                type(model.language_model).__module__
                + "."
                + type(model.language_model).__name__
            ),
            "image_path": str(image_path),
            "prefill_forwards": 1,
            "cfg_prefill_forwards": 1 if cfg_prepared is not None else 0,
            "temp_g_forward_count": int(srt_executor.temp_g_forward_count),
            "temp_g_allocated_token_count": int(
                srt_executor.temp_g_allocated_token_count
            ),
            "g_metadata": dict(segment.metadata),
        },
    )


def _run_sglang_native_edit_candidate(
    *,
    case: UGParityCase,
    model_path: str,
    image_path: Path,
    prompt: str,
    width: int,
    height: int,
    cfg_scale: float,
    img_cfg_scale: float,
    cfg_norm: str,
    cfg_interval: tuple[float, float],
    timestep_shift: float,
    num_steps: int,
    seed: int,
    dtype: str,
    mem_fraction_static: float,
    attention_backend: str | None,
    cuda_visible_devices: str | None,
    output_dir: Path,
) -> UGParityArtifact:
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    from transformers import AutoTokenizer

    from sglang.bench_one_batch import TreeCacheNamespace
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.mm_utils import init_mm_embedding_cache
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.ug.srt_executor import UGSRTSchedulerExecutor

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    server_args = ServerArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        trust_remote_code=False,
        disable_cuda_graph=True,
        disable_hybrid_swa_memory=True,
        mem_fraction_static=mem_fraction_static,
        chunked_prefill_size=-1,
        dtype=dtype,
        attention_backend=attention_backend,
    )
    port_args = PortArgs.init_new(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
        moe_ep_rank=0,
        moe_ep_size=1,
    )
    model = runner.model
    init_mm_embedding_cache()

    session_id = "u1-native-edit"
    session_ref = SimpleNamespace(handle=SimpleNamespace(session_id=session_id))
    condition_prepared = build_u1_native_edit_prepared_input(
        tokenizer=tokenizer,
        messages=[
            UGInterleavedMessage(type="image", content=str(image_path)),
            UGInterleavedMessage(type="text", content=prompt),
        ],
        session=session_ref,
    )
    needs_cfg = not (cfg_scale == 1.0 and img_cfg_scale == 1.0)
    needs_img_condition = needs_cfg and (
        img_cfg_scale == 1.0 or cfg_scale != img_cfg_scale
    )
    needs_uncondition = needs_cfg and img_cfg_scale != 1.0
    img_condition_prepared = None
    if needs_img_condition:
        img_condition_prepared = build_u1_native_edit_img_condition_prepared_input(
            tokenizer=tokenizer,
            messages=[
                UGInterleavedMessage(type="image", content=str(image_path)),
                UGInterleavedMessage(type="text", content=prompt),
            ],
            session=session_ref,
        )
    uncondition_prepared = None
    if needs_uncondition:
        uncondition_prepared = build_u1_native_edit_uncondition_prepared_input(
            tokenizer=tokenizer,
            session=session_ref,
        )

    try:
        with torch.no_grad():
            img_condition_binding = None
            if img_condition_prepared is not None:
                img_condition_binding = _run_native_u1_prefill_binding(
                    runner=runner,
                    model=model,
                    model_config=model_config,
                    prepared=img_condition_prepared,
                    session_id=f"{session_id}:u1_edit_img_condition",
                    request_id="u1-native-edit-img-condition-prefill",
                )
            uncondition_binding = None
            if uncondition_prepared is not None:
                uncondition_binding = _run_native_u1_prefill_binding(
                    runner=runner,
                    model=model,
                    model_config=model_config,
                    prepared=uncondition_prepared,
                    session_id=f"{session_id}:u1_edit_uncondition",
                    request_id="u1-native-edit-uncondition-prefill",
                )
            condition_binding = _run_native_u1_prefill_binding(
                runner=runner,
                model=model,
                model_config=model_config,
                prepared=condition_prepared,
                session_id=session_id,
                request_id="u1-native-edit-prefill",
            )
            tree_cache = TreeCacheNamespace(
                page_size=1,
                device=runner.device,
                token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            )
            scheduler = SimpleNamespace(
                model_worker=SimpleNamespace(model_runner=runner),
                tree_cache=tree_cache,
                session_controller=SimpleNamespace(),
                is_fully_idle=lambda: True,
            )
            srt_executor = UGSRTSchedulerExecutor(scheduler)
            native_executor = srt_executor.create_u1_native_srt_pixel_flow_executor()
            sampling_params = UGSamplingParams(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                cfg_text_scale=cfg_scale,
                cfg_img_scale=img_cfg_scale,
                cfg_interval=list(cfg_interval),
                cfg_renorm_type=cfg_norm,
                timestep_shift=timestep_shift,
            )
            batch = SimpleNamespace(
                sampling_params=sampling_params,
                seed=seed,
                height=height,
                width=width,
            )
            handle = UGSessionHandle(
                session_id=session_id,
                anchor_request_id=condition_binding.request_id,
                context_length=condition_binding.token_count,
                context_version=1,
            )
            contexts = UGContextBundle(
                full=UGContextHandle(
                    request_id=condition_binding.request_id,
                    token_count=condition_binding.token_count,
                    session=handle,
                    metadata={
                        "u1_g_position_start": condition_prepared.adapter_metadata[
                            "u1"
                        ]["g_position_start"]
                    },
                ),
                text_cfg=UGContextHandle(request_id="", token_count=0),
                image_cfg=UGContextHandle(request_id="", token_count=0),
            )
            segment = native_executor.generate(
                contexts=contexts,
                batch=batch,
                server_args=server_args,
                srt_kv_token_binding=condition_binding,
                cfg_img_condition_srt_kv_token_binding=img_condition_binding,
                cfg_uncondition_srt_kv_token_binding=uncondition_binding,
            )
    finally:
        runner.req_to_token_pool.clear()
        runner.token_to_kv_pool_allocator.clear()

    image_output = output_dir / f"{case.case_id}.candidate.png"
    segment.image.save(image_output)
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="sglang",
        image=summarize_ug_image(image_output),
        metadata={
            "candidate_backend": "u1_native_srt_edit_pixel_flow",
            "native_srt_model_runner": True,
            "attention_backend": attention_backend,
            "dtype": dtype,
            "mem_fraction_static": mem_fraction_static,
            "image_path": str(image_output),
            "prefill_forwards": 1,
            "cfg_img_condition_prefill_forwards": (
                1 if img_condition_prepared is not None else 0
            ),
            "cfg_uncondition_prefill_forwards": (
                1 if uncondition_prepared is not None else 0
            ),
            "temp_g_forward_count": int(srt_executor.temp_g_forward_count),
            "temp_g_allocated_token_count": int(
                srt_executor.temp_g_allocated_token_count
            ),
            "g_metadata": dict(segment.metadata),
        },
    )


def _run_sglang_native_interleave_candidate(
    *,
    case: UGParityCase,
    model_path: str,
    prompt: str,
    image_paths: list[Path],
    width: int,
    height: int,
    cfg_scale: float,
    img_cfg_scale: float,
    cfg_interval: tuple[float, float],
    timestep_shift: float,
    num_steps: int,
    seed: int,
    think_mode: bool,
    post_text_max_new_tokens: int,
    max_interleave_images: int,
    max_interleave_text_segments: int,
    dtype: str,
    mem_fraction_static: float,
    attention_backend: str | None,
    cuda_visible_devices: str | None,
    output_dir: Path,
    teacher_force_commit_paths: list[Path] | None = None,
    teacher_force_text_parts: list[str] | None = None,
    teacher_force_token_parts: list[list[int]] | None = None,
    force_u_decode_token_ids: list[int] | None = None,
    dump_debug_tensors: bool = False,
    alignment_gate: str | None = None,
) -> UGParityArtifact:
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    from transformers import AutoTokenizer

    from sglang.multimodal_gen.configs.pipeline_configs.ug import UGPipelineConfig
    from sglang.multimodal_gen.runtime.pipelines.ug import UGPipeline
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ug_u1 import (
        U1PixelFlowGSegmentExecutor,
    )
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs as UGServerArgs,
        set_global_server_args,
    )
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.mm_utils import init_mm_embedding_cache
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.session.session_controller import SessionController
    from sglang.srt.ug.adapter import UGModelRunnerAdapter
    from sglang.srt.ug.interleaved import UGGSegmentResult
    from sglang.srt.ug.runtime import UGSessionRuntime

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    server_args = ServerArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        trust_remote_code=False,
        disable_cuda_graph=True,
        disable_hybrid_swa_memory=True,
        mem_fraction_static=mem_fraction_static,
        chunked_prefill_size=-1,
        dtype=dtype,
        attention_backend=attention_backend,
    )
    port_args = PortArgs.init_new(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
        moe_ep_rank=0,
        moe_ep_size=1,
    )
    model = runner.model
    init_mm_embedding_cache()

    try:
        with torch.no_grad():
            tree_cache = _make_native_tree_cache(runner)
            session_controller = SessionController(_HarnessTreeCache())
            srt_executor = _HarnessNativeSRTExecutor(
                runner=runner,
                model_config=model_config,
                tree_cache=tree_cache,
                session_controller=session_controller,
            )
            if force_u_decode_token_ids is not None:
                srt_executor.force_u_decode_token_ids = list(force_u_decode_token_ids)
            if dump_debug_tensors:
                srt_executor.debug_tensor_dump_dir = str(
                    output_dir / f"{case.case_id}.candidate_debug"
                )
                srt_executor.debug_tensor_dump_max_commit_calls = int(
                    os.environ.get(
                        "SGLANG_TEST_U1_INTERLEAVE_DEBUG_MAX_G_CALLS",
                        "32",
                    )
                )
                srt_executor.debug_g_sublayer_layers = _parse_int_list(
                    os.environ.get("SGLANG_TEST_U1_INTERLEAVE_DEBUG_G_SUBLAYERS"),
                    default=(0,),
                )
            runtime = UGSessionRuntime(
                model_runner=UGModelRunnerAdapter(
                    U1UGModelAdapter(native_tokenizer=tokenizer)
                ),
                session_controller=session_controller,
                srt_request_executor=srt_executor,
                tokenizer=tokenizer,
                vocab_size=getattr(model_config, "vocab_size", 32000),
                srt_u_decode_max_new_tokens=post_text_max_new_tokens,
                srt_image_tokenization="multimodal",
            )
            bridge = U1SRTBackedUGMiddleBridge(runtime)
            if dump_debug_tensors:
                bridge.debug_tensor_dump_dir = str(
                    output_dir / f"{case.case_id}.candidate_debug"
                )
                bridge.debug_tensor_dump_max_g_calls = int(
                    os.environ.get(
                        "SGLANG_TEST_U1_INTERLEAVE_DEBUG_MAX_G_CALLS",
                        "32",
                    )
                )
                bridge.debug_g_sublayer_layers = _parse_int_list(
                    os.environ.get("SGLANG_TEST_U1_INTERLEAVE_DEBUG_G_SUBLAYERS"),
                    default=(0,),
                )
            teacher_forced_commit_count = 0
            if teacher_force_commit_paths:
                from PIL import Image

                original_commit_generated_segment = bridge.commit_generated_segment
                teacher_force_iter = iter(teacher_force_commit_paths)

                def commit_generated_segment_with_reference_image(*, contexts, segment):
                    nonlocal teacher_forced_commit_count
                    try:
                        reference_commit_path = next(teacher_force_iter)
                    except StopIteration:
                        return original_commit_generated_segment(
                            contexts=contexts,
                            segment=segment,
                        )
                    teacher_forced_commit_count += 1
                    if reference_commit_path.suffix == ".pt":
                        raw_pixel_values = torch.load(
                            reference_commit_path,
                            map_location="cpu",
                        )
                        commit_image = {
                            "pixel_values": raw_pixel_values,
                            "value_range": "minus_one_to_one",
                        }
                        if (
                            os.environ.get(
                                "SGLANG_TEST_U1_INTERLEAVE_TEACHER_FORCE_PRECOMPUTE_COMMIT",
                                "1",
                            )
                            != "0"
                        ):
                            commit_embeddings, commit_grid_hw = (
                                _u1_precompute_generated_image_commit_embeddings(
                                    model,
                                    raw_pixel_values,
                                    patch_size=int(getattr(model, "patch_size", 16)),
                                )
                            )
                            commit_image["grid_hw"] = commit_grid_hw
                            commit_image["precomputed_embeddings"] = commit_embeddings
                    else:
                        commit_image = Image.open(reference_commit_path).convert("RGB")
                    forced_segment = UGGSegmentResult(
                        type=segment.type,
                        image=segment.image,
                        metadata=dict(segment.metadata),
                        commit_image=commit_image,
                    )
                    return original_commit_generated_segment(
                        contexts=contexts,
                        segment=forced_segment,
                    )

                bridge.commit_generated_segment = (
                    commit_generated_segment_with_reference_image
                )
            else:
                teacher_forced_commit_count = 0
            teacher_forced_text_count = 0
            teacher_forced_marker_count = 0
            if teacher_force_token_parts is not None:
                teacher_force_text_parts = [
                    tokenizer.decode(token_ids, skip_special_tokens=True)
                    for token_ids in teacher_force_token_parts
                ]
            if teacher_force_text_parts:
                from sglang.srt.ug.denoiser import normalize_ug_interleaved_messages

                reference_decode_outputs = []
                for index, text in enumerate(teacher_force_text_parts):
                    token_ids = (
                        teacher_force_token_parts[index]
                        if teacher_force_token_parts is not None
                        else None
                    )
                    if text or token_ids:
                        reference_decode_outputs.append(("text", text, token_ids))
                    if index < len(teacher_force_text_parts) - 1:
                        reference_decode_outputs.append(("image_marker", None, None))
                reference_decode_iter = iter(reference_decode_outputs)
                original_prepare_u_context_from_messages = (
                    bridge.prepare_u_context_from_messages
                )
                original_continue_u_decode = bridge.continue_u_decode

                def next_reference_decode(contexts):
                    nonlocal teacher_forced_marker_count
                    nonlocal teacher_forced_text_count

                    try:
                        output_type, text, token_ids = next(reference_decode_iter)
                    except StopIteration:
                        return None

                    if output_type == "text":
                        _append_u1_reference_text(
                            runtime=runtime,
                            bridge=bridge,
                            contexts=contexts,
                            tokenizer=tokenizer,
                            text=text,
                            token_ids=token_ids,
                        )
                        teacher_forced_text_count += 1
                        return UGDecodeResult(type="text", text=text)

                    _append_u1_reference_image_marker(
                        runtime=runtime,
                        bridge=bridge,
                        contexts=contexts,
                        tokenizer=tokenizer,
                    )
                    teacher_forced_marker_count += 1
                    return UGDecodeResult(type="image_marker")

                def prepare_u_context_with_reference_text(
                    *,
                    messages,
                    think=False,
                    think_max_new_tokens=None,
                    sampling_params=None,
                ):
                    if think:
                        return original_prepare_u_context_from_messages(
                            messages=messages,
                            think=think,
                            think_max_new_tokens=think_max_new_tokens,
                            sampling_params=sampling_params,
                        )
                    normalized_messages = normalize_ug_interleaved_messages(messages)
                    with bridge._temporary_generation_settings(sampling_params):
                        session = runtime.prefill_interleaved(normalized_messages)
                    contexts = UGContextBundle(
                        full=UGContextHandle(
                            session.anchor_request_id,
                            session.context_length,
                            session=session,
                            metadata={"pre_image_segments": []},
                        ),
                        text_cfg=UGContextHandle(
                            f"{session.anchor_request_id}:text_cfg",
                            sum(
                                2
                                for message in normalized_messages
                                if message.type == "image"
                            ),
                            session=session,
                        ),
                        image_cfg=UGContextHandle(
                            f"{session.anchor_request_id}:image_cfg",
                            sum(
                                len(str(message.content).split())
                                for message in normalized_messages
                                if message.type == "text"
                            ),
                            session=session,
                        ),
                    )
                    bridge._attach_u1_context_metadata(contexts)
                    pre_image_segments = contexts.full.metadata["pre_image_segments"]
                    while True:
                        segment = next_reference_decode(contexts)
                        if segment is None:
                            raise RuntimeError(
                                "U1 interleave reference text teacher-force ran out "
                                "before the first image marker"
                            )
                        if segment.type == "text":
                            pre_image_segments.append(
                                {"type": "text", "text": segment.text or ""}
                            )
                            continue
                        if segment.type == "image_marker":
                            return contexts
                        raise ValueError(
                            "U1 interleave teacher-force expected text or marker, "
                            f"got {segment.type}"
                        )

                bridge.prepare_u_context_from_messages = (
                    prepare_u_context_with_reference_text
                )

                def continue_u_decode_with_reference_text(*, contexts):
                    segment = next_reference_decode(contexts)
                    if segment is None:
                        return original_continue_u_decode(contexts=contexts)
                    return segment

                bridge.continue_u_decode = continue_u_decode_with_reference_text
            sampling_params = UGSamplingParams(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                cfg_text_scale=cfg_scale,
                cfg_img_scale=img_cfg_scale,
                cfg_interval=list(cfg_interval),
                cfg_renorm_type="none",
                timestep_shift=timestep_shift,
                think=think_mode,
                think_max_new_tokens=max(1, min(post_text_max_new_tokens, 8)),
            )
            messages = [
                {"type": "image", "image": str(image_path)}
                for image_path in image_paths
            ]
            messages.append({"type": "text", "text": prompt})
            request = UGInterleavedRequest.from_segments(
                messages,
                sampling_params=sampling_params,
                metadata={
                    "mode": "interleave",
                    "max_interleave_images": max_interleave_images,
                    "max_interleave_text_segments": max_interleave_text_segments,
                },
            )
            ug_server_args = UGServerArgs(
                model_path=model_path,
                num_gpus=1,
                enable_cfg_parallel=False,
                pipeline_config=UGPipelineConfig(
                    default_height=height,
                    default_width=width,
                ),
            )
            set_global_server_args(ug_server_args)
            pipeline = UGPipeline(
                model_path,
                ug_server_args,
                loaded_modules={
                    "ug_bridge": bridge,
                    "ug_g_segment_executor": U1PixelFlowGSegmentExecutor(),
                },
                executor=SimpleNamespace(),
            )
            response = pipeline.forward_interleaved(request, server_args=ug_server_args)
    finally:
        runner.req_to_token_pool.clear()
        runner.token_to_kv_pool_allocator.clear()

    output_segments = response.to_segments()
    generated_images = [
        segment["image"]
        for segment in output_segments
        if segment.get("type") == "image" and segment.get("image") is not None
    ]
    generated_text = "\n".join(
        str(segment.get("text") or "")
        for segment in output_segments
        if segment.get("type") == "text"
    ).strip()
    image_output = output_dir / f"{case.case_id}.candidate.png"
    if not generated_images:
        return UGParityArtifact(
            case_id=case.case_id,
            model=case.model,
            task=case.task,
            runner="sglang",
            text=generated_text,
            metadata={
                "candidate_backend": "u1_native_srt_interleave_pixel_flow",
                "native_srt_model_runner": True,
                "segments": output_segments,
            },
            error="sglang_interleave_failed: missing_generated_image",
        )
    image_outputs = [
        output_dir / f"{case.case_id}.candidate_image_{index}.png"
        for index in range(len(generated_images))
    ]
    for image, path in zip(generated_images, image_outputs):
        image.save(path)
    generated_images[0].save(image_output)
    stats = response.stats
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="sglang",
        text=generated_text,
        image=summarize_ug_image(image_output),
        metadata={
            "candidate_backend": "u1_native_srt_interleave_pixel_flow",
            "alignment_gate": alignment_gate,
            "native_srt_model_runner": True,
            "attention_backend": attention_backend,
            "dtype": dtype,
            "mem_fraction_static": mem_fraction_static,
            "image_path": str(image_output),
            "image_paths": [str(path) for path in image_outputs],
            "segments": _summarize_output_segments(output_segments, image_outputs),
            "same_session_id": stats is not None and bool(stats.session_id),
            "session_id": stats.session_id if stats is not None else None,
            "prefill_count": stats.prefill_count if stats is not None else None,
            "append_image_count": (
                stats.append_image_count if stats is not None else None
            ),
            "decode_count": stats.decode_count if stats is not None else None,
            "srt_request_count": (
                stats.srt_request_count if stats is not None else None
            ),
            "srt_executed_request_count": (
                stats.srt_executed_request_count if stats is not None else None
            ),
            "srt_u_decode_request_count": (
                stats.srt_u_decode_request_count if stats is not None else None
            ),
            "temp_g_forward_count": int(srt_executor.temp_g_forward_count),
            "temp_g_allocated_token_count": int(
                srt_executor.temp_g_allocated_token_count
            ),
            "teacher_forced_reference_commit_count": teacher_forced_commit_count,
            "teacher_forced_reference_text_count": teacher_forced_text_count,
            "teacher_forced_reference_marker_count": teacher_forced_marker_count,
            "teacher_forced_reference_token_count": (
                sum(len(part) for part in teacher_force_token_parts)
                if teacher_force_token_parts is not None
                else None
            ),
            "forced_u_decode_token_count": (
                len(force_u_decode_token_ids)
                if force_u_decode_token_ids is not None
                else None
            ),
            "forced_u_decode_record_count": len(srt_executor.forced_u_decode_records),
            "forced_u_decode_greedy_prefix_match_count": (
                srt_executor.forced_u_decode_greedy_prefix_match_count()
            ),
            "forced_u_decode_first_greedy_mismatch": (
                srt_executor.forced_u_decode_first_greedy_mismatch()
            ),
            "forced_u_decode_records_preview": (
                srt_executor.forced_u_decode_records[:16]
            ),
            "debug_tensor_dump_dir": (
                str(output_dir / f"{case.case_id}.candidate_debug")
                if dump_debug_tensors
                else None
            ),
        },
    )


def _write_official_interleave_raw_dump_script(script_path: Path) -> None:
    script_path.write_text(
        r"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


def _load_official_inference(repo: Path):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    inference_path = repo / "examples" / "interleave" / "inference.py"
    spec = importlib.util.spec_from_file_location(
        "sensenova_u1_official_interleave_inference",
        inference_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official_repo", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--cfg_scale", type=float, required=True)
    parser.add_argument("--img_cfg_scale", type=float, required=True)
    parser.add_argument("--timestep_shift", type=float, required=True)
    parser.add_argument("--cfg_interval", type=float, nargs=2, required=True)
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], required=True)
    parser.add_argument("--attn_backend", choices=["auto", "flash", "sdpa"], required=True)
    parser.add_argument("--think_mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--dump_debug_tensors", action="store_true")
    parser.add_argument("--debug_max_g_calls", type=int, default=32)
    parser.add_argument("--debug_g_sublayers", default="0")
    return parser.parse_args()


def _json_int(value):
    if torch.is_tensor(value):
        return int(value.detach().max().item())
    return int(value)


def _debug_topk_payload(logits, k=8):
    logits = logits.detach()
    topk = torch.topk(logits[0].float(), k=min(k, logits.shape[-1]))
    return {
        "selected_token": int(torch.argmax(logits[0]).item()),
        "topk_ids": topk.indices.detach().cpu(),
        "topk_values": topk.values.detach().cpu(),
    }


def _last_token_hidden_state(hidden_states):
    if not torch.is_tensor(hidden_states):
        return hidden_states
    if hidden_states.ndim == 3:
        return hidden_states[:, -1:, :].reshape(-1, hidden_states.shape[-1])
    if hidden_states.ndim == 2:
        return hidden_states[-1:, :]
    return hidden_states


def _debug_g_branch_sequence(timestep, cfg_scale, img_cfg_scale, cfg_interval):
    t = float(timestep.detach().float().cpu().item())
    use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or cfg_interval[0] == 0
    if not use_cfg or (cfg_scale == 1 and img_cfg_scale == 1):
        return ("condition",)
    if img_cfg_scale == 1:
        return ("condition", "text_uncondition")
    if cfg_scale == img_cfg_scale:
        return ("condition", "img_uncondition")
    return ("condition", "text_uncondition", "img_uncondition")


def _parse_int_set(value, default):
    if value is None:
        return set(default)
    parts = [
        part.strip()
        for part in str(value).replace(",", " ").split()
        if part.strip()
    ]
    if not parts:
        return set(default)
    return {int(part) for part in parts}


def main() -> None:
    args = _parse_args()
    repo = Path(args.official_repo)
    official = _load_official_inference(repo)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    official.sensenova_u1.set_attn_backend(args.attn_backend)
    print(
        f"[attn] backend={args.attn_backend!r} "
        f"(effective={official.sensenova_u1.effective_attn_backend()!r})"
    )
    engine = official.SenseNovaU1Interleave(
        args.model_path,
        device=args.device,
        dtype=dtype,
    )
    generated_token_chunks = []
    original_tokenizer_decode = engine.tokenizer.decode

    def wrapped_tokenizer_decode(token_ids, *args_, **kwargs_):
        if kwargs_.get("skip_special_tokens") is True:
            if torch.is_tensor(token_ids):
                ids = token_ids.detach().cpu().view(-1).tolist()
            elif isinstance(token_ids, (list, tuple)):
                ids = list(token_ids)
            else:
                ids = [int(token_ids)]
            generated_token_chunks.append([int(token_id) for token_id in ids])
        return original_tokenizer_decode(token_ids, *args_, **kwargs_)

    engine.tokenizer.decode = wrapped_tokenizer_decode
    debug_dir = None
    debug_records = []
    if args.dump_debug_tensors:
        debug_dir = Path(args.output_dir) / f"{args.stem}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_g_sublayers = _parse_int_set(args.debug_g_sublayers, {0})
        original_predict_v = engine.model._t2i_predict_v
        predict_call_index = 0
        predict_step_timestep = None
        predict_step_branch_index = 0
        original_model_forward = engine.model.language_model.model.forward
        last_g_forward_payload = None
        last_u_model_hidden_states = None
        current_u_layer_hidden_states = []
        current_u_sublayer_states = []
        current_g_layer_hidden_states = []
        current_g_sublayer_states = []
        current_g_sublayer_layer = 0

        def record_u_sublayer_state(name, hidden_states):
            if torch.is_tensor(hidden_states):
                current_u_sublayer_states.append(
                    {
                        "layer": 0,
                        "name": str(name),
                        "hidden_states": _last_token_hidden_state(
                            hidden_states
                        ).detach().float().cpu(),
                    }
                )

        def record_g_sublayer_state(name, hidden_states):
            if torch.is_tensor(hidden_states):
                current_g_sublayer_states.append(
                    {
                        "layer": int(current_g_sublayer_layer),
                        "name": str(name),
                        "hidden_states": hidden_states.detach().float().cpu(),
                    }
                )

        def wrapped_model_forward(*args_, **kwargs_):
            nonlocal last_g_forward_payload
            nonlocal last_u_model_hidden_states
            nonlocal current_g_layer_hidden_states
            nonlocal current_g_sublayer_states
            outputs = original_model_forward(*args_, **kwargs_)
            image_gen_indicators = kwargs_.get("image_gen_indicators")
            hidden_states = getattr(outputs, "last_hidden_state", None)
            layer_hidden_states = current_g_layer_hidden_states
            sublayer_states = current_g_sublayer_states
            current_g_layer_hidden_states = []
            current_g_sublayer_states = []
            if (
                image_gen_indicators is not None
                and hidden_states is not None
                and bool(image_gen_indicators.detach().bool().all().item())
            ):
                last_g_forward_payload = {
                    "hidden_states": hidden_states.detach().float().cpu(),
                    "layer_hidden_states": layer_hidden_states,
                    "sublayer_states": sublayer_states,
                }
            elif hidden_states is not None:
                last_u_model_hidden_states = _last_token_hidden_state(
                    hidden_states
                ).detach().float().cpu()
            return outputs

        engine.model.language_model.model.forward = wrapped_model_forward
        original_layer_forward_gens = []
        record_layer0_u_attention = False
        record_layer0_g_attention = False
        current_u_rotary_names = None
        current_rotary_names = None

        def make_wrapped_layer_forward_u(layer_index, original_forward):
            def wrapped_layer_forward_u(*args_, **kwargs_):
                nonlocal current_u_rotary_names
                nonlocal record_layer0_u_attention
                layer_input = kwargs_.get("hidden_states")
                if layer_input is None and args_:
                    layer_input = args_[0]
                if int(layer_index) == 0:
                    record_u_sublayer_state("layer_input", layer_input)
                    record_layer0_u_attention = True
                    current_u_rotary_names = iter(
                        (
                            ("attn_q_rope_t", "attn_k_rope_t"),
                            ("attn_q_rope_h", "attn_k_rope_h"),
                            ("attn_q_rope_w", "attn_k_rope_w"),
                        )
                    )
                try:
                    out = original_forward(*args_, **kwargs_)
                finally:
                    if int(layer_index) == 0:
                        record_layer0_u_attention = False
                hidden_states = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(hidden_states):
                    current_u_layer_hidden_states.append(
                        {
                            "layer": int(layer_index),
                            "hidden_states": _last_token_hidden_state(
                                hidden_states
                            ).detach().float().cpu(),
                        }
                    )
                    if int(layer_index) == 0:
                        record_u_sublayer_state("layer_out", hidden_states)
                return out

            return wrapped_layer_forward_u

        def make_wrapped_layer_forward_gen(layer_index, original_forward_gen):
            def wrapped_layer_forward_gen(*args_, **kwargs_):
                nonlocal current_rotary_names
                nonlocal record_layer0_g_attention
                nonlocal current_g_sublayer_layer
                arg_names = (
                    "hidden_states",
                    "image_gen_indicators",
                    "exist_non_image_gen_tokens",
                    "exist_image_gen_tokens",
                    "indexes",
                    "attention_mask",
                    "position_ids",
                    "past_key_values",
                    "use_cache",
                    "cache_position",
                )
                call_kwargs = dict(kwargs_)
                for arg_name, arg_value in zip(arg_names, args_):
                    call_kwargs.setdefault(arg_name, arg_value)
                layer_input = call_kwargs.get("hidden_states")
                capture_basic_sublayers = int(layer_index) in debug_g_sublayers
                if capture_basic_sublayers and int(layer_index) != 0:
                    current_g_sublayer_layer = int(layer_index)
                    try:
                        layer = engine.model.language_model.model.layers[
                            int(layer_index)
                        ]
                        hidden_states = layer_input
                        residual = hidden_states
                        record_g_sublayer_state("layer_input", hidden_states)
                        hidden_states = layer.input_layernorm_mot_gen(hidden_states)
                        record_g_sublayer_state("input_norm", hidden_states)
                        attn_kwargs = dict(call_kwargs)
                        attn_kwargs.pop("hidden_states", None)
                        hidden_states, _ = layer.self_attn(
                            hidden_states=hidden_states,
                            **attn_kwargs,
                        )
                        record_g_sublayer_state("attn_out", hidden_states)
                        hidden_states = residual + hidden_states
                        record_g_sublayer_state(
                            "post_attn_residual",
                            hidden_states,
                        )
                        residual = hidden_states
                        hidden_states = layer.post_attention_layernorm_mot_gen(
                            hidden_states
                        )
                        record_g_sublayer_state("post_attn_norm", hidden_states)
                        hidden_states = layer.mlp_mot_gen(hidden_states)
                        record_g_sublayer_state("mlp_out", hidden_states)
                        hidden_states = residual + hidden_states
                        record_g_sublayer_state("layer_out", hidden_states)
                        current_g_layer_hidden_states.append(
                            {
                                "layer": int(layer_index),
                                "hidden_states": hidden_states.detach().float().cpu(),
                            }
                        )
                        return hidden_states
                    finally:
                        current_g_sublayer_layer = 0
                if capture_basic_sublayers:
                    current_g_sublayer_layer = int(layer_index)
                    record_g_sublayer_state("layer_input", layer_input)
                if int(layer_index) == 0:
                    record_layer0_g_attention = True
                    current_rotary_names = iter(
                        (
                            ("attn_q_rope_t", "attn_k_rope_t"),
                            ("attn_q_rope_h", "attn_k_rope_h"),
                            ("attn_q_rope_w", "attn_k_rope_w"),
                        )
                    )
                try:
                    out = original_forward_gen(*args_, **kwargs_)
                finally:
                    if int(layer_index) == 0:
                        record_layer0_g_attention = False
                    current_g_sublayer_layer = 0
                hidden_states = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(hidden_states):
                    current_g_layer_hidden_states.append(
                        {
                            "layer": int(layer_index),
                            "hidden_states": hidden_states.detach().float().cpu(),
                        }
                    )
                    if int(layer_index) == 0:
                        record_g_sublayer_state("layer_out", hidden_states)
                return out

            return wrapped_layer_forward_gen

        layer0 = engine.model.language_model.model.layers[0]

        def wrap_layer0_u_module_forward(module, name):
            original_forward = module.forward

            def wrapped_forward(*args_, **kwargs_):
                out = original_forward(*args_, **kwargs_)
                if record_layer0_u_attention:
                    hidden_states = out[0] if isinstance(out, tuple) else out
                    record_u_sublayer_state(name, hidden_states)
                return out

            module.forward = wrapped_forward

        def wrap_layer0_module_forward(module, name):
            original_forward = module.forward

            def wrapped_forward(*args_, **kwargs_):
                out = original_forward(*args_, **kwargs_)
                hidden_states = out[0] if isinstance(out, tuple) else out
                record_g_sublayer_state(name, hidden_states)
                return out

            module.forward = wrapped_forward

        wrap_layer0_module_forward(layer0.input_layernorm_mot_gen, "input_norm")

        def wrap_layer0_u_attention_projection(module, output_name):
            original_forward = module.forward

            def wrapped_forward(*args_, **kwargs_):
                out = original_forward(*args_, **kwargs_)
                if record_layer0_u_attention:
                    record_u_sublayer_state(output_name, out)
                return out

            module.forward = wrapped_forward

        def wrap_layer0_attention_projection(module, output_name):
            original_forward = module.forward

            def wrapped_forward(*args_, **kwargs_):
                out = original_forward(*args_, **kwargs_)
                record_g_sublayer_state(output_name, out)
                return out

            module.forward = wrapped_forward

        def wrap_layer0_u_attention_output_projection(module):
            original_forward = module.forward

            def wrapped_forward(*args_, **kwargs_):
                if record_layer0_u_attention and args_:
                    record_u_sublayer_state("attn_context", args_[0])
                out = original_forward(*args_, **kwargs_)
                if record_layer0_u_attention:
                    record_u_sublayer_state("attn_o_proj_out", out)
                return out

            module.forward = wrapped_forward

        def wrap_layer0_attention_output_projection(module):
            original_forward = module.forward

            def wrapped_forward(*args_, **kwargs_):
                if args_:
                    record_g_sublayer_state("attn_context", args_[0])
                out = original_forward(*args_, **kwargs_)
                record_g_sublayer_state("attn_o_proj_out", out)
                return out

            module.forward = wrapped_forward

        wrap_layer0_u_attention_projection(layer0.self_attn.q_proj, "attn_q_proj")
        wrap_layer0_u_attention_projection(layer0.self_attn.k_proj, "attn_k_proj")
        wrap_layer0_u_attention_projection(layer0.self_attn.v_proj, "attn_v_proj")
        wrap_layer0_attention_projection(layer0.self_attn.q_proj_mot_gen, "attn_q_proj")
        wrap_layer0_attention_projection(layer0.self_attn.k_proj_mot_gen, "attn_k_proj")
        wrap_layer0_attention_projection(layer0.self_attn.v_proj_mot_gen, "attn_v_proj")

        def flatten_head_tensor(x):
            if not torch.is_tensor(x):
                return x
            if x.ndim == 4 and x.shape[0] == 1:
                return x.squeeze(0).reshape(x.shape[1], -1)
            return x

        def flatten_bhsd_tensor(x):
            if not torch.is_tensor(x):
                return x
            if x.ndim == 4 and x.shape[0] == 1:
                return x.transpose(1, 2).squeeze(0).reshape(x.shape[2], -1)
            return x

        def wrap_layer0_norm(module, output_name, split_names=None):
            original_forward = module.forward

            def wrapped_forward(*args_, **kwargs_):
                out = original_forward(*args_, **kwargs_)
                if not record_layer0_g_attention:
                    return out
                if split_names is None:
                    record_g_sublayer_state(output_name, flatten_head_tensor(out))
                else:
                    left, right = out.chunk(2, dim=-1)
                    record_g_sublayer_state(split_names[0], flatten_head_tensor(left))
                    record_g_sublayer_state(split_names[1], flatten_head_tensor(right))
                return out

            module.forward = wrapped_forward

        def wrap_layer0_u_norm(module, output_name, split_names=None):
            original_forward = module.forward

            def wrapped_forward(*args_, **kwargs_):
                out = original_forward(*args_, **kwargs_)
                if not record_layer0_u_attention:
                    return out
                if split_names is None:
                    record_u_sublayer_state(output_name, flatten_head_tensor(out))
                else:
                    left, right = out.chunk(2, dim=-1)
                    record_u_sublayer_state(split_names[0], flatten_head_tensor(left))
                    record_u_sublayer_state(split_names[1], flatten_head_tensor(right))
                return out

            module.forward = wrapped_forward

        wrap_layer0_u_module_forward(layer0.input_layernorm, "input_norm")
        wrap_layer0_u_norm(layer0.self_attn.q_norm, "attn_q_norm_t")
        wrap_layer0_u_norm(
            layer0.self_attn.q_norm_hw,
            None,
            split_names=("attn_q_norm_h", "attn_q_norm_w"),
        )
        wrap_layer0_u_norm(layer0.self_attn.k_norm, "attn_k_norm_t")
        wrap_layer0_u_norm(
            layer0.self_attn.k_norm_hw,
            None,
            split_names=("attn_k_norm_h", "attn_k_norm_w"),
        )

        wrap_layer0_norm(layer0.self_attn.q_norm_mot_gen, "attn_q_norm_t")
        wrap_layer0_norm(
            layer0.self_attn.q_norm_hw_mot_gen,
            None,
            split_names=("attn_q_norm_h", "attn_q_norm_w"),
        )
        wrap_layer0_norm(layer0.self_attn.k_norm_mot_gen, "attn_k_norm_t")
        wrap_layer0_norm(
            layer0.self_attn.k_norm_hw_mot_gen,
            None,
            split_names=("attn_k_norm_h", "attn_k_norm_w"),
        )

        attn_module = sys.modules[layer0.self_attn.__class__.__module__]
        original_apply_rotary_pos_emb = attn_module.apply_rotary_pos_emb

        def wrapped_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            q_out, k_out = original_apply_rotary_pos_emb(
                q,
                k,
                cos,
                sin,
                position_ids=position_ids,
                unsqueeze_dim=unsqueeze_dim,
            )
            if record_layer0_u_attention and current_u_rotary_names is not None:
                try:
                    q_name, k_name = next(current_u_rotary_names)
                except StopIteration:
                    return q_out, k_out
                record_u_sublayer_state(q_name, flatten_bhsd_tensor(q_out))
                record_u_sublayer_state(k_name, flatten_bhsd_tensor(k_out))
                return q_out, k_out
            if not record_layer0_g_attention or current_rotary_names is None:
                return q_out, k_out
            try:
                q_name, k_name = next(current_rotary_names)
            except StopIteration:
                return q_out, k_out
            record_g_sublayer_state(q_name, flatten_bhsd_tensor(q_out))
            record_g_sublayer_state(k_name, flatten_bhsd_tensor(k_out))
            return q_out, k_out

        attn_module.apply_rotary_pos_emb = wrapped_apply_rotary_pos_emb

        original_eager_attention_forward = attn_module.eager_attention_forward

        def wrapped_eager_attention_forward(module, q, k, v, *args_, **kwargs_):
            if not record_layer0_u_attention:
                return original_eager_attention_forward(
                    module,
                    q,
                    k,
                    v,
                    *args_,
                    **kwargs_,
                )
            q_len = q.shape[2]
            record_u_sublayer_state("attn_q_ready", flatten_bhsd_tensor(q))
            record_u_sublayer_state(
                "attn_k_ready",
                flatten_bhsd_tensor(k[:, :, -q_len:]),
            )
            record_u_sublayer_state(
                "attn_v_ready",
                flatten_bhsd_tensor(v[:, :, -q_len:]),
            )
            record_u_sublayer_state("attn_k_full", flatten_bhsd_tensor(k))
            record_u_sublayer_state("attn_v_full", flatten_bhsd_tensor(v))
            out = original_eager_attention_forward(
                module,
                q,
                k,
                v,
                *args_,
                **kwargs_,
            )
            attn_output = out[0] if isinstance(out, tuple) else out
            record_u_sublayer_state("attn_context", flatten_head_tensor(attn_output))
            return out

        attn_module.eager_attention_forward = wrapped_eager_attention_forward

        original_flash_or_sdpa = attn_module._flash_or_sdpa

        def wrapped_flash_or_sdpa(q, k, v, *args_, **kwargs_):
            if not record_layer0_g_attention:
                return original_flash_or_sdpa(q, k, v, *args_, **kwargs_)
            q_len = q.shape[1]
            record_g_sublayer_state("attn_q_ready", flatten_head_tensor(q))
            record_g_sublayer_state("attn_k_ready", flatten_head_tensor(k[:, -q_len:]))
            record_g_sublayer_state("attn_v_ready", flatten_head_tensor(v[:, -q_len:]))
            record_g_sublayer_state("attn_k_full", flatten_head_tensor(k))
            record_g_sublayer_state("attn_v_full", flatten_head_tensor(v))
            return original_flash_or_sdpa(q, k, v, *args_, **kwargs_)

        attn_module._flash_or_sdpa = wrapped_flash_or_sdpa

        wrap_layer0_u_attention_output_projection(layer0.self_attn.o_proj)
        wrap_layer0_attention_output_projection(layer0.self_attn.o_proj_mot_gen)
        wrap_layer0_u_module_forward(layer0.self_attn, "attn_out")
        wrap_layer0_module_forward(layer0.self_attn, "attn_out")
        wrap_layer0_u_module_forward(
            layer0.post_attention_layernorm,
            "post_attn_norm",
        )
        wrap_layer0_module_forward(
            layer0.post_attention_layernorm_mot_gen,
            "post_attn_norm",
        )

        def wrapped_layer0_u_mlp_forward(x):
            gate = layer0.mlp.gate_proj(x)
            record_u_sublayer_state("mlp_gate", gate)
            up = layer0.mlp.up_proj(x)
            record_u_sublayer_state("mlp_up", up)
            act = layer0.mlp.act_fn(gate)
            record_u_sublayer_state("mlp_act", act)
            act_mul = act * up
            record_u_sublayer_state("mlp_act_mul", act_mul)
            out = layer0.mlp.down_proj(act_mul)
            record_u_sublayer_state("mlp_out", out)
            return out

        layer0.mlp.forward = wrapped_layer0_u_mlp_forward

        def wrapped_layer0_mlp_forward(x):
            gate = layer0.mlp_mot_gen.gate_proj(x)
            record_g_sublayer_state("mlp_gate", gate)
            up = layer0.mlp_mot_gen.up_proj(x)
            record_g_sublayer_state("mlp_up", up)
            act = layer0.mlp_mot_gen.act_fn(gate)
            record_g_sublayer_state("mlp_act", act)
            act_mul = act * up
            record_g_sublayer_state("mlp_act_mul", act_mul)
            out = layer0.mlp_mot_gen.down_proj(act_mul)
            record_g_sublayer_state("mlp_out", out)
            return out

        layer0.mlp_mot_gen.forward = wrapped_layer0_mlp_forward

        for layer_index, layer in enumerate(engine.model.language_model.model.layers):
            layer.forward = make_wrapped_layer_forward_u(
                layer_index,
                layer.forward,
            )
            original_forward_gen = layer.forward_gen
            original_layer_forward_gens.append(original_forward_gen)
            layer.forward_gen = make_wrapped_layer_forward_gen(
                layer_index,
                original_forward_gen,
            )

        def wrapped_predict_v(
            input_embeds,
            indexes_image,
            attn_mask,
            past_key_values,
            t,
            z,
            image_token_num,
            timestep_embeddings=None,
            image_size=None,
        ):
            nonlocal predict_call_index
            nonlocal predict_step_branch_index
            nonlocal predict_step_timestep
            nonlocal last_g_forward_payload
            out = original_predict_v(
                input_embeds,
                indexes_image,
                attn_mask,
                past_key_values,
                t,
                z,
                image_token_num,
                timestep_embeddings=timestep_embeddings,
                image_size=image_size,
            )
            call_index = predict_call_index
            predict_call_index += 1
            current_timestep = float(t.detach().float().cpu().item())
            if (
                predict_step_timestep is None
                or abs(current_timestep - predict_step_timestep) > 1e-7
            ):
                predict_step_timestep = current_timestep
                predict_step_branch_index = 0
            branch_sequence = _debug_g_branch_sequence(
                t,
                args.cfg_scale,
                args.img_cfg_scale,
                tuple(args.cfg_interval),
            )
            if predict_step_branch_index < len(branch_sequence):
                branch = branch_sequence[predict_step_branch_index]
            else:
                branch = f"branch_{predict_step_branch_index}"
            predict_step_branch_index += 1
            if call_index < args.debug_max_g_calls:
                payload = {
                    "call_index": call_index,
                    "branch": branch,
                    "branch_index_in_step": predict_step_branch_index - 1,
                    "branch_sequence": branch_sequence,
                    "input_embeds": input_embeds.detach().float().cpu(),
                    "v": out.detach().float().cpu(),
                    "indexes_image": indexes_image.detach().cpu(),
                    "timestep": float(t.detach().float().cpu().item()),
                    "z": z.detach().float().cpu(),
                }
                for layer_id in (0, 1, 2):
                    layer = past_key_values.layers[layer_id]
                    payload[f"prefix_layer{layer_id}_keys"] = (
                        layer.keys.detach().float().cpu()
                    )
                    payload[f"prefix_layer{layer_id}_values"] = (
                        layer.values.detach().float().cpu()
                    )
                if last_g_forward_payload is not None:
                    payload.update(last_g_forward_payload)
                    last_g_forward_payload = None
                path = debug_dir / f"official_g_call_{call_index:04d}.pt"
                torch.save(payload, path)
                debug_records.append(
                    {
                        "kind": "g_call",
                        "call_index": call_index,
                        "branch": branch,
                        "path": str(path),
                    }
                )
            return out

        engine.model._t2i_predict_v = wrapped_predict_v
        original_lm_forward = engine.model.language_model.forward
        commit_call_index = 0
        prefix_kv_call_index = 0
        u_logits_call_index = 0

        def wrapped_lm_forward(*args_, **kwargs_):
            nonlocal commit_call_index
            nonlocal prefix_kv_call_index
            nonlocal u_logits_call_index
            nonlocal last_u_model_hidden_states
            nonlocal current_u_layer_hidden_states
            nonlocal current_u_sublayer_states
            outputs = original_lm_forward(*args_, **kwargs_)
            u_hidden_states = last_u_model_hidden_states
            last_u_model_hidden_states = None
            u_layer_hidden_states = current_u_layer_hidden_states
            u_sublayer_states = current_u_sublayer_states
            current_u_layer_hidden_states = []
            current_u_sublayer_states = []
            input_ids = kwargs_.get("input_ids")
            inputs_embeds = kwargs_.get("inputs_embeds")
            past_key_values = kwargs_.get("past_key_values")
            indexes = kwargs_.get("indexes")
            output_past_key_values = getattr(outputs, "past_key_values", None)
            logits = getattr(outputs, "logits", None)
            if (
                logits is not None
                and (inputs_embeds is not None or input_ids is not None)
                and u_logits_call_index < args.debug_max_g_calls
            ):
                logits_path = debug_dir / f"official_u_logits_{u_logits_call_index:04d}.pt"
                payload = {
                    "call_index": u_logits_call_index,
                    "logits": logits[:, -1, :].detach().float().cpu(),
                    "input_len": (
                        int(inputs_embeds.shape[1])
                        if inputs_embeds is not None
                        else int(input_ids.numel())
                    ),
                    "has_past_key_values": past_key_values is not None,
                    "has_output_past_key_values": output_past_key_values is not None,
                    "current_index": getattr(
                        engine.model.language_model.model,
                        "current_index",
                        None,
                    ),
                    **_debug_topk_payload(logits[:, -1, :]),
                }
                if indexes is not None:
                    payload["indexes"] = indexes.detach().cpu()
                if inputs_embeds is not None:
                    payload["input_embeds"] = inputs_embeds.detach().float().cpu()
                if input_ids is not None:
                    payload["input_ids"] = input_ids.detach().cpu()
                if u_hidden_states is not None:
                    payload["hidden_states"] = u_hidden_states
                if u_layer_hidden_states:
                    payload["layer_hidden_states"] = u_layer_hidden_states
                if u_sublayer_states:
                    payload["sublayer_states"] = u_sublayer_states
                if output_past_key_values is not None:
                    layer0 = output_past_key_values.layers[0]
                    if getattr(layer0, "keys", None) is not None:
                        payload["layer0_keys"] = layer0.keys.detach().float().cpu()
                    if getattr(layer0, "values", None) is not None:
                        payload["layer0_values"] = (
                            layer0.values.detach().float().cpu()
                        )
                torch.save(payload, logits_path)
                debug_records.append(
                    {
                        "kind": "u_logits",
                        "call_index": u_logits_call_index,
                        "path": str(logits_path),
                    }
                )
                u_logits_call_index += 1
            if (
                inputs_embeds is not None
                and past_key_values is None
                and output_past_key_values is not None
                and indexes is not None
                and prefix_kv_call_index < args.debug_max_g_calls
            ):
                prefix_path = (
                    debug_dir
                    / f"official_prefix_kv_{prefix_kv_call_index:04d}.pt"
                )
                payload = {
                    "call_index": prefix_kv_call_index,
                    "input_embeds": inputs_embeds.detach().float().cpu(),
                    "indexes": indexes.detach().cpu(),
                }
                for layer_id in (0, 1, 2):
                    layer = output_past_key_values.layers[layer_id]
                    payload[f"layer{layer_id}_keys"] = (
                        layer.keys.detach().float().cpu()
                    )
                    payload[f"layer{layer_id}_values"] = (
                        layer.values.detach().float().cpu()
                    )
                torch.save(payload, prefix_path)
                debug_records.append(
                    {
                        "kind": "prefix_kv",
                        "call_index": prefix_kv_call_index,
                        "path": str(prefix_path),
                    }
                )
                prefix_kv_call_index += 1
            if (
                inputs_embeds is not None
                and past_key_values is not None
                and indexes is not None
                and inputs_embeds.shape[1] > 1
                and commit_call_index < args.debug_max_g_calls
            ):
                if logits is not None:
                    path = debug_dir / f"official_commit_logits_{commit_call_index:04d}.pt"
                    payload = {
                        "call_index": commit_call_index,
                        "logits": logits[:, -1, :].detach().float().cpu(),
                        "input_embeds": inputs_embeds.detach().float().cpu(),
                        "indexes": indexes.detach().cpu(),
                        "attention_mask": (
                            kwargs_.get("attention_mask", {})
                            .get("full_attention")
                            .detach()
                            .cpu()
                            if isinstance(kwargs_.get("attention_mask"), dict)
                            and kwargs_.get("attention_mask", {}).get("full_attention")
                            is not None
                            else None
                        ),
                    }
                    if u_hidden_states is not None:
                        payload["hidden_states"] = u_hidden_states
                    if u_layer_hidden_states:
                        payload["layer_hidden_states"] = u_layer_hidden_states
                    if u_sublayer_states:
                        payload["sublayer_states"] = u_sublayer_states
                    if output_past_key_values is not None:
                        layer0 = output_past_key_values.layers[0]
                        if getattr(layer0, "keys", None) is not None:
                            payload["layer0_keys"] = (
                                layer0.keys.detach().float().cpu()
                            )
                        if getattr(layer0, "values", None) is not None:
                            payload["layer0_values"] = (
                                layer0.values.detach().float().cpu()
                            )
                    torch.save(payload, path)
                    debug_records.append(
                        {
                            "kind": "commit_logits",
                            "call_index": commit_call_index,
                            "path": str(path),
                        }
                    )
                commit_call_index += 1
            return outputs

        engine.model.language_model.forward = wrapped_lm_forward
    input_images = official._load_input_images(args.image)
    width, height = official._resolve_image_size(
        input_images,
        args.width,
        args.height,
    )
    records = []
    original_build = engine.model._build_t2i_image_indexes

    def wrapped_build(token_h, token_w, text_len, device):
        records.append(
            {
                "token_h": _json_int(token_h),
                "token_w": _json_int(token_w),
                "text_len": _json_int(text_len),
            }
        )
        return original_build(token_h, token_w, text_len, device=device)

    engine.model._build_t2i_image_indexes = wrapped_build
    text, image_tensors = engine.model.interleave_gen(
        engine.tokenizer,
        args.prompt,
        images=list(input_images),
        image_size=(width, height),
        cfg_scale=args.cfg_scale,
        img_cfg_scale=args.img_cfg_scale,
        timestep_shift=args.timestep_shift,
        cfg_interval=tuple(args.cfg_interval),
        num_steps=args.num_steps,
        system_message=official.DEFAULT_SYSTEM_MESSAGE,
        think_mode=args.think_mode,
        seed=args.seed,
    )
    images = [official._to_pil(image_tensor) for image_tensor in image_tensors]
    out_dir = Path(args.output_dir)
    official._save_outputs(
        text,
        images,
        out_dir,
        args.stem,
        input_images=input_images,
        prompt=args.prompt,
    )
    text_parts = text.split("<image>")
    text_parts_token_ids = []
    token_chunk_index = 0
    for part in text_parts:
        if part and token_chunk_index < len(generated_token_chunks):
            text_parts_token_ids.append(generated_token_chunks[token_chunk_index])
            token_chunk_index += 1
        else:
            text_parts_token_ids.append([])
    for index, image_tensor in enumerate(image_tensors):
        raw_path = out_dir / f"{args.stem}_image_{index}.pt"
        torch.save(image_tensor.detach().float().cpu(), raw_path)
        print(f"[saved] {raw_path}")

    labels = ("condition", "text_uncondition", "img_uncondition")
    triples = []
    for index in range(0, len(records), 3):
        group = records[index : index + 3]
        for branch_index, record in enumerate(group):
            record["image_index"] = index // 3
            record["branch"] = (
                labels[branch_index]
                if branch_index < len(labels)
                else f"branch_{branch_index}"
            )
        triples.append(group)
    position_path = out_dir / f"{args.stem}_positions.json"
    position_path.write_text(
        json.dumps(
            {
                "records": records,
                "triples": triples,
                "raw_image_paths": [
                    str(out_dir / f"{args.stem}_image_{index}.pt")
                    for index in range(len(image_tensors))
                ],
                "text_parts_token_ids": text_parts_token_ids,
                "generated_token_chunks": generated_token_chunks,
                "debug_records": debug_records,
                "debug_dir": str(debug_dir) if debug_dir is not None else None,
            },
            indent=2,
        )
    )
    print(f"[saved] {position_path}")


if __name__ == "__main__":
    main()
""".lstrip(),
        encoding="utf-8",
    )


class _HarnessNativeSRTExecutor:
    finish_request_after_execute = True

    def __init__(
        self,
        *,
        runner,
        model_config,
        tree_cache,
        session_controller,
    ) -> None:
        self.runner = runner
        self.model_config = model_config
        self.tree_cache = tree_cache
        self.session_controller = session_controller
        self.token_bindings: list[UGSRTKVTokenBinding] = []
        self._active_batches_by_session: dict[str, object] = {}
        self._last_logits_by_session: dict[str, torch.Tensor] = {}
        self._last_u_debug_by_session: dict[str, dict[str, Any] | None] = {}
        self.debug_tensor_dump_dir: str | None = None
        self.debug_tensor_dump_max_commit_calls = 32
        self.debug_g_sublayer_layers: tuple[int, ...] = (0,)
        self._debug_commit_call_index = 0
        self._debug_u_decode_call_index = 0
        self.force_u_decode_token_ids: list[int] | None = None
        self._forced_u_decode_index = 0
        self.forced_u_decode_records: list[dict[str, int]] = []

    @property
    def temp_g_forward_count(self) -> int:
        return int(self._g_scheduler_executor.temp_g_forward_count)

    @property
    def temp_g_allocated_token_count(self) -> int:
        return int(self._g_scheduler_executor.temp_g_allocated_token_count)

    @property
    def _g_scheduler_executor(self):
        from sglang.srt.ug.srt_executor import UGSRTSchedulerExecutor

        executor = getattr(self, "_cached_g_scheduler_executor", None)
        if executor is not None:
            executor.debug_tensor_dump_dir = self.debug_tensor_dump_dir
            executor.debug_tensor_dump_max_g_calls = int(
                self.debug_tensor_dump_max_commit_calls
            )
            executor.debug_g_sublayer_layers = tuple(
                int(layer_id) for layer_id in self.debug_g_sublayer_layers
            )
            return executor
        fake_scheduler = SimpleNamespace(
            model_worker=SimpleNamespace(model_runner=self.runner),
            tree_cache=self.tree_cache,
            session_controller=self.session_controller,
            is_fully_idle=lambda: True,
        )
        executor = UGSRTSchedulerExecutor(fake_scheduler)
        executor.debug_tensor_dump_dir = self.debug_tensor_dump_dir
        executor.debug_tensor_dump_max_g_calls = int(
            self.debug_tensor_dump_max_commit_calls
        )
        executor.debug_g_sublayer_layers = tuple(
            int(layer_id) for layer_id in self.debug_g_sublayer_layers
        )
        self._cached_g_scheduler_executor = executor
        return executor

    def pad_input_ids(self, input_ids: list[int], mm_inputs) -> list[int]:
        pad_input_ids = getattr(
            getattr(self.runner, "model", None), "pad_input_ids", None
        )
        if callable(pad_input_ids) and mm_inputs is not None:
            return pad_input_ids(list(input_ids), mm_inputs)
        return list(input_ids)

    def create_u1_native_srt_pixel_flow_executor(self):
        return self._g_scheduler_executor.create_u1_native_srt_pixel_flow_executor()

    def get_latest_ug_session_token_binding(
        self,
        session_id: str,
    ) -> UGSRTKVTokenBinding | None:
        for binding in reversed(self.token_bindings):
            if binding.session_id == session_id:
                return binding
        return None

    def execute_ug_request(self, *, record, req, state) -> None:
        max_new_tokens = int(getattr(req.sampling_params, "max_new_tokens", 0) or 0)
        if state == UGSegmentState.U_DECODE and max_new_tokens > 0:
            self._execute_decode_request(record=record, req=req)
            return
        self._execute_extend_request(record=record, req=req)

    def _execute_extend_request(self, *, record, req) -> None:
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self._install_prefix(req, session_id=record.session_id)
        if req.extend_input_len <= 0:
            self._reuse_latest_binding(record.session_id, req)
            return
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=self.runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.runner.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(worker_batch, self.runner)
        self._enable_debug_u_capture(forward_batch)
        output, _ = self.runner.forward_extend(forward_batch)
        logits = output.next_token_logits.detach()
        self._last_u_debug_by_session[record.session_id] = (
            self._capture_debug_u_forward_payload()
        )
        self._dump_debug_commit_forward(
            req=req,
            forward_batch=forward_batch,
            logits=logits,
        )
        self._last_logits_by_session[record.session_id] = logits
        self._active_batches_by_session[record.session_id] = batch
        self._capture_binding(record.session_id, req)

    def _reuse_latest_binding(self, session_id: str, req) -> None:
        prev = self.get_latest_ug_session_token_binding(session_id)
        if prev is None:
            raise RuntimeError(
                "native U1 harness zero-extend commit requires existing KV binding"
            )
        req.output_ids = []
        req.kv_committed_len = int(prev.token_count)
        binding = UGSRTKVTokenBinding(
            session_id=session_id,
            request_id=req.rid,
            token_count=int(prev.token_count),
            token_indices=prev.token_indices.clone(),
            position_count=self._request_position_count(req) or prev.position_count,
        )
        self.token_bindings.append(binding)

    def _execute_decode_request(self, *, record, req) -> None:
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        batch = self._active_batches_by_session.get(record.session_id)
        logits = self._last_logits_by_session.get(record.session_id)
        if batch is None or logits is None:
            raise RuntimeError(
                "native U1 harness decode requires a prior SRT U prefill/commit"
            )
        active_req = batch.reqs[0]
        max_new_tokens = int(req.sampling_params.max_new_tokens)
        generated: list[int] = []
        token = self._select_u_decode_token(record.session_id, logits)
        for step in range(max_new_tokens):
            self._dump_debug_u_decode_logits(
                req=req,
                active_req=active_req,
                logits=logits,
                selected_token=token,
                debug_payload=self._last_u_debug_by_session.get(record.session_id),
            )
            generated.append(token)
            active_req.output_ids.append(token)
            batch.output_ids = torch.tensor(
                [token],
                dtype=torch.int64,
                device=self.runner.device,
            )
            batch.prepare_for_decode()
            worker_batch = batch.get_model_worker_batch()
            forward_batch = ForwardBatch.init_new(worker_batch, self.runner)
            self._enable_debug_u_capture(forward_batch)
            output = self.runner.forward_decode(forward_batch)
            logits = output.next_token_logits.detach()
            self._last_u_debug_by_session[record.session_id] = (
                self._capture_debug_u_forward_payload()
            )
            if step + 1 < max_new_tokens:
                token = self._select_u_decode_token(record.session_id, logits)
        self._last_logits_by_session[record.session_id] = logits
        req.output_ids = list(generated)
        req.req_pool_idx = getattr(active_req, "req_pool_idx", None)
        req.kv_committed_len = getattr(active_req, "kv_committed_len", 0)
        self._capture_binding(record.session_id, active_req, request_id=req.rid)

    def _select_u_decode_token(self, session_id: str, logits: torch.Tensor) -> int:
        del session_id
        greedy_token = _greedy_next_token(logits)
        forced_tokens = self.force_u_decode_token_ids
        if forced_tokens is None:
            return greedy_token
        index = int(self._forced_u_decode_index)
        if index >= len(forced_tokens):
            return greedy_token
        forced_token = int(forced_tokens[index])
        self._forced_u_decode_index = index + 1
        self.forced_u_decode_records.append(
            {
                "index": index,
                "greedy": int(greedy_token),
                "forced": int(forced_token),
            }
        )
        return forced_token

    def forced_u_decode_greedy_prefix_match_count(self) -> int | None:
        if self.force_u_decode_token_ids is None:
            return None
        count = 0
        for record in self.forced_u_decode_records:
            if int(record["greedy"]) != int(record["forced"]):
                break
            count += 1
        return count

    def forced_u_decode_first_greedy_mismatch(self) -> dict[str, int] | None:
        if self.force_u_decode_token_ids is None:
            return None
        for record in self.forced_u_decode_records:
            if int(record["greedy"]) != int(record["forced"]):
                return {
                    "index": int(record["index"]),
                    "greedy": int(record["greedy"]),
                    "forced": int(record["forced"]),
                }
        return None

    def _install_prefix(self, req, *, session_id: str) -> None:
        prev = self.get_latest_ug_session_token_binding(session_id)
        if prev is None:
            prefix_indices = torch.empty((0,), dtype=torch.int64)
        else:
            prefix_len = min(int(prev.token_count), len(req.origin_input_ids))
            prefix_indices = prev.token_indices[:prefix_len].to(dtype=torch.int64)
        req.prefix_indices = prefix_indices
        req.fill_ids = list(req.origin_input_ids)
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

    def _capture_binding(
        self,
        session_id: str,
        req,
        *,
        request_id: str | None = None,
    ) -> None:
        pool_idx = getattr(req, "req_pool_idx", None)
        token_count = int(getattr(req, "kv_committed_len", 0) or 0)
        if pool_idx is None or token_count <= 0:
            return
        token_indices = self.runner.req_to_token_pool.req_to_token[
            int(pool_idx), :token_count
        ].to(dtype=torch.int64)
        binding = UGSRTKVTokenBinding(
            session_id=session_id,
            request_id=request_id or req.rid,
            token_count=token_count,
            token_indices=token_indices.clone(),
            position_count=self._request_position_count(req),
        )
        self.token_bindings.append(binding)

    @staticmethod
    def _request_position_count(req) -> int | None:
        metadata = getattr(req, "ug_u_forward_metadata", {}) or {}
        adapter_metadata = metadata.get("adapter_metadata") or {}
        u1_metadata = adapter_metadata.get("u1") or {}
        g_position_start = u1_metadata.get("g_position_start")
        if g_position_start is None:
            model_state = adapter_metadata.get("ug_model_state") or {}
            u1_state = model_state.get("u1") or {}
            g_position_start = u1_state.get("g_position_start")
        return int(g_position_start) if g_position_start is not None else None

    def _enable_debug_u_capture(self, forward_batch) -> None:
        if self.debug_tensor_dump_dir:
            forward_batch.ug_debug_capture_u_layers = True
            forward_batch.ug_debug_capture_u_sublayers = (0,)

    def _capture_debug_u_forward_payload(self) -> dict[str, Any] | None:
        if not self.debug_tensor_dump_dir:
            return None
        model = getattr(self.runner, "model", None)
        language_model = getattr(model, "language_model", None)
        inner_model = getattr(language_model, "model", None)
        if inner_model is None:
            return None

        payload: dict[str, Any] = {}
        hidden_states = getattr(inner_model, "_last_u1_u_hidden_states", None)
        if hidden_states is not None:
            payload["hidden_states"] = (
                self._last_token_hidden_state(hidden_states).detach().float().cpu()
            )
        layer_hidden_states = getattr(
            inner_model,
            "_last_u1_u_layer_hidden_states",
            None,
        )
        if layer_hidden_states is not None:
            payload["layer_hidden_states"] = [
                {
                    "layer": int(layer_index),
                    "hidden_states": self._last_token_hidden_state(layer_hidden)
                    .detach()
                    .float()
                    .cpu(),
                }
                for layer_index, layer_hidden in layer_hidden_states
            ]
        sublayer_states = getattr(
            inner_model,
            "_last_u1_u_sublayer_states",
            None,
        )
        if sublayer_states is not None:
            payload["sublayer_states"] = [
                {
                    "layer": int(record["layer"]),
                    "name": str(record["name"]),
                    "hidden_states": self._last_token_hidden_state(
                        record["hidden_states"]
                    )
                    .detach()
                    .float()
                    .cpu(),
                }
                for record in sublayer_states
            ]
        return payload or None

    @staticmethod
    def _last_token_hidden_state(hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 3:
            return hidden_states[:, -1:, :].reshape(-1, hidden_states.shape[-1])
        if hidden_states.ndim == 2:
            return hidden_states[-1:, :]
        return hidden_states

    def _dump_debug_commit_forward(self, *, req, forward_batch, logits) -> None:
        dump_dir = getattr(self, "debug_tensor_dump_dir", None)
        if not dump_dir:
            return

        metadata = getattr(req, "ug_u_forward_metadata", {}) or {}
        adapter_metadata = metadata.get("adapter_metadata") or {}
        u1_metadata = adapter_metadata.get("u1") or {}
        if not (
            u1_metadata.get("native_generated_image_commit")
            or u1_metadata.get("generated_image_commit")
        ):
            return

        call_index = int(self._debug_commit_call_index)
        self._debug_commit_call_index = call_index + 1
        if call_index >= int(getattr(self, "debug_tensor_dump_max_commit_calls", 32)):
            return

        payload: dict[str, Any] = {
            "call_index": call_index,
            "logits": logits.detach().float().cpu(),
            "input_ids": forward_batch.input_ids.detach().cpu(),
            "positions": forward_batch.positions.detach().cpu(),
            "prefix_len": int((forward_batch.extend_prefix_lens_cpu or [0])[0]),
            "extend_len": int((forward_batch.extend_seq_lens_cpu or [0])[0]),
            "metadata": metadata,
        }
        if forward_batch.replace_positions is not None:
            payload["replace_positions"] = (
                forward_batch.replace_positions.detach().cpu()
            )
        if forward_batch.replace_embeds is not None:
            payload["replace_embeds"] = (
                forward_batch.replace_embeds.detach().float().cpu()
            )
        if forward_batch.cross_attention_custom_mask is not None:
            payload["attention_allowed_mask"] = (
                forward_batch.cross_attention_custom_mask.detach().cpu()
            )
        if getattr(forward_batch, "mm_input_embeds", None) is not None:
            payload["actual_input_embeds"] = (
                forward_batch.mm_input_embeds.detach().float().cpu()
            )
        debug_payload = self._capture_debug_u_forward_payload()
        if debug_payload is not None:
            payload.update(debug_payload)

        try:
            embed_layer = self.runner.model.get_input_embeddings()
            input_embeds = embed_layer(forward_batch.input_ids)
            if (
                forward_batch.replace_embeds is not None
                and forward_batch.replace_positions is not None
            ):
                input_embeds[forward_batch.replace_positions] = (
                    forward_batch.replace_embeds.to(input_embeds.dtype)
                )
            mm_inputs = getattr(req, "multimodal_inputs", None)
            if mm_inputs is not None:
                for item in getattr(mm_inputs, "mm_items", []) or []:
                    item_embeds = getattr(item, "precomputed_embeddings", None)
                    if item_embeds is None:
                        continue
                    item_embeds = item_embeds.to(
                        device=input_embeds.device,
                        dtype=input_embeds.dtype,
                    )
                    offset_cursor = 0
                    for start, end in getattr(item, "offsets", []) or []:
                        start = int(start)
                        end = int(end)
                        rel_start = start - int(
                            (forward_batch.extend_prefix_lens_cpu or [0])[0]
                        )
                        rel_end = end - int(
                            (forward_batch.extend_prefix_lens_cpu or [0])[0]
                        )
                        length = end - start + 1
                        if rel_start < 0 or rel_end >= int(input_embeds.shape[0]):
                            offset_cursor += length
                            continue
                        input_embeds[rel_start : rel_end + 1] = item_embeds[
                            offset_cursor : offset_cursor + length
                        ]
                        offset_cursor += length
            payload["input_embeds"] = input_embeds.detach().float().cpu()
        except Exception as exc:
            payload["input_embeds_error"] = repr(exc)

        path = Path(dump_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path / f"candidate_commit_logits_{call_index:04d}.pt")

    def _dump_debug_u_decode_logits(
        self,
        *,
        req,
        active_req,
        logits: torch.Tensor,
        selected_token: int,
        debug_payload: dict[str, Any] | None = None,
    ) -> None:
        dump_dir = getattr(self, "debug_tensor_dump_dir", None)
        if not dump_dir:
            return
        call_index = int(self._debug_u_decode_call_index)
        self._debug_u_decode_call_index = call_index + 1
        if call_index >= int(getattr(self, "debug_tensor_dump_max_commit_calls", 32)):
            return

        payload: dict[str, Any] = {
            "call_index": call_index,
            "request_id": getattr(req, "rid", None),
            "active_request_id": getattr(active_req, "rid", None),
            "logits": logits.detach().float().cpu(),
            "selected_token": int(selected_token),
            "active_output_len": len(getattr(active_req, "output_ids", []) or []),
            "active_origin_input_len": len(
                getattr(active_req, "origin_input_ids", []) or []
            ),
            "ug_decode_position_id": getattr(req, "ug_decode_position_id", None),
            "metadata": getattr(req, "ug_u_forward_metadata", {}) or {},
            **_debug_topk_payload(logits),
        }
        self._attach_debug_layer0_kv(payload, active_req)
        if debug_payload is not None:
            payload.update(debug_payload)
        path = Path(dump_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path / f"candidate_u_logits_{call_index:04d}.pt")

    def _attach_debug_layer0_kv(self, payload: dict[str, Any], req) -> None:
        pool_idx = getattr(req, "req_pool_idx", None)
        token_count = int(getattr(req, "kv_committed_len", 0) or 0)
        if pool_idx is None or token_count <= 0:
            return
        token_indices = self.runner.req_to_token_pool.req_to_token[
            int(pool_idx), :token_count
        ].to(dtype=torch.int64)
        payload["token_indices"] = token_indices.detach().cpu()
        key_buffer = self.runner.token_to_kv_pool.get_key_buffer(0)
        value_buffer = self.runner.token_to_kv_pool.get_value_buffer(0)
        payload["layer0_keys"] = key_buffer[token_indices].detach().float().cpu()
        payload["layer0_values"] = value_buffer[token_indices].detach().float().cpu()


def _make_native_tree_cache(runner):
    from sglang.bench_one_batch import TreeCacheNamespace

    return TreeCacheNamespace(
        page_size=getattr(runner.server_args, "page_size", 1),
        device=runner.device,
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
    )


def _run_native_u1_prefill_binding(
    *,
    runner,
    model,
    model_config,
    prepared,
    session_id: str,
    request_id: str,
) -> UGSRTKVTokenBinding:
    from sglang.bench_one_batch import TreeCacheNamespace
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    mm_inputs = getattr(prepared, "mm_inputs", None)
    pad_input_ids = getattr(model, "pad_input_ids", None)
    if mm_inputs is not None and callable(pad_input_ids):
        input_ids = pad_input_ids(list(prepared.input_ids), mm_inputs)
    else:
        input_ids = list(prepared.input_ids)
    req = Req(
        rid=request_id,
        origin_input_text=getattr(prepared, "input_text", "") or "",
        origin_input_ids=input_ids,
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=1),
    )
    req.fill_ids = list(input_ids)
    req.multimodal_inputs = mm_inputs
    req.logprob_start_len = -1
    req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
    tree_cache = TreeCacheNamespace(
        page_size=1,
        device=runner.device,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=[req],
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(worker_batch, runner)
    runner.forward_extend(forward_batch)

    pool_idx = getattr(req, "req_pool_idx", None)
    if pool_idx is None:
        raise RuntimeError("native U1 prefill did not allocate req_pool_idx")
    token_count = len(input_ids)
    token_indices = runner.req_to_token_pool.req_to_token[
        int(pool_idx), :token_count
    ].to(dtype=torch.int64)
    position_count = (
        (getattr(prepared, "adapter_metadata", {}) or {})
        .get("u1", {})
        .get("g_position_start")
    )
    return UGSRTKVTokenBinding(
        session_id=session_id,
        request_id=request_id,
        token_count=token_count,
        token_indices=token_indices.clone(),
        position_count=int(position_count) if position_count is not None else None,
    )


def _run_sglang_vlm_candidate(
    *,
    case: UGParityCase,
    official_py: Path,
    official_repo: Path,
    model_path: str,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    device: str,
    dtype: str,
    attn_backend: str,
    timeout: int,
    cuda_visible_devices: str | None,
    output_dir: Path,
) -> UGParityArtifact:
    from sglang.srt.session.session_controller import SessionController
    from sglang.srt.ug.adapter import UGModelRunnerAdapter
    from sglang.srt.ug.runtime import UGSessionRuntime

    backend = U1SubprocessVLMBackend(
        python=official_py,
        repo=official_repo,
        model_path=model_path,
        device=device,
        dtype=dtype,
        attn_backend=attn_backend,
        timeout=timeout,
        cuda_visible_devices=cuda_visible_devices,
        output_dir=output_dir,
    )
    adapter = U1UGModelAdapter(vlm_backend=backend)
    runtime = UGSessionRuntime(
        model_runner=UGModelRunnerAdapter(adapter),
        session_controller=SessionController(_HarnessTreeCache()),
        srt_image_tokenization="text_placeholder",
    )
    bridge = U1SRTBackedUGMiddleBridge(runtime)
    result = bridge.generate_vlm_text(
        messages=[
            UGInterleavedMessage(type="image", content=str(image_path)),
            UGInterleavedMessage(type="text", content=question),
        ],
        max_new_tokens=max_new_tokens,
    )
    try:
        debug_counters = runtime.get_debug_counters(result.session)
    finally:
        runtime.close_session(result.session)
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="sglang",
        text=result.text,
        image=summarize_ug_image(image_path),
        metadata={
            "candidate_backend": "u1_external_vlm_backend",
            "native_srt_model_runner": False,
        },
        debug_counters=debug_counters,
    )


def _run_official_vlm_reference(
    *,
    case: UGParityCase,
    official_py: Path,
    official_repo: Path,
    model_path: str,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    device: str,
    dtype: str,
    attn_backend: str,
    timeout: int,
    output_dir: Path,
    env,
) -> UGParityArtifact:
    script = official_repo / "examples/vqa/inference.py"
    raw_output = output_dir / f"{case.case_id}.official.txt"
    cmd = [
        str(official_py),
        str(script),
        "--model_path",
        model_path,
        "--image",
        str(image_path),
        "--question",
        question,
        "--output",
        str(raw_output),
        "--max_new_tokens",
        str(max_new_tokens),
        "--device",
        device,
        "--dtype",
        dtype,
        "--attn_backend",
        attn_backend,
    ]
    run_env = os.environ.copy()
    cuda_visible_devices = env.get("SGLANG_TEST_U1_CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    try:
        completed = subprocess.run(
            cmd,
            cwd=official_repo,
            env=run_env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return _official_vlm_artifact_from_completed(
            case=case,
            image_path=image_path,
            raw_output=raw_output,
            cmd=cmd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return UGParityArtifact(
            case_id=case.case_id,
            model=case.model,
            task=case.task,
            runner="official",
            image=summarize_ug_image(image_path),
            metadata={
                "command": cmd,
                "timeout_seconds": timeout,
                "stdout_tail": _tail(exc.stdout),
                "stderr_tail": _tail(exc.stderr),
            },
            error=f"official_vlm_timeout: timeout={timeout}s",
        )


def _run_official_t2i_reference(
    *,
    case: UGParityCase,
    official_py: Path,
    official_repo: Path,
    model_path: str,
    prompt: str,
    width: int,
    height: int,
    cfg_scale: float,
    cfg_norm: str,
    cfg_interval: tuple[float, float],
    timestep_shift: float,
    num_steps: int,
    seed: int,
    device: str,
    dtype: str,
    attn_backend: str,
    timeout: int,
    output_dir: Path,
    env,
) -> UGParityArtifact:
    script = official_repo / "examples/t2i/inference.py"
    image_output = output_dir / f"{case.case_id}.official.png"
    cmd = [
        str(official_py),
        str(script),
        "--model_path",
        model_path,
        "--prompt",
        prompt,
        "--output",
        str(image_output),
        "--width",
        str(width),
        "--height",
        str(height),
        "--cfg_scale",
        str(cfg_scale),
        "--cfg_norm",
        cfg_norm,
        "--timestep_shift",
        str(timestep_shift),
        "--cfg_interval",
        str(cfg_interval[0]),
        str(cfg_interval[1]),
        "--num_steps",
        str(num_steps),
        "--batch_size",
        "1",
        "--seed",
        str(seed),
        "--device",
        device,
        "--dtype",
        dtype,
        "--attn_backend",
        attn_backend,
    ]
    run_env = os.environ.copy()
    cuda_visible_devices = env.get("SGLANG_TEST_U1_CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    try:
        completed = subprocess.run(
            cmd,
            cwd=official_repo,
            env=run_env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return _official_image_artifact_from_completed(
            case=case,
            image_output=image_output,
            cmd=cmd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return UGParityArtifact(
            case_id=case.case_id,
            model=case.model,
            task=case.task,
            runner="official",
            metadata={
                "command": cmd,
                "timeout_seconds": timeout,
                "stdout_tail": _tail(exc.stdout),
                "stderr_tail": _tail(exc.stderr),
            },
            error=f"official_t2i_timeout: timeout={timeout}s",
        )


def _run_official_edit_reference(
    *,
    case: UGParityCase,
    official_py: Path,
    official_repo: Path,
    model_path: str,
    image_path: Path,
    prompt: str,
    width: int,
    height: int,
    cfg_scale: float,
    img_cfg_scale: float,
    cfg_norm: str,
    cfg_interval: tuple[float, float],
    timestep_shift: float,
    num_steps: int,
    seed: int,
    device: str,
    dtype: str,
    attn_backend: str,
    timeout: int,
    output_dir: Path,
    env,
) -> UGParityArtifact:
    script = official_repo / "examples/editing/inference.py"
    image_output = output_dir / f"{case.case_id}.official.png"
    cmd = [
        str(official_py),
        str(script),
        "--model_path",
        model_path,
        "--prompt",
        prompt,
        "--image",
        str(image_path),
        "--output",
        str(image_output),
        "--width",
        str(width),
        "--height",
        str(height),
        "--cfg_scale",
        str(cfg_scale),
        "--img_cfg_scale",
        str(img_cfg_scale),
        "--cfg_norm",
        cfg_norm,
        "--timestep_shift",
        str(timestep_shift),
        "--cfg_interval",
        str(cfg_interval[0]),
        str(cfg_interval[1]),
        "--num_steps",
        str(num_steps),
        "--batch_size",
        "1",
        "--seed",
        str(seed),
        "--device",
        device,
        "--dtype",
        dtype,
        "--attn_backend",
        attn_backend,
        "--no-do-resize",
    ]
    run_env = os.environ.copy()
    cuda_visible_devices = env.get("SGLANG_TEST_U1_CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    try:
        completed = subprocess.run(
            cmd,
            cwd=official_repo,
            env=run_env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return _official_image_artifact_from_completed(
            case=case,
            image_output=image_output,
            cmd=cmd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return UGParityArtifact(
            case_id=case.case_id,
            model=case.model,
            task=case.task,
            runner="official",
            metadata={
                "command": cmd,
                "timeout_seconds": timeout,
                "stdout_tail": _tail(exc.stdout),
                "stderr_tail": _tail(exc.stderr),
            },
            error=f"official_edit_timeout: timeout={timeout}s",
        )


def _run_official_interleave_reference(
    *,
    case: UGParityCase,
    official_py: Path,
    official_repo: Path,
    model_path: str,
    prompt: str,
    image_paths: list[Path],
    width: int,
    height: int,
    cfg_scale: float,
    img_cfg_scale: float,
    cfg_interval: tuple[float, float],
    timestep_shift: float,
    num_steps: int,
    seed: int,
    think_mode: bool,
    device: str,
    dtype: str,
    attn_backend: str,
    timeout: int,
    output_dir: Path,
    env,
    dump_raw_images: bool = False,
    dump_debug_tensors: bool = False,
) -> UGParityArtifact:
    script = official_repo / "examples/interleave/inference.py"
    work_dir = output_dir / "official-work"
    stem = case.case_id
    if dump_raw_images:
        script = output_dir / "official_interleave_raw_dump.py"
        _write_official_interleave_raw_dump_script(script)
    cmd = [
        str(official_py),
        str(script),
    ]
    if dump_raw_images:
        cmd.extend(["--official_repo", str(official_repo)])
    cmd.extend(
        [
            "--model_path",
            model_path,
            "--prompt",
            prompt,
            "--output_dir",
            str(work_dir),
            "--stem",
            stem,
            "--width",
            str(width),
            "--height",
            str(height),
            "--cfg_scale",
            str(cfg_scale),
            "--img_cfg_scale",
            str(img_cfg_scale),
            "--timestep_shift",
            str(timestep_shift),
            "--cfg_interval",
            str(cfg_interval[0]),
            str(cfg_interval[1]),
            "--num_steps",
            str(num_steps),
            "--seed",
            str(seed),
            "--device",
            device,
            "--dtype",
            dtype,
            "--attn_backend",
            attn_backend,
        ]
    )
    if think_mode:
        cmd.append("--think_mode")
    else:
        cmd.append("--no-think_mode")
    if dump_raw_images and dump_debug_tensors:
        cmd.append("--dump_debug_tensors")
        cmd.extend(
            [
                "--debug_max_g_calls",
                env.get("SGLANG_TEST_U1_INTERLEAVE_DEBUG_MAX_G_CALLS", "32"),
                "--debug_g_sublayers",
                env.get("SGLANG_TEST_U1_INTERLEAVE_DEBUG_G_SUBLAYERS", "0"),
            ]
        )
    for image_path in image_paths:
        cmd.extend(["--image", str(image_path)])

    run_env = os.environ.copy()
    cuda_visible_devices = env.get("SGLANG_TEST_U1_CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    try:
        completed = subprocess.run(
            cmd,
            cwd=official_repo,
            env=run_env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return _official_interleave_artifact_from_completed(
            case=case,
            work_dir=work_dir,
            stem=stem,
            input_image_paths=image_paths,
            cmd=cmd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return UGParityArtifact(
            case_id=case.case_id,
            model=case.model,
            task=case.task,
            runner="official",
            metadata={
                "command": cmd,
                "timeout_seconds": timeout,
                "stdout_tail": _tail(exc.stdout),
                "stderr_tail": _tail(exc.stderr),
            },
            error=f"official_interleave_timeout: timeout={timeout}s",
        )


def _official_image_artifact_from_completed(
    *,
    case: UGParityCase,
    image_output: Path,
    cmd: list[str],
    returncode: int,
    stdout: str | None,
    stderr: str | None,
) -> UGParityArtifact:
    image = summarize_ug_image(image_output) if image_output.exists() else None
    error = None
    if returncode != 0:
        error = f"official_{case.task}_failed: returncode={returncode}"
    elif image is None:
        error = f"official_{case.task}_failed: missing_output"
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="official",
        image=image,
        metadata={
            "command": cmd,
            "returncode": returncode,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "image_path": str(image_output),
        },
        error=error,
    )


def _official_interleave_artifact_from_completed(
    *,
    case: UGParityCase,
    work_dir: Path,
    stem: str,
    input_image_paths: list[Path],
    cmd: list[str],
    returncode: int,
    stdout: str | None,
    stderr: str | None,
) -> UGParityArtifact:
    text_path = work_dir / f"{stem}.txt"
    image_paths = sorted(work_dir.glob(f"{stem}_image_*.png"))
    raw_image_paths = sorted(work_dir.glob(f"{stem}_image_*.pt"))
    position_path = work_dir / f"{stem}_positions.json"
    position_trace = None
    if position_path.exists():
        position_trace = json.loads(position_path.read_text())
    output_text = _read_interleave_output_text(text_path)
    image_path = image_paths[0] if image_paths else None
    image = summarize_ug_image(image_path) if image_path is not None else None
    error = None
    if returncode != 0:
        error = f"official_interleave_failed: returncode={returncode}"
    elif output_text is None:
        error = "official_interleave_failed: missing_text_output"
    elif image is None:
        error = "official_interleave_failed: missing_image_output"

    segments: list[dict[str, object]] = []
    if output_text is not None:
        segments.append({"type": "text", "text": output_text})
    for path in image_paths:
        segments.append({"type": "image", "image_path": str(path)})
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="official",
        text=output_text,
        image=image,
        metadata={
            "command": cmd,
            "returncode": returncode,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "text_path": str(text_path),
            "image_path": str(image_path) if image_path is not None else None,
            "image_paths": [str(path) for path in image_paths],
            "raw_image_paths": [str(path) for path in raw_image_paths],
            "position_trace_path": (
                str(position_path) if position_path.exists() else None
            ),
            "position_trace": position_trace,
            "input_image_paths": [str(path) for path in input_image_paths],
            "segments": segments,
        },
        error=error,
    )


def _official_vlm_artifact_from_completed(
    *,
    case: UGParityCase,
    image_path: Path,
    raw_output: Path,
    cmd: list[str],
    returncode: int,
    stdout: str | None,
    stderr: str | None,
) -> UGParityArtifact:
    text = raw_output.read_text() if raw_output.exists() else None
    error = None
    if returncode != 0:
        error = f"official_vlm_failed: returncode={returncode}"
    elif text is None:
        error = "official_vlm_failed: missing_output"

    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="official",
        text=text,
        image=summarize_ug_image(image_path),
        metadata={
            "command": cmd,
            "returncode": returncode,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
        },
        error=error,
    )


def _candidate_unavailable_artifact(
    case: UGParityCase,
    *,
    image_path: Path,
) -> UGParityArtifact:
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="sglang",
        image=summarize_ug_image(image_path),
        metadata={
            "mode": "vlm_official_reference",
            "stop_signal": "native_u1_vlm_not_wired",
        },
        error="SGLang U1 VLM native path is not wired yet",
    )


def _candidate_t2i_unavailable_artifact(case: UGParityCase) -> UGParityArtifact:
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="sglang",
        metadata={
            "mode": "t2i_official_reference",
            "stop_signal": "native_u1_t2i_candidate_not_requested",
        },
        error="SGLang U1 T2I native candidate was not requested",
    )


def _candidate_image_unavailable_artifact(case: UGParityCase) -> UGParityArtifact:
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="sglang",
        metadata={
            "mode": f"{case.task}_official_reference",
            "stop_signal": f"native_u1_{case.task}_candidate_not_requested",
        },
        error=f"SGLang U1 {case.task} native candidate was not requested",
    )


def _candidate_interleave_unavailable_artifact(case: UGParityCase) -> UGParityArtifact:
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="sglang",
        metadata={
            "mode": "interleave_official_reference",
            "stop_signal": "native_u1_interleave_candidate_not_requested",
        },
        error="SGLang U1 interleave native candidate was not requested",
    )


def _compare_image_artifacts_with_tolerance(
    reference: UGParityArtifact,
    candidate: UGParityArtifact,
    *,
    mean_threshold: float,
    max_threshold: float,
    p99_threshold: float,
    psnr_threshold: float,
    ssim_threshold: float,
) -> UGParityReport:
    diffs: list[UGParityDiff] = []
    _append_diff_if_changed(diffs, "case_id", reference.case_id, candidate.case_id)
    _append_diff_if_changed(diffs, "model", reference.model, candidate.model)
    _append_diff_if_changed(diffs, "task", reference.task, candidate.task)
    _append_diff_if_changed(diffs, "error", reference.error, candidate.error)
    _append_diff_if_changed(diffs, "text", reference.text, candidate.text)

    metrics = {
        "mean_threshold": mean_threshold,
        "max_threshold": max_threshold,
        "max_threshold_policy": "record_only",
        "p99_threshold": p99_threshold,
        "psnr_threshold": psnr_threshold,
        "ssim_threshold": ssim_threshold,
    }
    ref_path = (reference.metadata or {}).get("image_path")
    cand_path = (candidate.metadata or {}).get("image_path")
    if reference.image is None or candidate.image is None:
        _append_diff_if_changed(diffs, "image", reference.image, candidate.image)
    elif reference.image.size != candidate.image.size:
        diffs.append(
            UGParityDiff(
                field="image.size",
                reference=reference.image.size,
                candidate=candidate.image.size,
                reason="value mismatch",
            )
        )
    elif (
        ref_path and cand_path and Path(ref_path).exists() and Path(cand_path).exists()
    ):
        image_metrics = _image_precision_diagnostics(Path(ref_path), Path(cand_path))
        metrics.update(image_metrics)
        mean_diff = image_metrics["image_mean_abs_diff"]
        p99_diff = image_metrics["image_abs_diff_p99"]
        psnr = image_metrics["image_psnr_db"]
        ssim = image_metrics["image_ssim_luma_global"]
        if mean_diff > mean_threshold or p99_diff > p99_threshold:
            diffs.append(
                UGParityDiff(
                    field="image.pixel_abs_diff",
                    reference={"mean<=": mean_threshold, "p99<=": p99_threshold},
                    candidate={"mean": mean_diff, "p99": p99_diff},
                    reason="image diff exceeds tolerance",
                )
            )
        if psnr < psnr_threshold:
            diffs.append(
                UGParityDiff(
                    field="image.psnr_db",
                    reference={">=": psnr_threshold},
                    candidate=psnr,
                    reason="image PSNR below tolerance",
                )
            )
        if ssim < ssim_threshold:
            diffs.append(
                UGParityDiff(
                    field="image.ssim_luma_global",
                    reference={">=": ssim_threshold},
                    candidate=ssim,
                    reason="image SSIM below tolerance",
                )
            )
    else:
        _append_diff_if_changed(
            diffs,
            "image.sha256",
            reference.image.sha256,
            candidate.image.sha256,
        )

    return UGParityReport(
        case_id=reference.case_id,
        model=reference.model,
        passed=not diffs,
        diffs=tuple(diffs),
        metadata={
            "reference_runner": reference.runner,
            "candidate_runner": candidate.runner,
            **metrics,
        },
    )


def _compare_interleave_artifacts_with_tolerance(
    reference: UGParityArtifact,
    candidate: UGParityArtifact,
    *,
    mean_threshold: float,
    max_threshold: float,
    p99_threshold: float,
    psnr_threshold: float,
    ssim_threshold: float,
    require_text: bool,
    require_text_exact: bool,
    min_candidate_images: int = 1,
    require_text_after_each_image: bool = True,
    teacher_force_text: bool = False,
) -> UGParityReport:
    report = _compare_image_artifacts_with_tolerance(
        reference,
        candidate,
        mean_threshold=mean_threshold,
        max_threshold=max_threshold,
        p99_threshold=p99_threshold,
        psnr_threshold=psnr_threshold,
        ssim_threshold=ssim_threshold,
    )
    diffs = list(report.diffs)
    if not require_text_exact:
        diffs = [diff for diff in diffs if diff.field != "text"]
    metrics = dict(report.metadata)
    metrics["require_text"] = require_text
    metrics["require_text_exact"] = require_text_exact
    metrics["min_candidate_images"] = int(min_candidate_images)
    metrics["require_text_after_each_image"] = bool(require_text_after_each_image)
    metrics["teacher_force_text_gate"] = bool(teacher_force_text)
    metrics["reference_segment_types"] = _artifact_segment_types(reference)
    metrics["candidate_segment_types"] = _artifact_segment_types(candidate)
    metrics["candidate_interleave_event_types"] = _interleave_event_types(candidate)
    metrics.update(_interleave_text_token_alignment_metrics(reference, candidate))
    interleave_image_metrics, interleave_image_diffs = (
        _compare_interleave_image_sequence_with_tolerance(
            reference,
            candidate,
            mean_threshold=mean_threshold,
            p99_threshold=p99_threshold,
            psnr_threshold=psnr_threshold,
            ssim_threshold=ssim_threshold,
            min_candidate_images=min_candidate_images,
        )
    )
    metrics.update(interleave_image_metrics)
    diffs.extend(interleave_image_diffs)

    if require_text and not (reference.text or "").strip():
        diffs.append(
            UGParityDiff(
                field="reference.text",
                reference="non-empty text",
                candidate=reference.text,
                reason="missing official interleave text",
            )
        )
    if require_text and not (candidate.text or "").strip():
        diffs.append(
            UGParityDiff(
                field="candidate.text",
                reference="non-empty text",
                candidate=candidate.text,
                reason="missing SGLang interleave post-image text",
            )
        )
    if require_text_exact:
        _append_diff_if_changed(diffs, "text", reference.text, candidate.text)

    candidate_metadata = candidate.metadata or {}
    if candidate.error is None:
        _append_interleave_sequence_diffs(
            diffs,
            candidate=candidate,
            min_candidate_images=min_candidate_images,
            require_text_after_each_image=require_text_after_each_image,
        )
        if not candidate_metadata.get("same_session_id"):
            diffs.append(
                UGParityDiff(
                    field="candidate.same_session_id",
                    reference=True,
                    candidate=candidate_metadata.get("same_session_id"),
                    reason="interleave candidate did not preserve one UG session",
                )
            )
        if candidate_metadata.get("prefill_count") != 1:
            diffs.append(
                UGParityDiff(
                    field="candidate.prefill_count",
                    reference=1,
                    candidate=candidate_metadata.get("prefill_count"),
                    reason="interleave candidate should reuse one U prefill",
                )
            )
        if (
            int(candidate_metadata.get("append_image_count") or 0)
            < min_candidate_images
        ):
            diffs.append(
                UGParityDiff(
                    field="candidate.append_image_count",
                    reference=f">={min_candidate_images}",
                    candidate=candidate_metadata.get("append_image_count"),
                    reason="not every generated image was committed back through U",
                )
            )
        if teacher_force_text:
            if (
                int(candidate_metadata.get("teacher_forced_reference_text_count") or 0)
                < 1
            ):
                diffs.append(
                    UGParityDiff(
                        field="candidate.teacher_forced_reference_text_count",
                        reference=">=1",
                        candidate=candidate_metadata.get(
                            "teacher_forced_reference_text_count"
                        ),
                        reason=(
                            "teacher_force_text gate did not append official text "
                            "into the SRT session"
                        ),
                    )
                )
            if (
                int(
                    candidate_metadata.get("teacher_forced_reference_marker_count") or 0
                )
                < 1
            ):
                diffs.append(
                    UGParityDiff(
                        field="candidate.teacher_forced_reference_marker_count",
                        reference=">=1",
                        candidate=candidate_metadata.get(
                            "teacher_forced_reference_marker_count"
                        ),
                        reason=(
                            "teacher_force_text gate did not append an official "
                            "image marker into the SRT session"
                        ),
                    )
                )
        elif int(candidate_metadata.get("srt_u_decode_request_count") or 0) < 1:
            diffs.append(
                UGParityDiff(
                    field="candidate.srt_u_decode_request_count",
                    reference=">=1",
                    candidate=candidate_metadata.get("srt_u_decode_request_count"),
                    reason="post-image U decode did not run through SRT",
                )
            )

    return UGParityReport(
        case_id=reference.case_id,
        model=reference.model,
        passed=not diffs,
        diffs=tuple(diffs),
        metadata=metrics,
    )


def _append_diff_if_changed(
    diffs: list[UGParityDiff],
    field: str,
    reference,
    candidate,
) -> None:
    if reference != candidate:
        diffs.append(
            UGParityDiff(
                field=field,
                reference=reference,
                candidate=candidate,
                reason="value mismatch",
            )
        )


def _image_precision_diagnostics(
    reference_path: Path, candidate_path: Path
) -> dict[str, object]:
    with Image.open(reference_path) as ref_img, Image.open(candidate_path) as cand_img:
        ref = np.asarray(ref_img.convert("RGB"), dtype=np.int16)
        cand = np.asarray(cand_img.convert("RGB"), dtype=np.int16)
    signed = ref.astype(np.float32) - cand.astype(np.float32)
    diff = np.abs(signed)
    mse = float(np.square(signed).mean())
    rmse = float(np.sqrt(mse))
    psnr = 100.0 if mse == 0.0 else float(20.0 * np.log10(255.0 / rmse))
    percentiles = np.percentile(diff, [50, 90, 95, 99])

    absdiff = np.clip(diff, 0, 255).astype(np.uint8)
    heatmap = _image_diff_heatmap(diff)
    artifact_prefix = candidate_path.with_suffix("")
    absdiff_path = artifact_prefix.with_name(f"{artifact_prefix.name}.absdiff.png")
    heatmap_path = artifact_prefix.with_name(f"{artifact_prefix.name}.heatmap.png")
    Image.fromarray(absdiff).save(absdiff_path)
    Image.fromarray(heatmap).save(heatmap_path)

    return {
        "image_mean_abs_diff": float(diff.mean()),
        "image_max_abs_diff": float(diff.max()),
        "image_abs_diff_p50": float(percentiles[0]),
        "image_abs_diff_p90": float(percentiles[1]),
        "image_abs_diff_p95": float(percentiles[2]),
        "image_abs_diff_p99": float(percentiles[3]),
        "image_abs_diff_nonzero_ratio": float(np.count_nonzero(diff) / diff.size),
        "image_abs_diff_channel_mean": [
            float(value) for value in diff.reshape(-1, 3).mean(axis=0)
        ],
        "image_rmse": rmse,
        "image_psnr_db": psnr,
        "image_ssim_luma_global": _global_luma_ssim(ref, cand),
        "image_absdiff_path": str(absdiff_path),
        "image_heatmap_path": str(heatmap_path),
    }


def _image_diff_heatmap(diff: np.ndarray) -> np.ndarray:
    per_pixel = diff.max(axis=2)
    scale = max(float(per_pixel.max()), 1.0)
    intensity = np.clip(per_pixel / scale * 255.0, 0, 255).astype(np.uint8)
    zeros = np.zeros_like(intensity)
    return np.stack([intensity, zeros, 255 - intensity], axis=2)


def _global_luma_ssim(ref_rgb: np.ndarray, cand_rgb: np.ndarray) -> float:
    ref = ref_rgb.astype(np.float32)
    cand = cand_rgb.astype(np.float32)
    ref_y = 0.299 * ref[..., 0] + 0.587 * ref[..., 1] + 0.114 * ref[..., 2]
    cand_y = 0.299 * cand[..., 0] + 0.587 * cand[..., 1] + 0.114 * cand[..., 2]
    ref_mean = float(ref_y.mean())
    cand_mean = float(cand_y.mean())
    ref_var = float(np.square(ref_y - ref_mean).mean())
    cand_var = float(np.square(cand_y - cand_mean).mean())
    covariance = float(((ref_y - ref_mean) * (cand_y - cand_mean)).mean())
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * ref_mean * cand_mean + c1) * (2.0 * covariance + c2)
    denominator = (ref_mean**2 + cand_mean**2 + c1) * (ref_var + cand_var + c2)
    return float(numerator / denominator)


def _artifact_segment_types(artifact: UGParityArtifact) -> list[str]:
    segments = (artifact.metadata or {}).get("segments") or []
    return [
        str(segment.get("type")) for segment in segments if isinstance(segment, dict)
    ]


def _interleave_text_token_alignment_metrics(
    reference: UGParityArtifact,
    candidate: UGParityArtifact,
) -> dict[str, object]:
    reference_tokens = _artifact_interleave_text_token_ids(reference)
    candidate_tokens = _artifact_interleave_text_token_ids(candidate)
    if reference_tokens is None and candidate_tokens is None:
        return {}
    if reference_tokens is None or candidate_tokens is None:
        return {
            "u_text_token_reference_available": reference_tokens is not None,
            "u_text_token_candidate_available": candidate_tokens is not None,
        }
    reference_tokens = reference_tokens or []
    candidate_tokens = candidate_tokens or []
    prefix_count = 0
    for reference_token, candidate_token in zip(reference_tokens, candidate_tokens):
        if reference_token != candidate_token:
            break
        prefix_count += 1
    denominator = max(len(reference_tokens), 1)
    metrics: dict[str, object] = {
        "u_text_token_reference_count": len(reference_tokens),
        "u_text_token_candidate_count": len(candidate_tokens),
        "u_text_token_prefix_match_count": prefix_count,
        "u_text_token_prefix_match_ratio": float(prefix_count / denominator),
    }
    if prefix_count < max(len(reference_tokens), len(candidate_tokens)):
        metrics["u_text_token_first_mismatch"] = {
            "index": prefix_count,
            "reference": (
                reference_tokens[prefix_count]
                if prefix_count < len(reference_tokens)
                else None
            ),
            "candidate": (
                candidate_tokens[prefix_count]
                if prefix_count < len(candidate_tokens)
                else None
            ),
        }
    return metrics


def _artifact_interleave_text_token_ids(
    artifact: UGParityArtifact,
) -> list[int] | None:
    metadata = artifact.metadata or {}
    position_trace = metadata.get("position_trace")
    if isinstance(position_trace, dict):
        token_parts = position_trace.get("text_parts_token_ids")
        if isinstance(token_parts, list):
            flattened: list[int] = []
            for part in token_parts:
                if isinstance(part, list):
                    flattened.extend(int(token_id) for token_id in part)
            return flattened

    segments = metadata.get("segments") or []
    if not isinstance(segments, list):
        return None
    flattened = []
    saw_tokens = False
    for segment in segments:
        if not isinstance(segment, dict) or segment.get("type") != "text":
            continue
        segment_metadata = segment.get("metadata") or {}
        if not isinstance(segment_metadata, dict):
            continue
        token_ids = segment_metadata.get("token_ids")
        if not isinstance(token_ids, list):
            continue
        saw_tokens = True
        flattened.extend(int(token_id) for token_id in token_ids)
    return flattened if saw_tokens else None


def _artifact_image_paths(artifact: UGParityArtifact) -> list[Path]:
    metadata = artifact.metadata or {}
    raw_paths = metadata.get("image_paths")
    if isinstance(raw_paths, list):
        paths = [Path(str(path)) for path in raw_paths if path]
    else:
        paths = []
    if not paths and metadata.get("image_path"):
        paths = [Path(str(metadata["image_path"]))]
    if paths:
        return paths
    segments = metadata.get("segments") or []
    return [
        Path(str(segment["image_path"]))
        for segment in segments
        if isinstance(segment, dict) and segment.get("image_path")
    ]


def _artifact_raw_image_paths(artifact: UGParityArtifact) -> list[Path]:
    metadata = artifact.metadata or {}
    raw_paths = metadata.get("raw_image_paths")
    if not isinstance(raw_paths, list):
        return []
    return [Path(str(path)) for path in raw_paths if path]


def _interleave_teacher_force_commit_paths(
    *,
    reference: UGParityArtifact,
    mode: str | None,
) -> list[Path] | None:
    normalized = str(mode or "").lower()
    if normalized in ("", "0", "false", "none"):
        return None
    if normalized in ("1", "true", "png", "image", "pil"):
        paths = _artifact_image_paths(reference)
    elif normalized in ("raw", "tensor", "pt"):
        paths = _artifact_raw_image_paths(reference)
    else:
        raise ValueError(
            "Unsupported SGLANG_TEST_U1_INTERLEAVE_TEACHER_FORCE_REFERENCE_IMAGES "
            f"value: {mode!r}"
        )
    if not paths:
        raise RuntimeError(
            "U1 interleave teacher-force requested but reference commit paths "
            "are missing"
        )
    return paths


def _interleave_teacher_force_text_parts(reference: UGParityArtifact) -> list[str]:
    text = reference.text or ""
    if "<image>" not in text:
        raise RuntimeError(
            "U1 interleave reference text teacher-force requested but the "
            "official output has no <image> markers"
        )
    return text.split("<image>")


def _interleave_teacher_force_token_parts(
    reference: UGParityArtifact,
) -> list[list[int]] | None:
    position_trace = reference.metadata.get("position_trace")
    if not isinstance(position_trace, dict):
        return None
    token_parts = position_trace.get("text_parts_token_ids")
    if not isinstance(token_parts, list):
        return None
    normalized: list[list[int]] = []
    for part in token_parts:
        if not isinstance(part, list):
            return None
        normalized.append([int(token_id) for token_id in part])
    return normalized


def _interleave_reference_u_decode_token_ids(
    *,
    reference: UGParityArtifact,
    tokenizer_model_path: str,
) -> list[int]:
    token_parts = _interleave_teacher_force_token_parts(reference)
    if token_parts is None:
        raise RuntimeError(
            "U1 reference U decode forcing requires official token trace; set "
            "SGLANG_TEST_U1_INTERLEAVE_DUMP_TOKEN_TRACE=1"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
    img_start_id = int(tokenizer.convert_tokens_to_ids("<img>"))
    token_ids: list[int] = []
    for index, part in enumerate(token_parts):
        token_ids.extend(int(token_id) for token_id in part)
        if index < len(token_parts) - 1:
            token_ids.append(img_start_id)
    return token_ids


def _append_u1_reference_text(
    *,
    runtime: Any,
    bridge: Any,
    contexts: UGContextBundle,
    tokenizer: Any,
    text: str,
    token_ids: list[int] | None = None,
) -> None:
    if token_ids is None:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    else:
        token_ids = [int(token_id) for token_id in token_ids]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
    if not token_ids:
        return
    start = _u1_reference_logical_position(runtime, contexts)
    end = start + len(token_ids)
    prepared = UGSRTPreparedInput(
        input_ids=[int(token_id) for token_id in token_ids],
        input_text=text,
        messages=[UGInterleavedMessage(type="text", content=text)],
        position_ids=list(range(start, end)),
        adapter_metadata={
            "u1": {
                "segment_type": "interleave_reference_text",
                "source": "official_reference_text_teacher_force",
                "g_position_start": end,
            },
            "ug_model_state_updates": {
                "u1": {
                    "last_segment_type": "interleave",
                    "last_source": "official_reference_text_teacher_force",
                    "native_interleave_prompt": True,
                    "open_image_marker": False,
                    "interleave_pending_image_marker": False,
                    "g_position_start": end,
                }
            },
        },
    )
    _append_u1_reference_prepared_input(
        runtime=runtime,
        contexts=contexts,
        prepared=prepared,
    )
    bridge._attach_u1_context_metadata(contexts)


def _append_u1_reference_image_marker(
    *,
    runtime: Any,
    bridge: Any,
    contexts: UGContextBundle,
    tokenizer: Any,
) -> None:
    marker_position = _u1_reference_logical_position(runtime, contexts)
    img_start_id = tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN)
    prepared = UGSRTPreparedInput(
        input_ids=[int(img_start_id)],
        input_text=U1_IMG_START_TOKEN,
        messages=[],
        position_ids=[[marker_position, 0, 0]],
        adapter_metadata={
            "u1": {
                "segment_type": "interleave_reference_image_marker",
                "source": "official_reference_text_teacher_force",
                "g_position_start": marker_position + 1,
            },
            "ug_model_state_updates": {
                "u1": {
                    "last_segment_type": "interleave",
                    "last_source": "official_reference_image_marker_teacher_force",
                    "native_interleave_prompt": True,
                    "open_image_marker": True,
                    "interleave_pending_image_marker": False,
                    "g_position_start": marker_position + 1,
                }
            },
        },
    )
    _append_u1_reference_prepared_input(
        runtime=runtime,
        contexts=contexts,
        prepared=prepared,
    )
    adapter = getattr(runtime.model_runner, "adapter", None)
    append_marker = getattr(adapter, "_append_interleave_text_uncondition_marker", None)
    if callable(append_marker):
        append_marker(runtime=runtime, session=contexts.full.session)
    bridge._attach_u1_context_metadata(contexts)
    runtime._record_for(contexts.full.session).state = UGSegmentState.G_DENOISE


def _append_u1_reference_prepared_input(
    *,
    runtime: Any,
    contexts: UGContextBundle,
    prepared: UGSRTPreparedInput,
) -> None:
    record = runtime._record_for(contexts.full.session)
    record.state = UGSegmentState.U_DECODE
    next_context_version = record.context_version + 1
    request_id = f"{record.session_id}:reference_text:{next_context_version}"
    record.anchor_request_id = request_id
    req, recv_req = runtime._create_srt_session_req(
        record,
        request_id=request_id,
        input_ids=prepared.input_ids,
        input_text=prepared.input_text,
        mm_inputs=prepared.mm_inputs,
        max_new_tokens=0,
    )
    adapter_metadata = runtime._apply_prepared_srt_input(
        req,
        record=record,
        recv_req=recv_req,
        prepared=prepared,
        is_final_segment=True,
    )
    runtime._record_srt_req(record, req, request_id=request_id)
    runtime._attach_srt_u_forward_metadata(
        record,
        req,
        state=UGSegmentState.U_DECODE,
        input_text=prepared.input_text,
        messages=prepared.messages,
        adapter_metadata=adapter_metadata,
    )
    runtime._execute_srt_req(record, req, state=UGSegmentState.U_DECODE)
    runtime._notify_srt_u_forward(
        record,
        req,
        state=UGSegmentState.U_DECODE,
        input_text=prepared.input_text,
        messages=prepared.messages,
    )
    record.context_length = len(req.origin_input_ids)
    record.context_version = next_context_version
    record.state = UGSegmentState.U_DECODE
    contexts.full.session = record.handle()
    contexts.full.request_id = record.anchor_request_id
    contexts.full.token_count = record.context_length


def _u1_reference_logical_position(runtime: Any, contexts: UGContextBundle) -> int:
    counters = runtime.get_debug_counters(contexts.full.session)
    u1_state = (counters.get("ug_model_state") or {}).get("u1") or {}
    g_position_start = u1_state.get("g_position_start")
    if g_position_start is not None:
        return int(g_position_start)
    return int(contexts.full.token_count)


def _compare_interleave_image_sequence_with_tolerance(
    reference: UGParityArtifact,
    candidate: UGParityArtifact,
    *,
    mean_threshold: float,
    p99_threshold: float,
    psnr_threshold: float,
    ssim_threshold: float,
    min_candidate_images: int = 1,
) -> tuple[dict[str, object], list[UGParityDiff]]:
    reference_paths = _artifact_image_paths(reference)
    candidate_paths = _artifact_image_paths(candidate)
    pair_count = min(len(reference_paths), len(candidate_paths))
    diffs: list[UGParityDiff] = []
    image_metrics: list[dict[str, object]] = []
    if len(reference_paths) < min_candidate_images:
        diffs.append(
            UGParityDiff(
                field="reference.image_count",
                reference=f">={min_candidate_images}",
                candidate=len(reference_paths),
                reason="official interleave reference produced too few images",
            )
        )
    if len(candidate_paths) < min_candidate_images:
        diffs.append(
            UGParityDiff(
                field="candidate.image_count",
                reference=f">={min_candidate_images}",
                candidate=len(candidate_paths),
                reason="SGLang interleave candidate produced too few images",
            )
        )
    for index in range(pair_count):
        ref_path = reference_paths[index]
        cand_path = candidate_paths[index]
        if not ref_path.exists() or not cand_path.exists():
            diffs.append(
                UGParityDiff(
                    field=f"image[{index}].path",
                    reference=str(ref_path),
                    candidate=str(cand_path),
                    reason="missing interleave image artifact",
                )
            )
            continue
        metrics = _image_precision_diagnostics(ref_path, cand_path)
        metrics = {"index": index, **metrics}
        image_metrics.append(metrics)
        mean_diff = metrics["image_mean_abs_diff"]
        p99_diff = metrics["image_abs_diff_p99"]
        psnr = metrics["image_psnr_db"]
        ssim = metrics["image_ssim_luma_global"]
        if mean_diff > mean_threshold or p99_diff > p99_threshold:
            diffs.append(
                UGParityDiff(
                    field=f"image[{index}].pixel_abs_diff",
                    reference={"mean<=": mean_threshold, "p99<=": p99_threshold},
                    candidate={"mean": mean_diff, "p99": p99_diff},
                    reason="interleave image diff exceeds tolerance",
                )
            )
        if psnr < psnr_threshold:
            diffs.append(
                UGParityDiff(
                    field=f"image[{index}].psnr_db",
                    reference={">=": psnr_threshold},
                    candidate=psnr,
                    reason="interleave image PSNR below tolerance",
                )
            )
        if ssim < ssim_threshold:
            diffs.append(
                UGParityDiff(
                    field=f"image[{index}].ssim_luma_global",
                    reference={">=": ssim_threshold},
                    candidate=ssim,
                    reason="interleave image SSIM below tolerance",
                )
            )
    return (
        {
            "reference_image_count": len(reference_paths),
            "candidate_image_count": len(candidate_paths),
            "compared_image_count": pair_count,
            "interleave_image_metrics": image_metrics,
        },
        diffs,
    )


def _append_interleave_sequence_diffs(
    diffs: list[UGParityDiff],
    *,
    candidate: UGParityArtifact,
    min_candidate_images: int,
    require_text_after_each_image: bool,
) -> None:
    if min_candidate_images <= 0:
        return
    event_types = _interleave_event_types(candidate)
    required: list[str] = []
    for _ in range(min_candidate_images):
        required.extend(("text", "image"))
    if require_text_after_each_image:
        required.append("text")
    if event_types[: len(required)] == required:
        return
    diffs.append(
        UGParityDiff(
            field="candidate.interleave_event_types",
            reference=required,
            candidate=event_types,
            reason=(
                "interleave candidate did not alternate U text and G image "
                "segments for the requested number of images"
            ),
        )
    )


def _interleave_event_types(artifact: UGParityArtifact) -> list[str]:
    segments = (artifact.metadata or {}).get("segments") or []
    if not isinstance(segments, list):
        return []
    event_types: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_type = segment.get("type")
        if segment_type == "text":
            if not str(segment.get("text") or "").strip():
                continue
            event_type = "text"
        elif segment_type == "image":
            event_type = "image"
        else:
            continue
        if not event_types or event_types[-1] != event_type:
            event_types.append(event_type)
    return event_types


def _summarize_output_segments(
    segments: list[dict[str, object]],
    image_paths: list[Path],
) -> list[dict[str, object]]:
    summarized: list[dict[str, object]] = []
    image_index = 0
    for segment in segments:
        segment_type = segment.get("type")
        if segment_type == "text":
            summarized.append(
                {
                    "type": "text",
                    "text": str(segment.get("text") or ""),
                    "metadata": dict(segment.get("metadata") or {}),
                }
            )
        elif segment_type == "image":
            path = image_paths[image_index] if image_index < len(image_paths) else None
            summarized.append(
                {
                    "type": "image",
                    "image_path": str(path) if path is not None else None,
                    "metadata": dict(segment.get("metadata") or {}),
                }
            )
            image_index += 1
    return summarized


def _read_interleave_output_text(text_path: Path) -> str | None:
    if not text_path.exists():
        return None
    text = text_path.read_text(encoding="utf-8")
    marker = "# OUTPUT\n"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def _parse_path_list(value: str | None) -> list[Path]:
    if not value:
        return []
    return [
        Path(part.strip())
        for part in value.replace(os.pathsep, ",").split(",")
        if part.strip()
    ]


def _parse_int_list(value: str | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if not value:
        return default
    parts = [
        part.strip()
        for part in value.replace(os.pathsep, ",").replace(",", " ").split()
        if part.strip()
    ]
    if not parts:
        return default
    return tuple(int(part) for part in parts)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float_pair(
    value: str | None, *, default: tuple[float, float]
) -> tuple[float, float]:
    if not value:
        return default
    parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two floats, got {value!r}")
    return (float(parts[0]), float(parts[1]))


def _tail(text: str | bytes | None, *, limit: int = 2000) -> str | None:
    if text is None:
        return None
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[-limit:]


def _write_dry_run_bundle(output_dir: Path) -> Path:
    case = UGParityCase(
        case_id="u1-dry-run",
        model="sensenova-u1",
        task="vlm",
        prompt="describe this image",
        seed=1,
        sampling_params={"max_new_tokens": 4},
        dump_points=("text", "u_logits"),
    )
    reference = _dry_run_artifact(case, runner="official")
    candidate = _dry_run_artifact(case, runner="sglang")
    report = compare_ug_parity_artifacts(reference, candidate)
    return write_ug_parity_bundle(
        output_dir=output_dir,
        case=case,
        reference=reference,
        candidate=candidate,
        report=report,
    )


def _dry_run_artifact(case: UGParityCase, *, runner: str) -> UGParityArtifact:
    image = Image.fromarray(np.full((4, 4, 3), 7, dtype=np.uint8), "RGB")
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner=runner,
        text="dry run",
        image=summarize_ug_image(image),
        tensors={"u_logits": UGTensorSummary.from_tensor(torch.ones(2, 2))},
        metadata={"dry_run": True},
    )


def _case_from_artifact(artifact: UGParityArtifact) -> UGParityCase:
    return UGParityCase(
        case_id=artifact.case_id,
        model=artifact.model,
        task=artifact.task,
        metadata={"source": "artifact_env"},
    )


class _HarnessTreeCache:
    def __init__(self) -> None:
        self.released_sessions: list[str] = []

    def release_session(self, session_id: str) -> None:
        self.released_sessions.append(session_id)


if __name__ == "__main__":
    unittest.main()
