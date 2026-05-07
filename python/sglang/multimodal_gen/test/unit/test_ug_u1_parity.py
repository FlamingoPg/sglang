# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from sglang.srt.ug.parity import (
    UGParityArtifact,
    UGParityCase,
    UGParityRunner,
    UGTensorSummary,
    compare_ug_parity_artifacts,
    summarize_ug_image,
    write_ug_parity_bundle,
)


class TestU1OfficialParityHarness(unittest.TestCase):
    def test_u1_parity_case_roundtrip(self):
        case = UGParityCase(
            case_id="u1-vlm-smoke",
            model="sensenova-u1",
            task="vlm",
            prompt="Describe this image.",
            image_path="/tmp/u1-input.png",
            seed=123,
            sampling_params={"max_new_tokens": 16},
            dump_points=("text", "u_logits"),
        )

        restored = UGParityCase.from_json(case.to_json())

        self.assertEqual(restored.case_id, case.case_id)
        self.assertEqual(restored.model, "sensenova-u1")
        self.assertEqual(restored.task, "vlm")
        self.assertEqual(restored.sampling_params["max_new_tokens"], 16)
        self.assertEqual(restored.dump_points, ("text", "u_logits"))

    def test_u1_tensor_and_image_summary_are_stable(self):
        tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        tensor_summary = UGTensorSummary.from_tensor(tensor)
        same_tensor_summary = UGTensorSummary.from_tensor(tensor.clone())
        image = Image.fromarray(np.full((3, 4, 3), 17, dtype=np.uint8), "RGB")
        image_summary = summarize_ug_image(image)

        self.assertEqual(tensor_summary, same_tensor_summary)
        self.assertEqual(tensor_summary.shape, (2, 2))
        self.assertEqual(tensor_summary.dtype, "torch.float32")
        self.assertEqual(image_summary.size, (4, 3))
        self.assertTrue(image_summary.sha256)

    def test_fake_runner_artifacts_pass_and_mismatch_fail(self):
        case = _case()
        reference = _FakeRunner(runner="official", text="a cup").run(case)
        candidate = _FakeRunner(runner="sglang", text="a cup").run(case)
        mismatch = _FakeRunner(runner="sglang", text="a vase").run(case)

        passed = compare_ug_parity_artifacts(reference, candidate)
        failed = compare_ug_parity_artifacts(reference, mismatch)

        self.assertTrue(passed.passed)
        self.assertFalse(failed.passed)
        self.assertIn("text", {diff.field for diff in failed.diffs})

    def test_write_u1_parity_bundle(self):
        case = _case()
        reference = _FakeRunner(runner="official", text="a cup").run(case)
        candidate = _FakeRunner(runner="sglang", text="a cup").run(case)
        report = compare_ug_parity_artifacts(reference, candidate)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = write_ug_parity_bundle(
                output_dir=tmpdir,
                case=case,
                reference=reference,
                candidate=candidate,
                report=report,
            )

            self.assertTrue((bundle / "case.json").exists())
            self.assertTrue((bundle / "reference.json").exists())
            self.assertTrue((bundle / "candidate.json").exists())
            report_json = json.loads((bundle / "report.json").read_text())
            self.assertTrue(report_json["passed"])

    def test_u1_vlm_official_reference_mode_writes_candidate_error_bundle(self):
        run_from_env = _load_u1_official_parity_harness()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_vqa_repo(root, text="official answer")
            image_path = _write_fake_image(root)
            output_dir = root / "bundle"

            bundle = run_from_env(
                {
                    "SGLANG_TEST_U1_PARITY_MODE": "vlm_official_reference",
                    "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                    "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                    "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                    "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                    "SGLANG_TEST_U1_VLM_IMAGE": str(image_path),
                    "SGLANG_TEST_U1_VLM_QUESTION": "what is here?",
                    "SGLANG_TEST_U1_VLM_MAX_NEW_TOKENS": "4",
                    "SGLANG_TEST_U1_VLM_DEVICE": "cpu",
                }
            )

            reference = json.loads((bundle / "reference.json").read_text())
            candidate = json.loads((bundle / "candidate.json").read_text())
            report = json.loads((bundle / "report.json").read_text())

            self.assertEqual(reference["text"], "official answer")
            self.assertIsNone(reference["error"])
            self.assertIn("not wired", candidate["error"])
            self.assertFalse(report["passed"])
            self.assertEqual(report["diffs"][0]["field"], "error")

    def test_u1_vlm_official_reference_mode_can_run_sglang_candidate(self):
        run_from_env = _load_u1_official_parity_harness()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_vqa_repo(root, text="aligned answer")
            image_path = _write_fake_image(root)
            output_dir = root / "bundle"

            bundle = run_from_env(
                {
                    "SGLANG_TEST_U1_PARITY_MODE": "vlm_official_reference",
                    "SGLANG_TEST_U1_PARITY_RUN_SGLANG_CANDIDATE": "1",
                    "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                    "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                    "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                    "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                    "SGLANG_TEST_U1_VLM_IMAGE": str(image_path),
                    "SGLANG_TEST_U1_VLM_QUESTION": "what is here?",
                    "SGLANG_TEST_U1_VLM_MAX_NEW_TOKENS": "4",
                    "SGLANG_TEST_U1_VLM_DEVICE": "cpu",
                }
            )

            candidate = json.loads((bundle / "candidate.json").read_text())
            report = json.loads((bundle / "report.json").read_text())

            self.assertEqual(candidate["text"], "aligned answer")
            self.assertFalse(candidate["metadata"]["native_srt_model_runner"])
            self.assertEqual(candidate["debug_counters"]["prefill_count"], 1)
            self.assertTrue(report["passed"])

    def test_u1_vlm_official_reference_mode_can_run_native_srt_candidate(self):
        module = _load_u1_official_parity_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_vqa_repo(root, text="native answer")
            image_path = _write_fake_image(root)
            output_dir = root / "bundle"

            with patch.object(module, "_run_sglang_native_vlm_candidate") as run_native:
                run_native.return_value = UGParityArtifact(
                    case_id="u1-vlm-official-reference",
                    model="sensenova-u1",
                    task="vlm",
                    runner="sglang",
                    text="native answer",
                    image=summarize_ug_image(image_path),
                    metadata={
                        "candidate_backend": "u1_native_srt_vlm_kv_decode",
                        "native_srt_model_runner": True,
                        "kv_decode": True,
                        "prefill_forwards": 1,
                        "decode_forwards": 3,
                    },
                )
                bundle = module.run_u1_official_parity_from_env(
                    {
                        "SGLANG_TEST_U1_PARITY_MODE": "vlm_official_reference",
                        "SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE": "1",
                        "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                        "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                        "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                        "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                        "SGLANG_TEST_U1_VLM_IMAGE": str(image_path),
                        "SGLANG_TEST_U1_VLM_QUESTION": "what is here?",
                        "SGLANG_TEST_U1_VLM_MAX_NEW_TOKENS": "4",
                        "SGLANG_TEST_U1_VLM_DEVICE": "cpu",
                    }
                )

            candidate = json.loads((bundle / "candidate.json").read_text())
            report = json.loads((bundle / "report.json").read_text())

            run_native.assert_called_once()
            self.assertEqual(candidate["text"], "native answer")
            self.assertTrue(candidate["metadata"]["native_srt_model_runner"])
            self.assertEqual(
                candidate["metadata"]["candidate_backend"],
                "u1_native_srt_vlm_kv_decode",
            )
            self.assertTrue(candidate["metadata"]["kv_decode"])
            self.assertEqual(candidate["metadata"]["prefill_forwards"], 1)
            self.assertEqual(candidate["metadata"]["decode_forwards"], 3)
            self.assertTrue(report["passed"])

    def test_u1_t2i_official_reference_mode_writes_candidate_error_bundle(self):
        run_from_env = _load_u1_official_parity_harness()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_t2i_repo(root, color=(120, 10, 20))
            output_dir = root / "bundle"

            bundle = run_from_env(
                {
                    "SGLANG_TEST_U1_PARITY_MODE": "t2i_official_reference",
                    "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                    "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                    "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                    "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                    "SGLANG_TEST_U1_T2I_PROMPT": "draw a cup",
                    "SGLANG_TEST_U1_T2I_WIDTH": "8",
                    "SGLANG_TEST_U1_T2I_HEIGHT": "8",
                    "SGLANG_TEST_U1_T2I_NUM_STEPS": "1",
                    "SGLANG_TEST_U1_T2I_DEVICE": "cpu",
                }
            )

            reference = json.loads((bundle / "reference.json").read_text())
            candidate = json.loads((bundle / "candidate.json").read_text())
            report = json.loads((bundle / "report.json").read_text())

            self.assertIsNone(reference["error"])
            self.assertEqual(reference["image"]["size"], [8, 8])
            self.assertIn("not requested", candidate["error"])
            self.assertFalse(report["passed"])

    def test_u1_t2i_official_reference_mode_can_run_native_srt_candidate(self):
        module = _load_u1_official_parity_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_t2i_repo(root, color=(120, 10, 20))
            output_dir = root / "bundle"
            candidate_image = root / "candidate.png"
            Image.fromarray(
                np.full((8, 8, 3), [122, 11, 21], dtype=np.uint8), "RGB"
            ).save(candidate_image)

            with patch.object(module, "_run_sglang_native_t2i_candidate") as run_native:
                run_native.return_value = UGParityArtifact(
                    case_id="u1-t2i-official-reference",
                    model="sensenova-u1",
                    task="t2i",
                    runner="sglang",
                    image=summarize_ug_image(candidate_image),
                    metadata={
                        "candidate_backend": "u1_native_srt_t2i_pixel_flow",
                        "native_srt_model_runner": True,
                        "image_path": str(candidate_image),
                        "prefill_forwards": 1,
                        "temp_g_forward_count": 1,
                    },
                )
                bundle = module.run_u1_official_parity_from_env(
                    {
                        "SGLANG_TEST_U1_PARITY_MODE": "t2i_official_reference",
                        "SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE": "1",
                        "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                        "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                        "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                        "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                        "SGLANG_TEST_U1_T2I_PROMPT": "draw a cup",
                        "SGLANG_TEST_U1_T2I_WIDTH": "8",
                        "SGLANG_TEST_U1_T2I_HEIGHT": "8",
                        "SGLANG_TEST_U1_T2I_NUM_STEPS": "1",
                        "SGLANG_TEST_U1_T2I_DEVICE": "cpu",
                        "SGLANG_TEST_U1_T2I_MEAN_THRESHOLD": "4",
                        "SGLANG_TEST_U1_T2I_MAX_THRESHOLD": "4",
                    }
                )

            candidate = json.loads((bundle / "candidate.json").read_text())
            report = json.loads((bundle / "report.json").read_text())

            run_native.assert_called_once()
            self.assertEqual(
                run_native.call_args.kwargs["attention_backend"], "torch_native"
            )
            self.assertEqual(run_native.call_args.kwargs["dtype"], "bfloat16")
            self.assertEqual(run_native.call_args.kwargs["mem_fraction_static"], 0.80)
            self.assertTrue(candidate["metadata"]["native_srt_model_runner"])
            self.assertEqual(candidate["metadata"]["prefill_forwards"], 1)
            self.assertTrue(report["passed"])
            self.assertLessEqual(report["metadata"]["image_mean_abs_diff"], 4)
            self.assertLessEqual(report["metadata"]["image_max_abs_diff"], 4)
            self.assertEqual(report["metadata"]["max_threshold_policy"], "record_only")
            self.assertLessEqual(report["metadata"]["image_abs_diff_p99"], 4)
            self.assertGreater(report["metadata"]["image_psnr_db"], 30)
            self.assertGreater(report["metadata"]["image_ssim_luma_global"], 0.99)
            self.assertTrue(Path(report["metadata"]["image_absdiff_path"]).exists())
            self.assertTrue(Path(report["metadata"]["image_heatmap_path"]).exists())

    def test_u1_edit_official_reference_mode_writes_candidate_error_bundle(self):
        run_from_env = _load_u1_official_parity_harness()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_edit_repo(root, color=(40, 140, 20))
            image_path = _write_fake_image(root)
            output_dir = root / "bundle"

            bundle = run_from_env(
                {
                    "SGLANG_TEST_U1_PARITY_MODE": "edit_official_reference",
                    "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                    "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                    "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                    "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                    "SGLANG_TEST_U1_EDIT_IMAGE": str(image_path),
                    "SGLANG_TEST_U1_EDIT_PROMPT": "make it green",
                    "SGLANG_TEST_U1_EDIT_WIDTH": "8",
                    "SGLANG_TEST_U1_EDIT_HEIGHT": "8",
                    "SGLANG_TEST_U1_EDIT_NUM_STEPS": "1",
                    "SGLANG_TEST_U1_EDIT_DEVICE": "cpu",
                }
            )

            reference = json.loads((bundle / "reference.json").read_text())
            candidate = json.loads((bundle / "candidate.json").read_text())
            report = json.loads((bundle / "report.json").read_text())

            self.assertIsNone(reference["error"])
            self.assertEqual(reference["image"]["size"], [8, 8])
            self.assertIn("not requested", candidate["error"])
            self.assertFalse(report["passed"])

    def test_u1_edit_official_reference_mode_can_run_native_srt_candidate(self):
        module = _load_u1_official_parity_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_edit_repo(root, color=(40, 140, 20))
            image_path = _write_fake_image(root)
            output_dir = root / "bundle"
            candidate_image = root / "candidate.png"
            Image.fromarray(
                np.full((8, 8, 3), [42, 141, 21], dtype=np.uint8), "RGB"
            ).save(candidate_image)

            with patch.object(
                module, "_run_sglang_native_edit_candidate"
            ) as run_native:
                run_native.return_value = UGParityArtifact(
                    case_id="u1-edit-official-reference",
                    model="sensenova-u1",
                    task="edit",
                    runner="sglang",
                    image=summarize_ug_image(candidate_image),
                    metadata={
                        "candidate_backend": "u1_native_srt_edit_pixel_flow",
                        "native_srt_model_runner": True,
                        "image_path": str(candidate_image),
                        "prefill_forwards": 1,
                        "temp_g_forward_count": 1,
                    },
                )
                bundle = module.run_u1_official_parity_from_env(
                    {
                        "SGLANG_TEST_U1_PARITY_MODE": "edit_official_reference",
                        "SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE": "1",
                        "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                        "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                        "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                        "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                        "SGLANG_TEST_U1_EDIT_IMAGE": str(image_path),
                        "SGLANG_TEST_U1_EDIT_PROMPT": "make it green",
                        "SGLANG_TEST_U1_EDIT_WIDTH": "8",
                        "SGLANG_TEST_U1_EDIT_HEIGHT": "8",
                        "SGLANG_TEST_U1_EDIT_NUM_STEPS": "1",
                        "SGLANG_TEST_U1_EDIT_DEVICE": "cpu",
                        "SGLANG_TEST_U1_EDIT_MEAN_THRESHOLD": "4",
                        "SGLANG_TEST_U1_EDIT_MAX_THRESHOLD": "4",
                    }
                )

            candidate = json.loads((bundle / "candidate.json").read_text())
            report = json.loads((bundle / "report.json").read_text())

            run_native.assert_called_once()
            self.assertEqual(
                run_native.call_args.kwargs["attention_backend"], "torch_native"
            )
            self.assertEqual(run_native.call_args.kwargs["dtype"], "bfloat16")
            self.assertEqual(run_native.call_args.kwargs["mem_fraction_static"], 0.80)
            self.assertTrue(candidate["metadata"]["native_srt_model_runner"])
            self.assertEqual(candidate["metadata"]["prefill_forwards"], 1)
            self.assertTrue(report["passed"])
            self.assertEqual(report["metadata"]["max_threshold_policy"], "record_only")
            self.assertLessEqual(report["metadata"]["image_abs_diff_p99"], 4)
            self.assertGreater(report["metadata"]["image_psnr_db"], 30)
            self.assertGreater(report["metadata"]["image_ssim_luma_global"], 0.99)
            self.assertTrue(Path(report["metadata"]["image_absdiff_path"]).exists())
            self.assertTrue(Path(report["metadata"]["image_heatmap_path"]).exists())

    def test_u1_interleave_official_reference_mode_writes_candidate_error_bundle(self):
        run_from_env = _load_u1_official_parity_harness()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_interleave_repo(
                root, text="official interleave text", color=(10, 120, 200)
            )
            output_dir = root / "bundle"

            bundle = run_from_env(
                {
                    "SGLANG_TEST_U1_PARITY_MODE": "interleave_official_reference",
                    "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                    "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                    "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                    "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                    "SGLANG_TEST_U1_INTERLEAVE_PROMPT": "draw and describe",
                    "SGLANG_TEST_U1_INTERLEAVE_WIDTH": "8",
                    "SGLANG_TEST_U1_INTERLEAVE_HEIGHT": "8",
                    "SGLANG_TEST_U1_INTERLEAVE_NUM_STEPS": "1",
                    "SGLANG_TEST_U1_INTERLEAVE_DEVICE": "cpu",
                }
            )

            reference = json.loads((bundle / "reference.json").read_text())
            candidate = json.loads((bundle / "candidate.json").read_text())
            report = json.loads((bundle / "report.json").read_text())

            self.assertIsNone(reference["error"])
            self.assertEqual(reference["task"], "interleave")
            self.assertEqual(reference["text"], "official interleave text")
            self.assertEqual(reference["image"]["size"], [8, 8])
            self.assertIn("not requested", candidate["error"])
            self.assertFalse(report["passed"])

    def test_u1_interleave_official_reference_mode_can_run_native_srt_candidate(self):
        module = _load_u1_official_parity_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_interleave_repo(
                root, text="official interleave text", color=(10, 120, 200)
            )
            output_dir = root / "bundle"
            candidate_image = root / "candidate.png"
            Image.fromarray(
                np.full((8, 8, 3), [11, 121, 201], dtype=np.uint8), "RGB"
            ).save(candidate_image)

            with patch.object(
                module, "_run_sglang_native_interleave_candidate"
            ) as run_native:
                run_native.return_value = UGParityArtifact(
                    case_id="u1-interleave-official-reference",
                    model="sensenova-u1",
                    task="interleave",
                    runner="sglang",
                    text="sglang text after image",
                    image=summarize_ug_image(candidate_image),
                    metadata={
                        "candidate_backend": "u1_native_srt_interleave_pixel_flow",
                        "alignment_gate": "free_run",
                        "native_srt_model_runner": True,
                        "image_path": str(candidate_image),
                        "image_paths": [str(candidate_image)],
                        "same_session_id": True,
                        "prefill_count": 1,
                        "append_image_count": 1,
                        "srt_u_decode_request_count": 1,
                        "segments": [
                            {"type": "text", "text": "sglang text before image"},
                            {"type": "image", "image_path": str(candidate_image)},
                            {"type": "text", "text": "sglang text after image"},
                        ],
                    },
                )
                bundle = module.run_u1_official_parity_from_env(
                    {
                        "SGLANG_TEST_U1_PARITY_MODE": "interleave_official_reference",
                        "SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE": "1",
                        "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                        "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                        "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                        "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                        "SGLANG_TEST_U1_INTERLEAVE_PROMPT": "draw and describe",
                        "SGLANG_TEST_U1_INTERLEAVE_WIDTH": "8",
                        "SGLANG_TEST_U1_INTERLEAVE_HEIGHT": "8",
                        "SGLANG_TEST_U1_INTERLEAVE_NUM_STEPS": "1",
                        "SGLANG_TEST_U1_INTERLEAVE_DEVICE": "cpu",
                        "SGLANG_TEST_U1_INTERLEAVE_MEAN_THRESHOLD": "4",
                        "SGLANG_TEST_U1_INTERLEAVE_MAX_THRESHOLD": "4",
                    }
                )

            candidate = json.loads((bundle / "candidate.json").read_text())
            report = json.loads((bundle / "report.json").read_text())

            run_native.assert_called_once()
            self.assertEqual(
                run_native.call_args.kwargs["attention_backend"], "torch_native"
            )
            self.assertEqual(run_native.call_args.kwargs["dtype"], "bfloat16")
            self.assertEqual(run_native.call_args.kwargs["alignment_gate"], "free_run")
            self.assertTrue(candidate["metadata"]["native_srt_model_runner"])
            self.assertTrue(candidate["metadata"]["same_session_id"])
            self.assertEqual(candidate["metadata"]["append_image_count"], 1)
            self.assertEqual(candidate["metadata"]["srt_u_decode_request_count"], 1)
            self.assertEqual(candidate["metadata"]["alignment_gate"], "free_run")
            self.assertTrue(report["passed"])
            self.assertEqual(report["metadata"]["alignment_gate"], "free_run")
            self.assertTrue(report["metadata"]["fail_on_report"])
            self.assertFalse(report["metadata"]["require_text_exact"])
            self.assertIn("text", report["metadata"]["reference_segment_types"])
            self.assertIn("image", report["metadata"]["candidate_segment_types"])
            self.assertEqual(report["metadata"]["max_threshold_policy"], "record_only")

    def test_u1_interleave_alignment_gate_teacher_forces_reference_text(self):
        module = _load_u1_official_parity_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_interleave_repo(
                root, text="first <image> second", color=(10, 120, 200)
            )
            output_dir = root / "bundle"
            candidate_image = root / "candidate.png"
            Image.fromarray(
                np.full((8, 8, 3), [10, 120, 200], dtype=np.uint8), "RGB"
            ).save(candidate_image)

            with patch.object(
                module, "_run_sglang_native_interleave_candidate"
            ) as run_native:
                run_native.return_value = UGParityArtifact(
                    case_id="u1-interleave-official-reference",
                    model="sensenova-u1",
                    task="interleave",
                    runner="sglang",
                    text="candidate text",
                    image=summarize_ug_image(candidate_image),
                    metadata={
                        "candidate_backend": "u1_native_srt_interleave_pixel_flow",
                        "native_srt_model_runner": True,
                        "image_path": str(candidate_image),
                        "image_paths": [str(candidate_image)],
                        "same_session_id": True,
                        "prefill_count": 1,
                        "append_image_count": 1,
                        "srt_u_decode_request_count": 0,
                        "teacher_forced_reference_text_count": 2,
                        "teacher_forced_reference_marker_count": 1,
                        "segments": [
                            {"type": "text", "text": "candidate before"},
                            {"type": "image", "image_path": str(candidate_image)},
                            {"type": "text", "text": "candidate text"},
                        ],
                    },
                )
                bundle = module.run_u1_official_parity_from_env(
                    {
                        "SGLANG_TEST_U1_PARITY_MODE": "interleave_official_reference",
                        "SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE": "1",
                        "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                        "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                        "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                        "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                        "SGLANG_TEST_U1_INTERLEAVE_PROMPT": "draw and describe",
                        "SGLANG_TEST_U1_INTERLEAVE_WIDTH": "8",
                        "SGLANG_TEST_U1_INTERLEAVE_HEIGHT": "8",
                        "SGLANG_TEST_U1_INTERLEAVE_NUM_STEPS": "1",
                        "SGLANG_TEST_U1_INTERLEAVE_DEVICE": "cpu",
                        "SGLANG_TEST_U1_INTERLEAVE_ALIGNMENT_GATE": "teacher_force_text",
                    }
                )

            report = json.loads((bundle / "report.json").read_text())

            self.assertEqual(
                run_native.call_args.kwargs["teacher_force_text_parts"],
                ["first ", " second"],
            )
            self.assertEqual(
                run_native.call_args.kwargs["alignment_gate"], "teacher_force_text"
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["metadata"]["alignment_gate"], "teacher_force_text")
            self.assertTrue(report["metadata"]["teacher_force_reference_text"])

    def test_u1_interleave_compare_requires_multi_round_sequence_and_commits(self):
        module = _load_u1_official_parity_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_paths = []
            candidate_paths = []
            for index in range(3):
                reference_path = root / f"reference_{index}.png"
                candidate_path = root / f"candidate_{index}.png"
                Image.fromarray(
                    np.full((8, 8, 3), [10 + index, 120, 200], dtype=np.uint8),
                    "RGB",
                ).save(reference_path)
                Image.fromarray(
                    np.full((8, 8, 3), [10 + index, 120, 200], dtype=np.uint8),
                    "RGB",
                ).save(candidate_path)
                reference_paths.append(reference_path)
                candidate_paths.append(candidate_path)

            reference = UGParityArtifact(
                case_id="u1-interleave-official-reference",
                model="sensenova-u1",
                task="interleave",
                runner="official",
                text="text <image> text <image> text <image> text",
                image=summarize_ug_image(reference_paths[0]),
                metadata={
                    "image_path": str(reference_paths[0]),
                    "image_paths": [str(path) for path in reference_paths],
                    "segments": [
                        {"type": "text", "text": "text <image> text"},
                        *[
                            {"type": "image", "image_path": str(path)}
                            for path in reference_paths
                        ],
                    ],
                },
            )
            good_candidate = UGParityArtifact(
                case_id="u1-interleave-official-reference",
                model="sensenova-u1",
                task="interleave",
                runner="sglang",
                text="a b c d",
                image=summarize_ug_image(candidate_paths[0]),
                metadata={
                    "image_path": str(candidate_paths[0]),
                    "image_paths": [str(path) for path in candidate_paths],
                    "same_session_id": True,
                    "prefill_count": 1,
                    "append_image_count": 3,
                    "srt_u_decode_request_count": 9,
                    "segments": [
                        {"type": "text", "text": "a"},
                        {"type": "image", "image_path": str(candidate_paths[0])},
                        {"type": "text", "text": "b"},
                        {"type": "image", "image_path": str(candidate_paths[1])},
                        {"type": "text", "text": "c"},
                        {"type": "image", "image_path": str(candidate_paths[2])},
                        {"type": "text", "text": "d"},
                    ],
                },
            )
            bad_candidate = UGParityArtifact(
                case_id="u1-interleave-official-reference",
                model="sensenova-u1",
                task="interleave",
                runner="sglang",
                text="a b",
                image=summarize_ug_image(candidate_paths[0]),
                metadata={
                    "image_path": str(candidate_paths[0]),
                    "image_paths": [str(path) for path in candidate_paths[:2]],
                    "same_session_id": True,
                    "prefill_count": 1,
                    "append_image_count": 2,
                    "srt_u_decode_request_count": 3,
                    "segments": [
                        {"type": "text", "text": "a"},
                        {"type": "image", "image_path": str(candidate_paths[0])},
                        {"type": "text", "text": "b"},
                        {"type": "image", "image_path": str(candidate_paths[1])},
                    ],
                },
            )

            good_report = module._compare_interleave_artifacts_with_tolerance(
                reference,
                good_candidate,
                mean_threshold=1.0,
                max_threshold=1.0,
                p99_threshold=1.0,
                psnr_threshold=99.0,
                ssim_threshold=0.999,
                require_text=True,
                require_text_exact=False,
                min_candidate_images=3,
                require_text_after_each_image=True,
            )
            bad_report = module._compare_interleave_artifacts_with_tolerance(
                reference,
                bad_candidate,
                mean_threshold=1.0,
                max_threshold=1.0,
                p99_threshold=1.0,
                psnr_threshold=99.0,
                ssim_threshold=0.999,
                require_text=True,
                require_text_exact=False,
                min_candidate_images=3,
                require_text_after_each_image=True,
            )

        self.assertTrue(good_report.passed)
        self.assertFalse(bad_report.passed)
        self.assertIn(
            "candidate.image_count", {diff.field for diff in bad_report.diffs}
        )
        self.assertIn(
            "candidate.append_image_count", {diff.field for diff in bad_report.diffs}
        )
        self.assertIn(
            "candidate.interleave_event_types",
            {diff.field for diff in bad_report.diffs},
        )

    def test_u1_interleave_teacher_force_passes_official_token_parts(self):
        module = _load_u1_official_parity_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "bundle"
            image_path = root / "official.png"
            Image.fromarray(
                np.full((8, 8, 3), [10, 120, 200], dtype=np.uint8), "RGB"
            ).save(image_path)
            reference = UGParityArtifact(
                case_id="u1-interleave-official-reference",
                model="sensenova-u1",
                task="interleave",
                runner="official",
                text="first <image> second",
                image=summarize_ug_image(image_path),
                metadata={
                    "image_path": str(image_path),
                    "image_paths": [str(image_path)],
                    "segments": [
                        {"type": "text", "text": "first <image> second"},
                        {"type": "image", "image_path": str(image_path)},
                    ],
                    "position_trace": {
                        "text_parts_token_ids": [[101, 102, 103], [201]]
                    },
                },
            )
            candidate = UGParityArtifact(
                case_id="u1-interleave-official-reference",
                model="sensenova-u1",
                task="interleave",
                runner="sglang",
                text="candidate text",
                image=summarize_ug_image(image_path),
                metadata={
                    "candidate_backend": "u1_native_srt_interleave_pixel_flow",
                    "native_srt_model_runner": True,
                    "image_path": str(image_path),
                    "image_paths": [str(image_path)],
                    "same_session_id": True,
                    "prefill_count": 1,
                    "append_image_count": 1,
                    "srt_u_decode_request_count": 0,
                    "teacher_forced_reference_text_count": 2,
                    "teacher_forced_reference_marker_count": 1,
                    "teacher_forced_reference_token_count": 4,
                    "segments": [
                        {"type": "text", "text": "candidate before"},
                        {"type": "image", "image_path": str(image_path)},
                        {"type": "text", "text": "candidate text"},
                    ],
                },
            )

            with patch.object(
                module, "_run_official_interleave_reference", return_value=reference
            ), patch.object(
                module,
                "_run_sglang_native_interleave_candidate",
                return_value=candidate,
            ) as run_native:
                module.run_u1_official_parity_from_env(
                    {
                        "SGLANG_TEST_U1_PARITY_MODE": "interleave_official_reference",
                        "SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE": "1",
                        "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                        "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                        "SGLANG_TEST_U1_OFFICIAL_REPO": "/fake/repo",
                        "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                        "SGLANG_TEST_U1_INTERLEAVE_ALIGNMENT_GATE": "teacher_force_text",
                    }
                )

            self.assertEqual(
                run_native.call_args.kwargs["teacher_force_text_parts"],
                ["first ", " second"],
            )
            self.assertEqual(
                run_native.call_args.kwargs["teacher_force_token_parts"],
                [[101, 102, 103], [201]],
            )

    def test_u1_interleave_alignment_gate_uses_official_raw_commit(self):
        module = _load_u1_official_parity_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "bundle"
            image_path = root / "official.png"
            raw_path = root / "official.pt"
            Image.fromarray(
                np.full((8, 8, 3), [10, 120, 200], dtype=np.uint8), "RGB"
            ).save(image_path)
            raw_path.write_bytes(b"fake raw tensor path")
            reference = UGParityArtifact(
                case_id="u1-interleave-official-reference",
                model="sensenova-u1",
                task="interleave",
                runner="official",
                text="official text <image> more",
                image=summarize_ug_image(image_path),
                metadata={
                    "image_path": str(image_path),
                    "image_paths": [str(image_path)],
                    "raw_image_paths": [str(raw_path)],
                    "segments": [
                        {"type": "text", "text": "official text <image> more"},
                        {"type": "image", "image_path": str(image_path)},
                    ],
                },
            )
            candidate = UGParityArtifact(
                case_id="u1-interleave-official-reference",
                model="sensenova-u1",
                task="interleave",
                runner="sglang",
                text="candidate text",
                image=summarize_ug_image(image_path),
                metadata={
                    "candidate_backend": "u1_native_srt_interleave_pixel_flow",
                    "native_srt_model_runner": True,
                    "image_path": str(image_path),
                    "image_paths": [str(image_path)],
                    "same_session_id": True,
                    "prefill_count": 1,
                    "append_image_count": 1,
                    "srt_u_decode_request_count": 1,
                    "segments": [
                        {"type": "text", "text": "candidate before"},
                        {"type": "image", "image_path": str(image_path)},
                        {"type": "text", "text": "candidate text"},
                    ],
                },
            )

            with patch.object(
                module, "_run_official_interleave_reference", return_value=reference
            ) as run_reference, patch.object(
                module,
                "_run_sglang_native_interleave_candidate",
                return_value=candidate,
            ) as run_native:
                bundle = module.run_u1_official_parity_from_env(
                    {
                        "SGLANG_TEST_U1_PARITY_MODE": "interleave_official_reference",
                        "SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE": "1",
                        "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                        "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                        "SGLANG_TEST_U1_OFFICIAL_REPO": "/fake/repo",
                        "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                        "SGLANG_TEST_U1_INTERLEAVE_ALIGNMENT_GATE": "official_raw_image_commit",
                    }
                )

            report = json.loads((bundle / "report.json").read_text())

            run_reference.assert_called_once()
            self.assertEqual(
                run_native.call_args.kwargs["teacher_force_commit_paths"], [raw_path]
            )
            self.assertEqual(
                run_native.call_args.kwargs["alignment_gate"],
                "official_raw_image_commit",
            )
            self.assertTrue(report["passed"])
            self.assertEqual(
                report["metadata"]["alignment_gate"],
                "official_raw_image_commit",
            )
            self.assertEqual(
                report["metadata"]["teacher_force_reference_images"], "raw"
            )

    def test_u1_interleave_full_free_run_observation_does_not_raise(self):
        module = _load_u1_official_parity_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "bundle"
            reference_image = root / "reference.png"
            candidate_image = root / "candidate.png"
            Image.fromarray(
                np.full((8, 8, 3), [10, 120, 200], dtype=np.uint8), "RGB"
            ).save(reference_image)
            Image.fromarray(
                np.full((8, 8, 3), [200, 10, 20], dtype=np.uint8), "RGB"
            ).save(candidate_image)
            reference = UGParityArtifact(
                case_id="u1-interleave-official-reference",
                model="sensenova-u1",
                task="interleave",
                runner="official",
                text="official text <image> more",
                image=summarize_ug_image(reference_image),
                metadata={
                    "image_path": str(reference_image),
                    "image_paths": [str(reference_image)],
                    "segments": [
                        {"type": "text", "text": "official text <image> more"},
                        {"type": "image", "image_path": str(reference_image)},
                    ],
                },
            )
            candidate = UGParityArtifact(
                case_id="u1-interleave-official-reference",
                model="sensenova-u1",
                task="interleave",
                runner="sglang",
                text="different candidate text",
                image=summarize_ug_image(candidate_image),
                metadata={
                    "candidate_backend": "u1_native_srt_interleave_pixel_flow",
                    "native_srt_model_runner": True,
                    "image_path": str(candidate_image),
                    "image_paths": [str(candidate_image)],
                    "same_session_id": True,
                    "prefill_count": 1,
                    "append_image_count": 1,
                    "srt_u_decode_request_count": 1,
                    "segments": [
                        {"type": "text", "text": "candidate before"},
                        {"type": "image", "image_path": str(candidate_image)},
                        {"type": "text", "text": "different candidate text"},
                    ],
                },
            )

            with patch.object(
                module, "_run_official_interleave_reference", return_value=reference
            ), patch.object(
                module,
                "_run_sglang_native_interleave_candidate",
                return_value=candidate,
            ):
                bundle = module.run_u1_official_parity_from_env(
                    {
                        "SGLANG_TEST_U1_PARITY_MODE": "interleave_official_reference",
                        "SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE": "1",
                        "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                        "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                        "SGLANG_TEST_U1_OFFICIAL_REPO": "/fake/repo",
                        "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                        "SGLANG_TEST_U1_INTERLEAVE_ALIGNMENT_GATE": "full_free_run_observation",
                        "SGLANG_TEST_U1_INTERLEAVE_MEAN_THRESHOLD": "1",
                    }
                )

            report = json.loads((bundle / "report.json").read_text())

            self.assertFalse(report["passed"])
            self.assertEqual(
                report["metadata"]["alignment_gate"],
                "full_free_run_observation",
            )
            self.assertFalse(report["metadata"]["fail_on_report"])

    def test_u1_image_parity_records_sparse_max_diff_without_failing(self):
        module = _load_u1_official_parity_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_t2i_repo(root, color=(120, 10, 20))
            output_dir = root / "bundle"
            candidate_image = root / "candidate.png"
            candidate = np.full((32, 32, 3), [120, 10, 20], dtype=np.uint8)
            candidate[0, 0, 0] = 255
            Image.fromarray(candidate, "RGB").save(candidate_image)

            with patch.object(module, "_run_sglang_native_t2i_candidate") as run_native:
                run_native.return_value = UGParityArtifact(
                    case_id="u1-t2i-official-reference",
                    model="sensenova-u1",
                    task="t2i",
                    runner="sglang",
                    image=summarize_ug_image(candidate_image),
                    metadata={
                        "candidate_backend": "u1_native_srt_t2i_pixel_flow",
                        "native_srt_model_runner": True,
                        "image_path": str(candidate_image),
                    },
                )
                bundle = module.run_u1_official_parity_from_env(
                    {
                        "SGLANG_TEST_U1_PARITY_MODE": "t2i_official_reference",
                        "SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE": "1",
                        "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                        "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                        "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                        "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                        "SGLANG_TEST_U1_T2I_PROMPT": "draw a cup",
                        "SGLANG_TEST_U1_T2I_WIDTH": "32",
                        "SGLANG_TEST_U1_T2I_HEIGHT": "32",
                        "SGLANG_TEST_U1_T2I_NUM_STEPS": "1",
                        "SGLANG_TEST_U1_T2I_DEVICE": "cpu",
                        "SGLANG_TEST_U1_T2I_MEAN_THRESHOLD": "1",
                        "SGLANG_TEST_U1_T2I_MAX_THRESHOLD": "4",
                        "SGLANG_TEST_U1_T2I_P99_THRESHOLD": "1",
                        "SGLANG_TEST_U1_T2I_PSNR_THRESHOLD": "30",
                        "SGLANG_TEST_U1_T2I_SSIM_THRESHOLD": "0.95",
                    }
                )

            report = json.loads((bundle / "report.json").read_text())

            self.assertTrue(report["passed"])
            self.assertGreater(report["metadata"]["image_max_abs_diff"], 4)
            self.assertEqual(report["metadata"]["max_threshold_policy"], "record_only")
            self.assertLessEqual(report["metadata"]["image_abs_diff_p99"], 1)

    def test_runtime_import_firewall_blocks_official_u1_imports(self):
        repo = Path(__file__).resolve().parents[5]
        runtime_root = repo / "python" / "sglang"
        forbidden = [
            re.compile(r"^\s*(from|import)\s+sensenova(?:\.|\s|$)", re.MULTILINE),
            re.compile(r"^\s*(from|import)\s+seed(?:\.|\s|$)", re.MULTILINE),
            re.compile(r"^\s*(from|import)\s+u1_official(?:\.|\s|$)", re.MULTILINE),
        ]
        offenders = []

        for path in runtime_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in forbidden):
                offenders.append(str(path.relative_to(repo)))

        self.assertEqual(offenders, [])


class _FakeRunner(UGParityRunner):
    def __init__(self, *, runner: str, text: str):
        self.runner = runner
        self.text = text

    def run(self, case: UGParityCase) -> UGParityArtifact:
        image = Image.fromarray(np.full((4, 4, 3), 91, dtype=np.uint8), "RGB")
        return UGParityArtifact(
            case_id=case.case_id,
            model=case.model,
            task=case.task,
            runner=self.runner,
            text=self.text,
            image=summarize_ug_image(image),
            tensors={"u_logits": UGTensorSummary.from_tensor(torch.ones(2, 2))},
            metadata={"seed": case.seed},
        )


def _case():
    return UGParityCase(
        case_id="u1-t2i-smoke",
        model="sensenova-u1",
        task="t2i",
        prompt="draw a cup",
        seed=7,
        sampling_params={"num_inference_steps": 2},
        dump_points=("image", "u_logits"),
    )


def _load_u1_official_parity_harness():
    return _load_u1_official_parity_module().run_u1_official_parity_from_env


def _load_u1_official_parity_module():
    repo = Path(__file__).resolve().parents[5]
    path = repo / "test/registered/scheduler/test_u1_official_parity_harness.py"
    spec = importlib.util.spec_from_file_location("u1_official_parity_harness", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fake_u1_vqa_repo(root: Path, *, text: str) -> Path:
    official_repo = root / "official"
    vqa_dir = official_repo / "examples/vqa"
    vqa_dir.mkdir(parents=True)
    fake_script = vqa_dir / "inference.py"
    fake_script.write_text(
        "\n".join(
            [
                "import argparse",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--model_path')",
                "parser.add_argument('--image')",
                "parser.add_argument('--question')",
                "parser.add_argument('--output')",
                "parser.add_argument('--max_new_tokens')",
                "parser.add_argument('--device')",
                "parser.add_argument('--dtype')",
                "parser.add_argument('--attn_backend')",
                "args = parser.parse_args()",
                f"Path(args.output).write_text({text!r})",
                "print('fake official u1 ok')",
            ]
        ),
        encoding="utf-8",
    )
    return official_repo


def _write_fake_u1_t2i_repo(root: Path, *, color: tuple[int, int, int]) -> Path:
    official_repo = root / "official-t2i"
    t2i_dir = official_repo / "examples/t2i"
    t2i_dir.mkdir(parents=True)
    fake_script = t2i_dir / "inference.py"
    fake_script.write_text(
        "\n".join(
            [
                "import argparse",
                "import numpy as np",
                "from PIL import Image",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--model_path')",
                "parser.add_argument('--prompt')",
                "parser.add_argument('--output')",
                "parser.add_argument('--width', type=int)",
                "parser.add_argument('--height', type=int)",
                "parser.add_argument('--cfg_scale')",
                "parser.add_argument('--cfg_norm')",
                "parser.add_argument('--timestep_shift')",
                "parser.add_argument('--cfg_interval', nargs=2)",
                "parser.add_argument('--num_steps')",
                "parser.add_argument('--batch_size')",
                "parser.add_argument('--seed')",
                "parser.add_argument('--device')",
                "parser.add_argument('--dtype')",
                "parser.add_argument('--attn_backend')",
                "args = parser.parse_args()",
                f"color = np.array({list(color)!r}, dtype=np.uint8)",
                "arr = np.zeros((args.height, args.width, 3), dtype=np.uint8)",
                "arr[:] = color",
                "Image.fromarray(arr, 'RGB').save(args.output)",
                "print('fake official u1 t2i ok')",
            ]
        ),
        encoding="utf-8",
    )
    return official_repo


def _write_fake_u1_edit_repo(root: Path, *, color: tuple[int, int, int]) -> Path:
    official_repo = root / "official-edit"
    edit_dir = official_repo / "examples/editing"
    edit_dir.mkdir(parents=True)
    fake_script = edit_dir / "inference.py"
    fake_script.write_text(
        "\n".join(
            [
                "import argparse",
                "import numpy as np",
                "from PIL import Image",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--model_path')",
                "parser.add_argument('--prompt')",
                "parser.add_argument('--image')",
                "parser.add_argument('--output')",
                "parser.add_argument('--width', type=int)",
                "parser.add_argument('--height', type=int)",
                "parser.add_argument('--cfg_scale')",
                "parser.add_argument('--img_cfg_scale')",
                "parser.add_argument('--cfg_norm')",
                "parser.add_argument('--timestep_shift')",
                "parser.add_argument('--cfg_interval', nargs=2)",
                "parser.add_argument('--num_steps')",
                "parser.add_argument('--batch_size')",
                "parser.add_argument('--seed')",
                "parser.add_argument('--device')",
                "parser.add_argument('--dtype')",
                "parser.add_argument('--attn_backend')",
                "parser.add_argument('--no-do-resize', action='store_true')",
                "args = parser.parse_args()",
                f"color = np.array({list(color)!r}, dtype=np.uint8)",
                "arr = np.zeros((args.height, args.width, 3), dtype=np.uint8)",
                "arr[:] = color",
                "Image.fromarray(arr, 'RGB').save(args.output)",
                "print('fake official u1 edit ok')",
            ]
        ),
        encoding="utf-8",
    )
    return official_repo


def _write_fake_u1_interleave_repo(
    root: Path,
    *,
    text: str,
    color: tuple[int, int, int],
) -> Path:
    official_repo = root / "official-interleave"
    interleave_dir = official_repo / "examples/interleave"
    interleave_dir.mkdir(parents=True)
    fake_script = interleave_dir / "inference.py"
    fake_script.write_text(
        "\n".join(
            [
                "import argparse",
                "import numpy as np",
                "from pathlib import Path",
                "from PIL import Image",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--model_path')",
                "parser.add_argument('--prompt')",
                "parser.add_argument('--output_dir')",
                "parser.add_argument('--stem')",
                "parser.add_argument('--width', type=int)",
                "parser.add_argument('--height', type=int)",
                "parser.add_argument('--cfg_scale')",
                "parser.add_argument('--img_cfg_scale')",
                "parser.add_argument('--timestep_shift')",
                "parser.add_argument('--cfg_interval', nargs=2)",
                "parser.add_argument('--num_steps')",
                "parser.add_argument('--seed')",
                "parser.add_argument('--device')",
                "parser.add_argument('--dtype')",
                "parser.add_argument('--attn_backend')",
                "parser.add_argument('--think_mode', action='store_true')",
                "parser.add_argument('--no-think_mode', action='store_true')",
                "parser.add_argument('--image', action='append', default=[])",
                "args = parser.parse_args()",
                "out = Path(args.output_dir)",
                "out.mkdir(parents=True, exist_ok=True)",
                f"(out / f'{{args.stem}}.txt').write_text('# PROMPT\\n' + args.prompt + '\\n\\n# OUTPUT\\n' + {text!r})",
                f"color = np.array({list(color)!r}, dtype=np.uint8)",
                "arr = np.zeros((args.height, args.width, 3), dtype=np.uint8)",
                "arr[:] = color",
                "Image.fromarray(arr, 'RGB').save(out / f'{args.stem}_image_0.png')",
                "print('fake official u1 interleave ok')",
            ]
        ),
        encoding="utf-8",
    )
    return official_repo


def _write_fake_image(root: Path) -> Path:
    image_path = root / "image.png"
    Image.fromarray(np.full((8, 8, 3), 23, dtype=np.uint8), "RGB").save(image_path)
    return image_path


if __name__ == "__main__":
    unittest.main()
