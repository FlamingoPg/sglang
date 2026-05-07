# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class TestU1StandaloneEntrypointSmoke(unittest.TestCase):
    def test_u1_standalone_entrypoint_smoke(self):
        run_u1_standalone_entrypoint_from_env(os.environ)


def run_u1_standalone_entrypoint_from_env(env) -> Path:
    if env.get("SGLANG_TEST_U1_STANDALONE") != "1":
        raise unittest.SkipTest(
            "U1 standalone entrypoint smoke is opt-in; set "
            "SGLANG_TEST_U1_STANDALONE=1"
        )

    cuda_visible_devices = env.get("SGLANG_TEST_U1_CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    from sglang.multimodal_gen.configs.pipeline_configs.ug import UGPipelineConfig
    from sglang.multimodal_gen.configs.sample.ug import UGSamplingParams
    from sglang.multimodal_gen.runtime.pipelines.ug import UGPipeline
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs as UGServerArgs,
        set_global_server_args,
    )
    from sglang.srt.ug.interleaved import UGInterleavedRequest
    from sglang.srt.models.neo_chat_ug import create_u1_srt_scheduler

    output_dir = Path(
        env.get("SGLANG_TEST_U1_STANDALONE_OUTPUT")
        or tempfile.mkdtemp(prefix="u1-standalone-smoke-")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = env.get("SGLANG_TEST_U1_MODEL_PATH") or (
        "/data/models/SenseNova-U1-8B-MoT"
    )
    prompt = env.get("SGLANG_TEST_U1_STANDALONE_PROMPT") or (
        "Create an image of a red ceramic cup, then briefly describe the result."
    )
    width = int(env.get("SGLANG_TEST_U1_STANDALONE_WIDTH") or "512")
    height = int(env.get("SGLANG_TEST_U1_STANDALONE_HEIGHT") or "512")
    num_steps = int(env.get("SGLANG_TEST_U1_STANDALONE_NUM_STEPS") or "50")
    max_images = int(env.get("SGLANG_TEST_U1_STANDALONE_MAX_IMAGES") or "1")
    min_images = int(env.get("SGLANG_TEST_U1_STANDALONE_MIN_IMAGES") or "1")
    max_text_segments = int(
        env.get("SGLANG_TEST_U1_STANDALONE_MAX_TEXT_SEGMENTS") or "24"
    )
    decode_chunk = int(env.get("SGLANG_TEST_U1_STANDALONE_U_DECODE_CHUNK") or "8")
    cfg_scale = float(env.get("SGLANG_TEST_U1_STANDALONE_CFG_SCALE") or "4.0")
    img_cfg_scale = float(env.get("SGLANG_TEST_U1_STANDALONE_IMG_CFG_SCALE") or "1.0")
    timestep_shift = float(env.get("SGLANG_TEST_U1_STANDALONE_TIMESTEP_SHIFT") or "3.0")
    seed = int(env.get("SGLANG_TEST_U1_STANDALONE_SEED") or "42")
    attention_backend = env.get("SGLANG_TEST_U1_STANDALONE_ATTENTION_BACKEND") or None
    mem_fraction_static = float(
        env.get("SGLANG_TEST_U1_STANDALONE_MEM_FRACTION_STATIC") or "0.25"
    )
    think = _parse_bool(env.get("SGLANG_TEST_U1_STANDALONE_THINK"), False)

    handle = create_u1_srt_scheduler(
        checkpoint_dir=model_path,
        gpu_id=0,
        mem_fraction_static=mem_fraction_static,
        chunked_prefill_size=-1,
        attention_backend=attention_backend,
    )
    try:
        server_args = UGServerArgs(
            model_path=model_path,
            num_gpus=1,
            enable_cfg_parallel=False,
            pipeline_class_name="UGPipeline",
            pipeline_config=UGPipelineConfig(
                default_height=height,
                default_width=width,
            ),
        )
        setattr(server_args, "ug_srt_scheduler", handle.scheduler)
        setattr(server_args, "ug_srt_scheduler_handle", handle)
        setattr(server_args, "ug_srt_u_decode_max_new_tokens", decode_chunk)
        setattr(server_args, "ug_srt_attention_backend", attention_backend)
        set_global_server_args(server_args)

        pipeline = UGPipeline(model_path, server_args, executor=SimpleNamespace())
        sampling_params = UGSamplingParams(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            cfg_text_scale=cfg_scale,
            cfg_img_scale=img_cfg_scale,
            cfg_interval=[0.0, 1.0],
            cfg_renorm_type="none",
            timestep_shift=timestep_shift,
            seed=seed,
            think=think,
            think_max_new_tokens=max(1, min(decode_chunk, 8)),
        )
        request = UGInterleavedRequest.from_segments(
            [{"type": "text", "text": prompt}],
            sampling_params=sampling_params,
            metadata={
                "mode": "interleave",
                "max_interleave_images": max_images,
                "max_interleave_text_segments": max_text_segments,
            },
        )
        response = pipeline.forward_interleaved(request, server_args=server_args)
    finally:
        flush_cache = getattr(handle.scheduler, "flush_cache", None)
        if callable(flush_cache):
            try:
                flush_cache(empty_cache=True)
            except Exception:
                pass
        handle.close()

    segments = response.to_segments()
    image_paths = []
    text_segments = []
    display_segments = []
    for index, segment in enumerate(segments):
        if segment.get("type") == "text":
            text = str(segment.get("text") or "")
            text_segments.append(text)
            display_segments.append({"type": "text", "text": text})
        elif segment.get("type") == "image":
            image = segment.get("image")
            if image is None:
                continue
            image_path = output_dir / f"standalone_image_{len(image_paths)}.png"
            image.save(image_path)
            image_paths.append(str(image_path))
            display_segments.append(
                {
                    "type": "image",
                    "image_path": str(image_path),
                    "metadata": dict(segment.get("metadata") or {}),
                }
            )

    stats_dict = dataclasses.asdict(response.stats) if response.stats else None
    summary = {
        "model_path": model_path,
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_steps": num_steps,
        "max_images": max_images,
        "segment_types": [segment.get("type") for segment in segments],
        "segments": display_segments,
        "text": "".join(text_segments),
        "image_paths": image_paths,
        "stats": stats_dict,
        "attention_backend": getattr(
            handle.scheduler.server_args, "attention_backend", None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if len(image_paths) < min_images:
        raise AssertionError(
            f"Expected at least {min_images} generated image(s), got {len(image_paths)}"
        )
    if not any(text.strip() for text in text_segments):
        raise AssertionError("Expected standalone UG smoke to produce text")
    first_image_index = next(
        (i for i, segment in enumerate(segments) if segment.get("type") == "image"),
        None,
    )
    if first_image_index is None:
        raise AssertionError("Expected standalone UG smoke to produce an image")
    if not any(
        segment.get("type") == "text" and str(segment.get("text") or "").strip()
        for segment in segments[first_image_index + 1 :]
    ):
        raise AssertionError(
            "Expected standalone UG smoke to continue U decode after G"
        )
    if stats_dict is None:
        raise AssertionError("Expected standalone UG smoke to return runtime stats")
    if not stats_dict.get("session_id"):
        raise AssertionError("Expected runtime stats to include a session id")
    if int(stats_dict.get("prefill_count") or 0) != 1:
        raise AssertionError(f"Expected one U prefill, got {stats_dict}")
    if int(stats_dict.get("append_image_count") or 0) < len(image_paths):
        raise AssertionError(
            f"Expected generated images to be committed, got {stats_dict}"
        )
    if int(stats_dict.get("srt_u_decode_request_count") or 0) <= 0:
        raise AssertionError(f"Expected SRT U decode requests, got {stats_dict}")

    return output_dir


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    unittest.main()
