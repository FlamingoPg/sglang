# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.ug import UGPipelineConfig
from sglang.multimodal_gen.configs.sample.ug import UGSamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines.ug import UGPipeline
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


def _make_server_args() -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_config=UGPipelineConfig(
            default_height=32,
            default_width=32,
            latent_downsample=16,
            latent_patch_size=2,
            latent_channel=16,
        ),
        num_gpus=1,
        enable_cfg_parallel=False,
        disagg_mode=False,
        disagg_role=RoleType.MONOLITHIC,
        comfyui_mode=True,
    )


class TestUGDiffusionPipeline(unittest.TestCase):
    def test_fake_pipeline_runs_g_denoise_path(self):
        server_args = _make_server_args()
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )

        self.assertEqual(
            [stage.__class__.__name__ for stage in pipeline.stages],
            ["UGContextStage", "UGLatentStage", "UGDenoiseStage", "UGDecodeStage"],
        )

        batch = Req(
            sampling_params=UGSamplingParams(
                prompt="text and image",
                width=32,
                height=32,
                seed=123,
                num_inference_steps=4,
                return_trajectory_latents=True,
                suppress_logs=True,
            ),
            condition_image=Image.new("RGB", (16, 16), color="white"),
        )

        result = pipeline.forward(batch, server_args)

        self.assertEqual(result.output.shape, (1, 32, 32, 3))
        self.assertEqual(result.latents.shape, (1, 4, 64))
        self.assertEqual(result.extra["ug_contexts"].full.token_count, 7)
        self.assertIsNotNone(result.extra["ug_contexts"].full.session)
        self.assertEqual(result.extra["ug_post_image_segment"].type, "text")
        self.assertEqual(
            result.extra["ug_post_image_segment"].text,
            "generated_text_after_image",
        )
        self.assertEqual(result.trajectory_latents.shape[0], 3)
        self.assertEqual(result.trajectory_timesteps.shape[0], 3)

        counters = pipeline.get_module("ug_bridge").runtime.get_debug_counters(
            result.extra["ug_contexts"].full.session
        )
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["velocity_count"], 3)
        self.assertEqual(counters["append_image_count"], 1)
        self.assertEqual(counters["decode_count"], 2)
        self.assertEqual(counters["state"], "u_decode")

    def test_runtime_guard_rejects_cfg_parallel(self):
        server_args = _make_server_args()
        server_args.enable_cfg_parallel = True
        with patch(_GLOBAL_ARGS_PATCH, return_value=server_args):
            pipeline = UGPipeline(
                "sglang-internal/fake-ug",
                server_args,
                executor=SyncExecutor(server_args),
            )

        batch = Req(
            sampling_params=UGSamplingParams(
                prompt="guard",
                width=32,
                height=32,
                num_inference_steps=2,
                suppress_logs=True,
            )
        )

        with self.assertRaisesRegex(ValueError, "enable_cfg_parallel"):
            pipeline.forward(batch, server_args)


if __name__ == "__main__":
    unittest.main()
