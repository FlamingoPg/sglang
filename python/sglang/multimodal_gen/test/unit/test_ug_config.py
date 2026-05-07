# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from sglang.multimodal_gen.configs.pipeline_configs.ug import UGPipelineConfig
from sglang.multimodal_gen.configs.sample.ug import (
    UGSamplingParams,
    mark_ug_explicit_sampling_fields,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ug_bagel import (
    apply_bagel_official_sampling_defaults,
)
from sglang.multimodal_gen.runtime.pipelines.ug import _resolve_vlm_max_new_tokens
from sglang.srt.ug.interleaved import DEFAULT_UG_TEXT_MAX_NEW_TOKENS


class TestUGSamplingParams(unittest.TestCase):
    def test_defaults_are_image_generation_params(self):
        params = UGSamplingParams(prompt="a cube")

        self.assertEqual(params.height, 1024)
        self.assertEqual(params.width, 1024)
        self.assertEqual(params.num_frames, 1)
        self.assertEqual(params.num_inference_steps, 50)
        self.assertEqual(params.cfg_text_scale, 1.0)
        self.assertEqual(params.cfg_img_scale, 1.0)
        self.assertEqual(params.cfg_interval, [0.4, 1.0])

    def test_cfg_interval_must_be_ordered_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "cfg_interval"):
            UGSamplingParams(cfg_interval=[0.8, 0.2])

        with self.assertRaisesRegex(ValueError, "cfg_interval"):
            UGSamplingParams(cfg_interval=[0.1])

    def test_cfg_renorm_type_must_be_known(self):
        self.assertEqual(
            UGSamplingParams(cfg_renorm_type="none").cfg_renorm_type, "none"
        )

        with self.assertRaisesRegex(ValueError, "cfg_renorm_type"):
            UGSamplingParams(cfg_renorm_type="local")

    def test_ug_numeric_fields_must_be_finite_non_negative(self):
        with self.assertRaisesRegex(ValueError, "cfg_text_scale"):
            UGSamplingParams(cfg_text_scale=math.nan)

        with self.assertRaisesRegex(ValueError, "cfg_img_scale"):
            UGSamplingParams(cfg_img_scale=-1.0)

    def test_bagel_official_defaults_are_model_specific(self):
        t2i_params = UGSamplingParams(prompt="a cup")

        apply_bagel_official_sampling_defaults(
            t2i_params,
            mode="t2i",
            has_input_image=False,
        )

        self.assertEqual(t2i_params.cfg_text_scale, 4.0)
        self.assertEqual(t2i_params.cfg_img_scale, 1.5)
        self.assertEqual(t2i_params.cfg_interval, [0.4, 1.0])
        self.assertEqual(t2i_params.cfg_renorm_type, "global")
        self.assertEqual(t2i_params.num_inference_steps, 50)

        edit_params = UGSamplingParams(prompt="edit the cup")

        apply_bagel_official_sampling_defaults(
            edit_params,
            mode="interleave",
            has_input_image=True,
        )

        self.assertEqual(edit_params.cfg_text_scale, 4.0)
        self.assertEqual(edit_params.cfg_img_scale, 2.0)
        self.assertEqual(edit_params.cfg_interval, [0.0, 1.0])
        self.assertEqual(edit_params.cfg_renorm_type, "text_channel")

    def test_bagel_defaults_do_not_clobber_explicit_user_values(self):
        params = mark_ug_explicit_sampling_fields(
            UGSamplingParams(cfg_text_scale=1.0, cfg_img_scale=1.0),
            {"cfg_text_scale"},
        )

        apply_bagel_official_sampling_defaults(
            params,
            mode="t2i",
            has_input_image=False,
        )

        self.assertEqual(params.cfg_text_scale, 1.0)
        self.assertEqual(params.cfg_img_scale, 1.5)


class TestUGPipelineConfig(unittest.TestCase):
    def test_validate_runtime_accepts_single_gpu_monolithic(self):
        config = UGPipelineConfig()

        self.assertEqual(
            config.validate_runtime(
                num_gpus=1,
                enable_cfg_parallel=False,
                disagg_mode=False,
            ),
            [],
        )

    def test_validate_runtime_rejects_unsupported_modes(self):
        config = UGPipelineConfig()

        self.assertEqual(
            config.validate_runtime(
                num_gpus=2,
                enable_cfg_parallel=True,
                disagg_mode=True,
            ),
            ["num_gpus", "enable_cfg_parallel", "disagg_mode"],
        )


class TestUGTextGenerationDefaults(unittest.TestCase):
    def test_vlm_default_max_new_tokens_is_not_smoke_sized(self):
        self.assertEqual(
            _resolve_vlm_max_new_tokens({}),
            DEFAULT_UG_TEXT_MAX_NEW_TOKENS,
        )

    def test_vlm_max_new_tokens_can_be_overridden(self):
        self.assertEqual(_resolve_vlm_max_new_tokens({"max_new_tokens": 16}), 16)


if __name__ == "__main__":
    unittest.main()
