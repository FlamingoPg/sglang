# SPDX-License-Identifier: Apache-2.0

import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig

from sglang.multimodal_gen.configs.sample.ug import UGSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ug_u1 import (
    U1PixelFlowGSegmentExecutor,
    _u1_guidance_branch,
    _u1_patch_grid,
    _u1_timesteps,
)
from sglang.srt.configs.model_config import is_multimodal_model
from sglang.srt.configs.neo_chat import NEOChatConfig, NEOVisionConfig
from sglang.srt.models.neo_chat import (
    NEOChatModel,
    build_u1_block_causal_allowed_mask,
    build_u1_vlm_input_info,
    build_u1_vlm_thw_indexes,
    map_u1_language_model_weight_name,
)
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.ug.u1 import (
    U1UGModelAdapter,
    build_u1_t2i_prompt,
    build_u1_t2i_uncondition_prompt,
    build_u1_vlm_prompt,
)


class TestU1UGBackend(unittest.TestCase):
    def test_neo_chat_config_and_registry(self):
        config = NEOChatConfig(
            vision_config={
                "architectures": ["NEOVisionModel"],
                "patch_size": 16,
                "hidden_size": 1024,
                "llm_hidden_size": 4096,
            },
            llm_config={
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 1,
                "vocab_size": 128,
                "rope_theta_hw": 10000.0,
            },
            downsample_ratio=0.5,
        )

        self.assertIsInstance(config.vision_config, NEOVisionConfig)
        self.assertEqual(config.model_type, "neo_chat")
        self.assertIs(config.get_text_config(), config.llm_config)

        auto_config = AutoConfig.for_model(
            "neo_chat",
            vision_config={"architectures": ["NEOVisionModel"]},
            llm_config={
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_hidden_layers": 1,
                "vocab_size": 64,
            },
        )
        self.assertIsInstance(auto_config, NEOChatConfig)

        model_cls, arch = ModelRegistry.resolve_model_cls(["NEOChatModel"])
        self.assertIs(model_cls, NEOChatModel)
        self.assertEqual(arch, "NEOChatModel")
        self.assertTrue(is_multimodal_model(["NEOChatModel"]))

    def test_u1_vlm_position_helpers_match_block_semantics(self):
        input_ids = torch.tensor(
            [101, 151670, 151669, 151669, 151669, 151669, 151671, 102],
            dtype=torch.long,
        )

        indexes = build_u1_vlm_thw_indexes(
            input_ids,
            grid_hw=torch.tensor([[4, 4]], dtype=torch.long),
            downsample_ratio=0.5,
        )
        info = build_u1_vlm_input_info(
            [151670, 151669, 151669, 151669, 151669, 151671],
            grid_hw=[[4, 4]],
        )
        mask = build_u1_block_causal_allowed_mask(
            torch.tensor([0, 1, 2, 2, 2, 2, 3], dtype=torch.long)
        )

        self.assertEqual(indexes.shape, (3, 8))
        self.assertEqual(indexes[0].tolist(), [0, 1, 2, 2, 2, 2, 3, 4])
        self.assertEqual(info.image_context_token_count, 4)
        self.assertTrue(mask[2, 5])
        self.assertTrue(mask[5, 2])
        self.assertFalse(mask[1, 2])
        self.assertTrue(mask[6, 5])

    def test_u1_language_weight_mapper_routes_u_and_g_paths(self):
        self.assertEqual(
            map_u1_language_model_weight_name(
                "language_model.model.layers.0.self_attn.q_proj.weight"
            ),
            "model.layers.0.self_attn.q_proj.weight",
        )
        self.assertEqual(
            map_u1_language_model_weight_name(
                "model.language_model.layers.0.self_attn.q_proj.weight"
            ),
            "model.language_model.layers.0.self_attn.q_proj.weight",
        )
        self.assertEqual(
            map_u1_language_model_weight_name(
                "model.language_model.diffusion_model.layers.0.self_attn.q_proj.weight"
            ),
            "model.language_model.diffusion_model.layers.0.self_attn.q_proj.weight",
        )

    def test_u1_adapter_declares_pixel_flow_without_bagel_latent_api(self):
        adapter = U1UGModelAdapter()

        self.assertEqual(adapter.g_kind, "pixel_flow")
        self.assertFalse(hasattr(adapter, "predict_velocity_from_session"))
        self.assertFalse(hasattr(adapter, "decode_latents_to_image"))

    def test_u1_prompt_builders_keep_generation_and_vlm_shapes(self):
        self.assertIn("<img>", build_u1_t2i_prompt(prompt="draw a cup"))
        self.assertEqual(build_u1_t2i_uncondition_prompt().count("<img>"), 1)
        self.assertIn("<image>", build_u1_vlm_prompt(question="what is here?"))

    def test_u1_pixel_flow_executor_helpers(self):
        sampling = UGSamplingParams(prompt="draw", height=8, width=8)

        self.assertEqual(U1PixelFlowGSegmentExecutor.required_g_kind, "pixel_flow")
        self.assertEqual(_u1_patch_grid(height=33, width=65, patch_size=16), (3, 5))

        timesteps = _u1_timesteps(num_inference_steps=4, timestep_shift=2.0)
        self.assertEqual(len(timesteps), 4)
        self.assertGreater(timesteps[0], timesteps[-1])

        self.assertEqual(_u1_guidance_branch(sampling), "none")
        self.assertEqual(
            _u1_guidance_branch(
                SimpleNamespace(cfg_text_scale=2.0, cfg_img_scale=1.0)
            ),
            "text",
        )
        self.assertEqual(
            _u1_guidance_branch(
                SimpleNamespace(cfg_text_scale=1.0, cfg_img_scale=2.0)
            ),
            "image",
        )
        self.assertEqual(
            _u1_guidance_branch(
                SimpleNamespace(cfg_text_scale=2.0, cfg_img_scale=2.0)
            ),
            "text_image",
        )

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


if __name__ == "__main__":
    unittest.main()
