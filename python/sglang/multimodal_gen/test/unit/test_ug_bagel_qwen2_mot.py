# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from sglang.srt.models.bagel_qwen2_mot import (
    BAGELMoTTokenRouting,
    BAGELQwen2MoTForCausalLM,
    _iter_bagel_language_model_weights,
    _normalize_bagel_rope_scaling,
)
from sglang.srt.models.registry import ModelRegistry


class TestBAGELQwen2MoTModel(unittest.TestCase):
    def test_filters_bagel_checkpoint_to_language_model_weights(self):
        weights = [
            ("language_model.model.embed_tokens.weight", "embed"),
            (
                "language_model.model.layers.0.self_attn.q_proj_moe_gen.weight",
                "mot_q",
            ),
            ("language_model.model.layers.0.mlp_moe_gen.gate_proj.weight", "mot_mlp"),
            ("time_embedder.mlp.0.weight", "time"),
            ("vae2llm.weight", "vae2llm"),
            ("llm2vae.weight", "llm2vae"),
            ("latent_pos_embed.pos_embed", "latent_pos"),
            ("vit_model.vision_model.embeddings.patch_embedding.weight", "vit"),
            ("decoder.conv_in.weight", "vae"),
            ("connector.fc1.weight", "outer"),
            ("model.layers.0.self_attn.q_proj.weight", "plain_qwen"),
        ]

        filtered = list(_iter_bagel_language_model_weights(weights))

        self.assertEqual(
            filtered,
            [
                ("model.embed_tokens.weight", "embed"),
                ("model.layers.0.self_attn.q_proj_moe_gen.weight", "mot_q"),
                ("model.layers.0.mlp_moe_gen.gate_proj.weight", "mot_mlp"),
                ("time_embedder.mlp.0.weight", "time"),
                ("vae2llm.weight", "vae2llm"),
                ("llm2vae.weight", "llm2vae"),
                ("latent_pos_embed.pos_embed", "latent_pos"),
                ("model.layers.0.self_attn.q_proj.weight", "plain_qwen"),
            ],
        )

    def test_model_registry_sees_bagel_qwen2_mot_architecture(self):
        model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
            ["BAGELQwen2MoTForCausalLM"]
        )

        self.assertIs(model_cls, BAGELQwen2MoTForCausalLM)
        self.assertEqual(resolved_arch, "BAGELQwen2MoTForCausalLM")

    def test_default_rope_parameters_are_not_treated_as_scaling(self):
        self.assertIsNone(
            _normalize_bagel_rope_scaling(
                {"rope_theta": 1000000.0, "rope_type": "default"}
            )
        )
        scaled = {"rope_theta": 1000000.0, "rope_type": "linear", "factor": 2.0}
        self.assertIs(_normalize_bagel_rope_scaling(scaled), scaled)

    def test_mot_token_routing_rejects_invalid_indices(self):
        valid = BAGELMoTTokenRouting(
            text_token_indices=torch.tensor([0, 2]),
            vae_token_indices=torch.tensor([1]),
        )
        valid.validate(total_tokens=3)

        with self.assertRaisesRegex(ValueError, "cover each input token"):
            BAGELMoTTokenRouting(
                text_token_indices=torch.tensor([0]),
                vae_token_indices=torch.tensor([1]),
            ).validate(total_tokens=3)

        with self.assertRaisesRegex(ValueError, "disjoint"):
            BAGELMoTTokenRouting(
                text_token_indices=torch.tensor([0, 1]),
                vae_token_indices=torch.tensor([1]),
            ).validate(total_tokens=3)

        with self.assertRaisesRegex(ValueError, "out of range"):
            BAGELMoTTokenRouting(
                text_token_indices=torch.tensor([0, 3]),
                vae_token_indices=torch.tensor([1]),
            ).validate(total_tokens=3)


if __name__ == "__main__":
    unittest.main()
