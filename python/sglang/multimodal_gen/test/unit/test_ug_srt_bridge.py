# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
from PIL import Image

from sglang.srt.ug.denoiser import FakeUGDenoiserBridge


class TestFakeUGDenoiserBridge(unittest.TestCase):
    def test_build_contexts_splits_full_text_and_image_cfg(self):
        bridge = FakeUGDenoiserBridge()
        image = Image.new("RGB", (8, 8), color="white")

        contexts = bridge.build_contexts(prompt="a small cat", image=image)

        self.assertEqual(contexts.full.request_id, "full")
        self.assertEqual(contexts.full.token_count, 5)
        self.assertEqual(contexts.text_cfg.token_count, 2)
        self.assertEqual(contexts.image_cfg.token_count, 3)

    def test_predict_velocity_depends_on_full_context(self):
        bridge = FakeUGDenoiserBridge()
        contexts = bridge.build_contexts(prompt="hello world", image=None)
        latents = torch.zeros(1, 2, 4)
        timestep = torch.tensor([0.5])

        velocity = bridge.predict_velocity(
            contexts=contexts,
            latent_tokens=latents,
            timestep=timestep,
            latent_position_ids=torch.arange(2),
            sampling_params=None,
        )

        self.assertTrue(torch.allclose(velocity, torch.full_like(latents, 0.51)))


if __name__ == "__main__":
    unittest.main()
