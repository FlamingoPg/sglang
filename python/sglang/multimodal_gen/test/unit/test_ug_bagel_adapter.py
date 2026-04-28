# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest

import torch

from sglang.multimodal_gen.runtime.pipelines.ug import _load_ug_bridge
from sglang.srt.ug.adapter import UGModelSessionView
from sglang.srt.ug.bagel import (
    BAGELAdapterError,
    BAGELUGModelAdapter,
    MockBAGELBackend,
    create_bagel_ug_model_adapter,
)
from sglang.srt.ug.context import UGSessionHandle
from sglang.srt.ug.runtime import UGSegmentState, UGVelocityRequest


class TestBAGELUGModelAdapter(unittest.TestCase):
    def test_missing_checkpoint_path_reports_actionable_error(self):
        with self.assertRaisesRegex(
            BAGELAdapterError,
            "requires a local BAGEL checkpoint directory",
        ):
            BAGELUGModelAdapter("ByteDance-Seed/BAGEL-7B-MoT")

    def test_missing_checkpoint_files_reports_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                BAGELAdapterError,
                "missing required files",
            ):
                BAGELUGModelAdapter(tmpdir)

    def test_mock_bagel_adapter_factory_runs_u_g_u_loop(self):
        adapter = create_bagel_ug_model_adapter("sglang-internal/mock-bagel")
        self.assertIsInstance(adapter.backend, MockBAGELBackend)

        bridge = _load_ug_bridge("sglang-internal/mock-bagel")
        contexts = bridge.build_contexts(prompt="draw then explain", image=None)
        self.assertIsInstance(contexts.full.session, UGSessionHandle)

        latents = torch.zeros(1, 2, 4)
        for step in range(2):
            latents = bridge.predict_velocity(
                contexts=contexts,
                latent_tokens=latents,
                timestep=torch.tensor([1.0 - step * 0.5]),
                latent_position_ids=torch.arange(2),
                sampling_params=None,
            )

        bridge.append_generated_image(contexts=contexts, image=object())
        post_image = bridge.decode_next_segment(contexts=contexts)

        self.assertEqual(post_image.type, "text")
        self.assertEqual(post_image.text, "bagel_mock_text_after_image")
        self.assertEqual(contexts.full.token_count, 5)

        counters = bridge.runtime.get_debug_counters(contexts.full.session)
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["velocity_count"], 2)
        self.assertEqual(counters["append_image_count"], 1)
        self.assertEqual(counters["decode_count"], 2)
        self.assertEqual(counters["state"], "u_decode")

    def test_mock_bagel_velocity_depends_on_srt_session_view(self):
        adapter = create_bagel_ug_model_adapter("sglang-internal/mock-bagel")
        session = UGModelSessionView(
            handle=UGSessionHandle(
                session_id="bagel-view",
                anchor_request_id="bagel-view:u1",
                context_length=3,
                context_version=1,
            ),
            state=UGSegmentState.G_DENOISE,
            srt_request_count=3,
            srt_last_request_id="bagel-view:u1",
            srt_last_origin_input_len=3,
        )
        request = UGVelocityRequest(
            session=session.handle,
            latent_tokens=torch.zeros(1, 1, 2),
            timestep=torch.tensor([0.5]),
            latent_position_ids=torch.arange(1),
            sampling_params=None,
        )

        velocity = adapter.predict_velocity_from_session(
            session=session,
            request=request,
        )

        self.assertTrue(torch.allclose(velocity, torch.full_like(velocity, 1.15)))


if __name__ == "__main__":
    unittest.main()
