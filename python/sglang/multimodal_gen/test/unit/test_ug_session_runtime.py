# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import fields
from types import SimpleNamespace

import torch

from sglang.srt.ug.context import UGSessionHandle
from sglang.srt.ug.runtime import (
    FakeUGModelRunner,
    UGInterleavedMessage,
    UGSegmentState,
    UGSessionRuntime,
    UGVelocityRequest,
)


class TestUGSessionRuntime(unittest.TestCase):
    def test_handle_does_not_expose_kv_allocator_details(self):
        names = {field.name for field in fields(UGSessionHandle)}

        self.assertEqual(
            names,
            {"session_id", "anchor_request_id", "context_length", "context_version"},
        )
        self.assertFalse(any("kv" in name.lower() for name in names))
        self.assertFalse(any("slot" in name.lower() for name in names))
        self.assertFalse(any("page" in name.lower() for name in names))

    def test_runtime_uses_existing_srt_session_lifecycle(self):
        class FakeSessionController:
            def __init__(self):
                self.opened = []
                self.closed = []
                self.sessions = set()

            def __contains__(self, session_id):
                return session_id in self.sessions

            def open(self, req):
                self.opened.append(req.session_id)
                self.sessions.add(req.session_id)
                return SimpleNamespace(success=True)

            def close(self, req):
                self.closed.append(req.session_id)
                self.sessions.remove(req.session_id)

        controller = FakeSessionController()
        runtime = UGSessionRuntime(
            model_runner=FakeUGModelRunner(), session_controller=controller
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="hello")],
            session_id="srt-session",
        )
        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="again")],
            session_id=handle.session_id,
        )

        self.assertEqual(handle.session_id, "srt-session")
        self.assertEqual(controller.opened, ["srt-session"])
        self.assertEqual(runtime.get_debug_counters(handle)["prefill_count"], 2)

        runtime.close_session(handle)
        self.assertEqual(controller.closed, ["srt-session"])

    def test_u_g_u_minimal_loop_keeps_one_session(self):
        runtime = UGSessionRuntime(model_runner=FakeUGModelRunner())
        events = ["input_text", "input_image"]

        handle = runtime.prefill_interleaved(
            [
                UGInterleavedMessage(type="image", content=object()),
                UGInterleavedMessage(type="text", content="draw then explain"),
            ],
            session_id="ug-test-session",
        )
        self.assertEqual(runtime.get_state(handle), UGSegmentState.U_DECODE)

        decode = runtime.decode_next_segment(handle)
        self.assertEqual(decode.type, "image_marker")
        self.assertEqual(runtime.get_state(handle), UGSegmentState.G_DENOISE)

        latents = torch.zeros(1, 4, 8)
        for step in range(3):
            response = runtime.predict_velocity(
                UGVelocityRequest(
                    session=handle,
                    latent_tokens=latents,
                    timestep=torch.tensor([1.0 - step * 0.25]),
                    latent_position_ids=torch.arange(4),
                    sampling_params=None,
                )
            )
            latents = response.velocity

        events.append("generated_image")
        handle_after_image = runtime.append_generated_image(handle, image=object())
        self.assertEqual(handle_after_image.session_id, handle.session_id)
        self.assertGreater(handle_after_image.context_version, handle.context_version)
        self.assertEqual(runtime.get_state(handle_after_image), UGSegmentState.U_DECODE)

        decode_after_image = runtime.decode_next_segment(handle_after_image)
        self.assertEqual(decode_after_image.type, "text")
        self.assertEqual(decode_after_image.text, "generated_text_after_image")
        events.append("generated_text_after_image")

        self.assertEqual(
            events,
            [
                "input_text",
                "input_image",
                "generated_image",
                "generated_text_after_image",
            ],
        )

        counters = runtime.get_debug_counters(handle_after_image)
        self.assertEqual(counters["session_id"], "ug-test-session")
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["velocity_count"], 3)
        self.assertEqual(counters["append_image_count"], 1)

    def test_illegal_transitions_fail_early(self):
        runtime = UGSessionRuntime(model_runner=FakeUGModelRunner())
        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="hello")],
            session_id="illegal",
        )

        with self.assertRaisesRegex(ValueError, "Cannot predict UG velocity"):
            runtime.predict_velocity(
                UGVelocityRequest(
                    session=handle,
                    latent_tokens=torch.zeros(1, 1, 1),
                    timestep=torch.tensor([1.0]),
                    latent_position_ids=torch.arange(1),
                    sampling_params=None,
                )
            )

        with self.assertRaisesRegex(ValueError, "Cannot append generated image"):
            runtime.append_generated_image(handle, image=object())

        runtime.decode_next_segment(handle)
        handle_after_image = runtime.append_generated_image(handle, image=object())
        with self.assertRaisesRegex(ValueError, "Stale UG session handle"):
            runtime.decode_next_segment(handle)

        self.assertEqual(runtime.get_state(handle_after_image), UGSegmentState.U_DECODE)


if __name__ == "__main__":
    unittest.main()
