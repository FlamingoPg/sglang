# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.pipelines.ug import _load_ug_bridge
from sglang.srt.ug.adapter import UGModelRunnerAdapter, UGModelSessionView
from sglang.srt.ug.bagel import (
    BAGELAdapterError,
    BAGELDenoiseStepError,
    BAGELDenoiseStepRunner,
    BAGELInterleaveContextBackend,
    BAGELPreparedDenoise,
    BAGELUGModelAdapter,
    MockBAGELBackend,
    create_bagel_ug_model_adapter,
)
from sglang.srt.ug.context import UGSessionHandle
from sglang.srt.ug.runtime import (
    UGInterleavedMessage,
    UGSegmentState,
    UGSessionRuntime,
    UGVelocityRequest,
)


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


class FakeOfficialBAGELModel:
    def __init__(self):
        self.forward_flow_calls = []

    def _forward_flow(self, **kwargs):
        self.forward_flow_calls.append(kwargs)
        return torch.full_like(
            kwargs["x_t"],
            kwargs["cfg_text_scale"] + kwargs["cfg_img_scale"],
        )


class FakeContextBAGELModel(FakeOfficialBAGELModel):
    def __init__(self):
        super().__init__()
        self.prepare_vae_latent_calls = []
        self.prepare_vae_latent_cfg_calls = []

    def prepare_vae_latent(
        self,
        *,
        curr_kvlens,
        curr_rope,
        image_sizes,
        new_token_ids,
    ):
        self.prepare_vae_latent_calls.append(
            {
                "curr_kvlens": list(curr_kvlens),
                "curr_rope": list(curr_rope),
                "image_sizes": list(image_sizes),
                "new_token_ids": dict(new_token_ids),
            }
        )
        payload = dict(_fake_bagel_prepared().generation_input)
        payload["key_values_lens"] = torch.tensor(curr_kvlens, dtype=torch.int)
        return payload

    def prepare_vae_latent_cfg(self, *, curr_kvlens, curr_rope, image_sizes):
        self.prepare_vae_latent_cfg_calls.append(
            {
                "curr_kvlens": list(curr_kvlens),
                "curr_rope": list(curr_rope),
                "image_sizes": list(image_sizes),
            }
        )
        payload = dict(_fake_bagel_prepared().cfg_text_generation_input)
        payload["cfg_key_values_lens"] = torch.tensor(curr_kvlens, dtype=torch.int)
        return payload


class FakeImage:
    def __init__(self, size=(16, 8)):
        self.size = size


class FakeBAGELImageTransform:
    def __init__(self):
        self.resize_calls = []

    def resize_transform(self, image):
        self.resize_calls.append(image)
        return image


class FakeBAGELInferencer:
    def __init__(self):
        self.model = FakeContextBAGELModel()
        self.new_token_ids = {"start_of_image": 1, "end_of_image": 2}
        self.vae_transform = FakeBAGELImageTransform()
        self.events = []

    def init_gen_context(self):
        self.events.append(("init",))
        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": {"id": "ctx0"},
        }

    def update_context_text(self, text, gen_context):
        self.events.append(("text", text, gen_context["past_key_values"]["id"]))
        return {
            "kv_lens": [gen_context["kv_lens"][0] + len(text.split())],
            "ropes": [gen_context["ropes"][0] + 1],
            "past_key_values": {
                "id": f"{gen_context['past_key_values']['id']}:t{text}"
            },
        }

    def update_context_image(self, image, gen_context, vae=True, vit=True):
        self.events.append(
            ("image", image, vae, vit, gen_context["past_key_values"]["id"])
        )
        return {
            "kv_lens": [gen_context["kv_lens"][0] + 2],
            "ropes": [gen_context["ropes"][0] + 1],
            "past_key_values": {"id": f"{gen_context['past_key_values']['id']}:i"},
        }

    def gen_text(self, gen_context, **kwargs):
        self.events.append(("gen_text", gen_context["past_key_values"]["id"], kwargs))
        return "context_backend_text_after_image"


def _fake_bagel_prepared() -> BAGELPreparedDenoise:
    generation_input = {
        "packed_text_ids": torch.tensor([1, 2]),
        "packed_text_indexes": torch.tensor([0, 3]),
        "packed_vae_token_indexes": torch.tensor([1, 2]),
        "packed_vae_position_ids": torch.tensor([7, 8]),
        "packed_seqlens": torch.tensor([4], dtype=torch.int),
        "packed_position_ids": torch.tensor([0, 0, 0, 0]),
        "packed_indexes": torch.tensor([0, 1, 2, 3]),
        "key_values_lens": torch.tensor([5], dtype=torch.int),
        "packed_key_value_indexes": torch.tensor([0, 1, 2, 3, 4]),
    }
    cfg_generation_input = {
        "cfg_packed_position_ids": torch.tensor([0, 0, 0, 0]),
        "cfg_packed_query_indexes": torch.tensor([5, 6, 7, 8]),
        "cfg_key_values_lens": torch.tensor([5], dtype=torch.int),
        "cfg_packed_key_value_indexes": torch.tensor([0, 1, 2, 3, 4]),
    }
    return BAGELPreparedDenoise(
        generation_input=generation_input,
        cfg_text_generation_input=cfg_generation_input,
        cfg_img_generation_input=cfg_generation_input,
        past_key_values=object(),
        cfg_text_past_key_values=object(),
        cfg_img_past_key_values=object(),
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_interval=(0.4, 1.0),
    )


class TestBAGELDenoiseStepRunner(unittest.TestCase):
    def test_predict_velocity_calls_official_forward_flow_once(self):
        model = FakeOfficialBAGELModel()
        runner = BAGELDenoiseStepRunner()
        prepared = _fake_bagel_prepared()
        latents = torch.zeros(2, 3)

        velocity = runner.predict_velocity(
            model=model,
            prepared=prepared,
            latent_tokens=latents,
            timestep=torch.tensor([0.5]),
        )

        self.assertEqual(len(model.forward_flow_calls), 1)
        call = model.forward_flow_calls[0]
        self.assertIs(call["past_key_values"], prepared.past_key_values)
        self.assertIs(
            call["cfg_text_past_key_values"],
            prepared.cfg_text_past_key_values,
        )
        self.assertIs(
            call["cfg_img_past_key_values"],
            prepared.cfg_img_past_key_values,
        )
        self.assertEqual(call["cfg_text_scale"], 4.0)
        self.assertEqual(call["cfg_img_scale"], 1.5)
        self.assertEqual(tuple(call["timestep"].shape), (2,))
        self.assertTrue(torch.equal(call["packed_text_ids"], torch.tensor([1, 2])))
        self.assertTrue(torch.allclose(velocity, torch.full_like(latents, 5.5)))

    def test_predict_velocity_disables_cfg_outside_interval(self):
        model = FakeOfficialBAGELModel()
        runner = BAGELDenoiseStepRunner()

        velocity = runner.predict_velocity(
            model=model,
            prepared=_fake_bagel_prepared(),
            latent_tokens=torch.zeros(1, 3),
            timestep=torch.tensor([0.2]),
        )

        call = model.forward_flow_calls[0]
        self.assertEqual(call["cfg_text_scale"], 1.0)
        self.assertEqual(call["cfg_img_scale"], 1.0)
        self.assertTrue(torch.allclose(velocity, torch.full_like(velocity, 2.0)))

    def test_build_timesteps_matches_bagel_loop_shape(self):
        timesteps, dts = BAGELDenoiseStepRunner.build_timesteps(
            num_timesteps=4,
            timestep_shift=3.0,
            device="cpu",
        )

        self.assertEqual(tuple(timesteps.shape), (3,))
        self.assertEqual(tuple(dts.shape), (3,))
        self.assertTrue(torch.all(timesteps[:-1] > timesteps[1:]))

    def test_missing_generation_input_key_fails_fast(self):
        model = FakeOfficialBAGELModel()
        runner = BAGELDenoiseStepRunner()
        prepared = _fake_bagel_prepared()
        del prepared.generation_input["packed_text_ids"]

        with self.assertRaisesRegex(BAGELDenoiseStepError, "generation_input"):
            runner.predict_velocity(
                model=model,
                prepared=prepared,
                latent_tokens=torch.zeros(1, 3),
                timestep=torch.tensor([0.5]),
            )


class TestBAGELInterleaveContextBackend(unittest.TestCase):
    def test_context_backend_runs_u_g_u_with_single_prepare(self):
        inferencer = FakeBAGELInferencer()
        adapter = BAGELUGModelAdapter(
            "already-loaded-bagel",
            backend=BAGELInterleaveContextBackend(
                inferencer,
                default_image_shape=(32, 32),
            ),
        )
        runtime = UGSessionRuntime(model_runner=UGModelRunnerAdapter(adapter))
        image = FakeImage(size=(16, 8))

        handle = runtime.prefill_interleaved(
            [
                UGInterleavedMessage(type="image", content=image),
                UGInterleavedMessage(type="text", content="draw a calm lake"),
            ],
            session_id="bagel-context-session",
        )
        marker = runtime.decode_next_segment(handle)
        self.assertEqual(marker.type, "image_marker")

        sampling_params = SimpleNamespace(
            height=64,
            width=32,
            cfg_text_scale=5.0,
            cfg_img_scale=2.0,
            cfg_interval=(0.4, 1.0),
            cfg_renorm_min=0.1,
            cfg_renorm_type="channel",
        )
        latents = torch.zeros(2, 3)
        response = runtime.predict_velocity(
            UGVelocityRequest(
                session=handle,
                latent_tokens=latents,
                timestep=torch.tensor([0.5]),
                latent_position_ids=torch.arange(3),
                sampling_params=sampling_params,
            )
        )
        response = runtime.predict_velocity(
            UGVelocityRequest(
                session=response.session,
                latent_tokens=latents,
                timestep=torch.tensor([0.45]),
                latent_position_ids=torch.arange(3),
                sampling_params=sampling_params,
            )
        )

        self.assertTrue(
            torch.allclose(response.velocity, torch.full_like(latents, 7.0))
        )
        self.assertEqual(len(inferencer.model.prepare_vae_latent_calls), 1)
        self.assertEqual(len(inferencer.model.prepare_vae_latent_cfg_calls), 2)
        self.assertEqual(len(inferencer.model.forward_flow_calls), 2)
        self.assertEqual(
            inferencer.model.prepare_vae_latent_calls[0]["image_sizes"],
            [(64, 32)],
        )
        flow_call = inferencer.model.forward_flow_calls[0]
        self.assertEqual(flow_call["cfg_text_scale"], 5.0)
        self.assertEqual(flow_call["cfg_img_scale"], 2.0)
        self.assertEqual(flow_call["cfg_renorm_min"], 0.1)
        self.assertEqual(flow_call["cfg_renorm_type"], "channel")
        self.assertTrue(torch.equal(flow_call["key_values_lens"], torch.tensor([6])))

        generated_image = FakeImage()
        handle = runtime.append_generated_image(response.session, image=generated_image)
        text = runtime.decode_next_segment(handle)

        self.assertEqual(text.type, "text")
        self.assertEqual(text.text, "context_backend_text_after_image")
        self.assertEqual(
            inferencer.vae_transform.resize_calls, [image, generated_image]
        )
        self.assertEqual(runtime.get_debug_counters(handle)["prefill_count"], 1)
        self.assertEqual(runtime.get_debug_counters(handle)["velocity_count"], 2)
        self.assertEqual(runtime.get_debug_counters(handle)["append_image_count"], 1)

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="now draw a boat")],
            session_id="bagel-context-session",
        )
        marker = runtime.decode_next_segment(handle)
        self.assertEqual(marker.type, "image_marker")
        self.assertEqual(handle.session_id, response.session.session_id)
        self.assertEqual(runtime.get_debug_counters(handle)["prefill_count"], 2)

        runtime.close_session(handle)
        self.assertNotIn("bagel-context-session", adapter.backend.sessions)

    def test_context_backend_release_closes_backend_session(self):
        inferencer = FakeBAGELInferencer()
        backend = BAGELInterleaveContextBackend(inferencer)
        adapter = BAGELUGModelAdapter("already-loaded-bagel", backend=backend)
        runtime = UGSessionRuntime(model_runner=UGModelRunnerAdapter(adapter))
        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="draw a cat")],
            session_id="bagel-close-session",
        )
        self.assertIn("bagel-close-session", backend.sessions)

        runtime.close_session(handle)

        self.assertNotIn("bagel-close-session", backend.sessions)


if __name__ == "__main__":
    unittest.main()
