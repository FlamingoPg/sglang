# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from sglang.srt.ug.adapter import (
    UGModelAdapterProtocol,
    UGModelAppendImageResult,
    UGModelPrefillResult,
)
from sglang.srt.ug.runtime import (
    UGDecodeResult,
    UGInterleavedMessage,
    UGVelocityRequest,
)

_BAGEL_REQUIRED_CHECKPOINT_FILES = (
    "llm_config.json",
    "vit_config.json",
    "ae.safetensors",
    "ema.safetensors",
)
_BAGEL_REQUIRED_MODULES = (
    "inferencer",
    "modeling.bagel",
    "data.transforms",
)
_BAGEL_GENERATION_INPUT_KEYS = (
    "packed_text_ids",
    "packed_text_indexes",
    "packed_vae_token_indexes",
    "packed_vae_position_ids",
    "packed_seqlens",
    "packed_position_ids",
    "packed_indexes",
    "key_values_lens",
    "packed_key_value_indexes",
)
_BAGEL_CFG_TEXT_INPUT_KEYS = (
    "cfg_packed_position_ids",
    "cfg_packed_query_indexes",
    "cfg_key_values_lens",
    "cfg_packed_key_value_indexes",
)
_BAGEL_CFG_IMG_INPUT_KEYS = _BAGEL_CFG_TEXT_INPUT_KEYS


class BAGELAdapterError(RuntimeError):
    """Raised when the BAGEL adapter cannot be constructed safely."""


class BAGELDenoiseStepError(RuntimeError):
    """Raised when a BAGEL single-step denoise call is malformed."""


class BAGELPreparedDenoise:
    """Official BAGEL denoise inputs prepared from a SRT-owned UG session."""

    def __init__(
        self,
        *,
        generation_input: dict[str, Any],
        cfg_text_generation_input: dict[str, Any],
        cfg_img_generation_input: dict[str, Any],
        past_key_values: Any,
        cfg_text_past_key_values: Any | None = None,
        cfg_img_past_key_values: Any | None = None,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_type: str = "parallel",
    ) -> None:
        self.generation_input = generation_input
        self.cfg_text_generation_input = cfg_text_generation_input
        self.cfg_img_generation_input = cfg_img_generation_input
        self.past_key_values = past_key_values
        self.cfg_text_past_key_values = cfg_text_past_key_values
        self.cfg_img_past_key_values = cfg_img_past_key_values
        self.cfg_text_scale = cfg_text_scale
        self.cfg_img_scale = cfg_img_scale
        self.cfg_interval = cfg_interval
        self.cfg_renorm_min = cfg_renorm_min
        self.cfg_renorm_type = cfg_renorm_type
        self.cfg_type = cfg_type


class BAGELDenoiseStepRunner:
    """Runs the single `_forward_flow` step extracted from BAGEL.generate_image."""

    def predict_velocity(
        self,
        *,
        model: Any,
        prepared: BAGELPreparedDenoise,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_prepared(prepared)
        timestep = self._expand_timestep(timestep, latent_tokens)
        cfg_text_scale, cfg_img_scale = self._effective_cfg_scales(prepared, timestep)
        generation_input = prepared.generation_input
        cfg_text_input = prepared.cfg_text_generation_input
        cfg_img_input = prepared.cfg_img_generation_input

        return model._forward_flow(
            x_t=latent_tokens,
            timestep=timestep,
            packed_vae_token_indexes=generation_input["packed_vae_token_indexes"],
            packed_vae_position_ids=generation_input["packed_vae_position_ids"],
            packed_text_ids=generation_input["packed_text_ids"],
            packed_text_indexes=generation_input["packed_text_indexes"],
            packed_position_ids=generation_input["packed_position_ids"],
            packed_indexes=generation_input["packed_indexes"],
            packed_seqlens=generation_input["packed_seqlens"],
            key_values_lens=generation_input["key_values_lens"],
            past_key_values=prepared.past_key_values,
            packed_key_value_indexes=generation_input["packed_key_value_indexes"],
            cfg_renorm_min=prepared.cfg_renorm_min,
            cfg_renorm_type=prepared.cfg_renorm_type,
            cfg_text_scale=cfg_text_scale,
            cfg_text_packed_position_ids=cfg_text_input["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=cfg_text_input["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=cfg_text_input["cfg_key_values_lens"],
            cfg_text_past_key_values=prepared.cfg_text_past_key_values,
            cfg_text_packed_key_value_indexes=cfg_text_input[
                "cfg_packed_key_value_indexes"
            ],
            cfg_img_scale=cfg_img_scale,
            cfg_img_packed_position_ids=cfg_img_input["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=cfg_img_input["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=cfg_img_input["cfg_key_values_lens"],
            cfg_img_past_key_values=prepared.cfg_img_past_key_values,
            cfg_img_packed_key_value_indexes=cfg_img_input[
                "cfg_packed_key_value_indexes"
            ],
            cfg_type=prepared.cfg_type,
        )

    @staticmethod
    def build_timesteps(
        *,
        num_timesteps: int,
        timestep_shift: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_timesteps <= 1:
            raise BAGELDenoiseStepError(
                f"num_timesteps must be > 1, got {num_timesteps}"
            )
        timesteps = torch.linspace(1, 0, num_timesteps, device=device, dtype=dtype)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        return timesteps[:-1], dts

    @staticmethod
    def _effective_cfg_scales(
        prepared: BAGELPreparedDenoise,
        timestep: torch.Tensor,
    ) -> tuple[float, float]:
        t = float(timestep.flatten()[0].detach().cpu())
        start, end = prepared.cfg_interval
        if t > start and t <= end:
            return prepared.cfg_text_scale, prepared.cfg_img_scale
        return 1.0, 1.0

    @staticmethod
    def _expand_timestep(
        timestep: torch.Tensor,
        latent_tokens: torch.Tensor,
    ) -> torch.Tensor:
        timestep = timestep.to(device=latent_tokens.device)
        if timestep.numel() == 1:
            return timestep.reshape(1).expand(latent_tokens.shape[0])
        if timestep.shape[0] != latent_tokens.shape[0]:
            raise BAGELDenoiseStepError(
                "BAGEL timestep must be scalar or match latent token batch size: "
                f"{tuple(timestep.shape)} vs {tuple(latent_tokens.shape)}"
            )
        return timestep

    @staticmethod
    def _validate_prepared(prepared: BAGELPreparedDenoise) -> None:
        _require_keys(
            prepared.generation_input,
            _BAGEL_GENERATION_INPUT_KEYS,
            "generation_input",
        )
        _require_keys(
            prepared.cfg_text_generation_input,
            _BAGEL_CFG_TEXT_INPUT_KEYS,
            "cfg_text_generation_input",
        )
        _require_keys(
            prepared.cfg_img_generation_input,
            _BAGEL_CFG_IMG_INPUT_KEYS,
            "cfg_img_generation_input",
        )


@dataclass
class BAGELSessionContext:
    gen_context: dict[str, Any]
    cfg_text_context: dict[str, Any]
    cfg_img_context: dict[str, Any]
    image_shape: tuple[int, int]
    prepared_denoise: BAGELPreparedDenoise | None = None
    decode_count: int = 0
    append_image_count: int = 0


class BAGELInterleaveContextBackend:
    """Wraps an official BAGEL InterleaveInferencer behind UG adapter methods."""

    def __init__(
        self,
        inferencer: Any,
        *,
        step_runner: BAGELDenoiseStepRunner | None = None,
        default_image_shape: tuple[int, int] = (1024, 1024),
    ) -> None:
        self.inferencer = inferencer
        self.step_runner = step_runner or BAGELDenoiseStepRunner()
        self.default_image_shape = default_image_shape
        self.sessions: dict[str, BAGELSessionContext] = {}

    def prefill_interleaved(
        self, *, session, messages: list[UGInterleavedMessage]
    ) -> UGModelPrefillResult:
        state = self._state_for(session.handle.session_id)
        state.decode_count = 0
        added_tokens = 0
        for message in messages:
            if message.type == "text":
                text = str(message.content)
                state.cfg_text_context = _clone_context(state.gen_context)
                state.gen_context = self.inferencer.update_context_text(
                    text,
                    state.gen_context,
                )
                state.cfg_img_context = self.inferencer.update_context_text(
                    text,
                    state.cfg_img_context,
                )
                added_tokens += len(text.split())
            elif message.type == "image":
                image = self._prepare_image(message.content)
                state.gen_context = self.inferencer.update_context_image(
                    image,
                    state.gen_context,
                    vae=True,
                    vit=True,
                )
                state.image_shape = self._image_shape(image)
                state.cfg_text_context = _clone_context(state.gen_context)
                added_tokens += 2
            else:
                raise ValueError(f"Unsupported BAGEL message type: {message.type}")
            state.prepared_denoise = None
        return UGModelPrefillResult(added_tokens=added_tokens)

    def decode_next_segment(self, *, session) -> UGDecodeResult:
        state = self._state_for(session.handle.session_id)
        if state.decode_count == 0:
            state.decode_count += 1
            return UGDecodeResult(type="image_marker")
        if state.append_image_count > 0 and state.decode_count == 1:
            state.decode_count += 1
            text = self.inferencer.gen_text(
                state.gen_context,
                do_sample=False,
                temperature=0.3,
                max_length=512,
            )
            return UGDecodeResult(type="text", text=text)
        state.decode_count += 1
        return UGDecodeResult(type="done")

    def predict_velocity_from_session(
        self, *, session, request: UGVelocityRequest
    ) -> torch.Tensor:
        state = self._state_for(session.handle.session_id)
        if state.prepared_denoise is None:
            state.prepared_denoise = self._prepare_denoise(
                state, request.sampling_params
            )
        return self.step_runner.predict_velocity(
            model=self.inferencer.model,
            prepared=state.prepared_denoise,
            latent_tokens=request.latent_tokens,
            timestep=request.timestep,
        )

    def append_generated_image(
        self, *, session, image: Any | None
    ) -> UGModelAppendImageResult:
        state = self._state_for(session.handle.session_id)
        image = self._prepare_image(image)
        state.gen_context = self.inferencer.update_context_image(
            image,
            state.gen_context,
            vae=True,
            vit=True,
        )
        state.image_shape = self._image_shape(image)
        state.cfg_text_context = _clone_context(state.gen_context)
        state.append_image_count += 1
        state.prepared_denoise = None
        return UGModelAppendImageResult(added_tokens=2)

    def close_session(self, *, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def _state_for(self, session_id: str) -> BAGELSessionContext:
        state = self.sessions.get(session_id)
        if state is not None:
            return state
        gen_context = self.inferencer.init_gen_context()
        state = BAGELSessionContext(
            gen_context=gen_context,
            cfg_text_context=_clone_context(gen_context),
            cfg_img_context=_clone_context(gen_context),
            image_shape=self.default_image_shape,
        )
        self.sessions[session_id] = state
        return state

    def _prepare_denoise(
        self,
        state: BAGELSessionContext,
        sampling_params: Any | None,
    ) -> BAGELPreparedDenoise:
        image_shape = self._image_shape_from_params(sampling_params, state.image_shape)
        model = self.inferencer.model
        generation_input = model.prepare_vae_latent(
            curr_kvlens=state.gen_context["kv_lens"],
            curr_rope=state.gen_context["ropes"],
            image_sizes=[image_shape],
            new_token_ids=self.inferencer.new_token_ids,
        )
        cfg_text_generation_input = model.prepare_vae_latent_cfg(
            curr_kvlens=state.cfg_text_context["kv_lens"],
            curr_rope=state.cfg_text_context["ropes"],
            image_sizes=[image_shape],
        )
        cfg_img_generation_input = model.prepare_vae_latent_cfg(
            curr_kvlens=state.cfg_img_context["kv_lens"],
            curr_rope=state.cfg_img_context["ropes"],
            image_sizes=[image_shape],
        )
        return BAGELPreparedDenoise(
            generation_input=generation_input,
            cfg_text_generation_input=cfg_text_generation_input,
            cfg_img_generation_input=cfg_img_generation_input,
            past_key_values=state.gen_context["past_key_values"],
            cfg_text_past_key_values=state.cfg_text_context["past_key_values"],
            cfg_img_past_key_values=state.cfg_img_context["past_key_values"],
            cfg_text_scale=float(getattr(sampling_params, "cfg_text_scale", 4.0)),
            cfg_img_scale=float(getattr(sampling_params, "cfg_img_scale", 1.5)),
            cfg_interval=tuple(getattr(sampling_params, "cfg_interval", (0.4, 1.0))),
            cfg_renorm_min=float(getattr(sampling_params, "cfg_renorm_min", 0.0)),
            cfg_renorm_type=getattr(sampling_params, "cfg_renorm_type", "global"),
        )

    def _prepare_image(self, image: Any | None) -> Any | None:
        if image is None:
            return None
        transform = getattr(self.inferencer, "vae_transform", None)
        if transform is None:
            return image
        resize_transform = getattr(transform, "resize_transform", None)
        if resize_transform is None:
            return image
        return resize_transform(image)

    def _image_shape(self, image: Any | None) -> tuple[int, int]:
        size = getattr(image, "size", None)
        if isinstance(size, tuple) and len(size) == 2:
            width, height = size
            return int(height), int(width)
        return self.default_image_shape

    @staticmethod
    def _image_shape_from_params(
        sampling_params: Any | None,
        default: tuple[int, int],
    ) -> tuple[int, int]:
        if sampling_params is None:
            return default
        height = getattr(sampling_params, "height", None) or default[0]
        width = getattr(sampling_params, "width", None) or default[1]
        return int(height), int(width)


class BAGELBackendProtocol(Protocol):
    def prefill_interleaved(
        self, *, session, messages: list[UGInterleavedMessage]
    ) -> UGModelPrefillResult: ...

    def decode_next_segment(self, *, session) -> UGDecodeResult: ...

    def predict_velocity_from_session(
        self, *, session, request: UGVelocityRequest
    ) -> torch.Tensor: ...

    def append_generated_image(
        self, *, session, image: Any | None
    ) -> UGModelAppendImageResult: ...

    def close_session(self, *, session_id: str) -> None: ...


class BAGELUGModelAdapter(UGModelAdapterProtocol):
    """BAGEL-facing UG adapter shell.

    The real BAGEL backend is intentionally not loaded here yet. Official BAGEL
    exposes an interleaved inferencer whose image generation call owns the whole
    denoising loop; SGLang UG needs a per-step velocity hook first. Until that
    hook lands, tests and diffusion smoke use the mock backend below.
    """

    def __init__(
        self,
        model_path: str,
        *,
        backend: BAGELBackendProtocol | None = None,
    ) -> None:
        self.model_path = model_path
        self.backend = backend or self._load_real_backend(model_path)

    def prefill_interleaved(
        self, *, session, messages: list[UGInterleavedMessage]
    ) -> UGModelPrefillResult:
        return self.backend.prefill_interleaved(session=session, messages=messages)

    def decode_next_segment(self, *, session) -> UGDecodeResult:
        return self.backend.decode_next_segment(session=session)

    def predict_velocity_from_session(
        self, *, session, request: UGVelocityRequest
    ) -> torch.Tensor:
        return self.backend.predict_velocity_from_session(
            session=session,
            request=request,
        )

    def append_generated_image(
        self, *, session, image: Any | None
    ) -> UGModelAppendImageResult:
        return self.backend.append_generated_image(session=session, image=image)

    def close_session(self, *, session_id: str) -> None:
        self.backend.close_session(session_id=session_id)

    @staticmethod
    def _load_real_backend(model_path: str) -> BAGELBackendProtocol:
        checkpoint_dir = Path(model_path).expanduser()
        if not checkpoint_dir.exists():
            raise BAGELAdapterError(
                "BAGELUGModelAdapter requires a local BAGEL checkpoint directory. "
                "Download ByteDance-Seed/BAGEL-7B-MoT first, then pass the local "
                "directory path; use sglang-internal/mock-bagel for adapter smoke "
                "tests."
            )
        missing_files = [
            name
            for name in _BAGEL_REQUIRED_CHECKPOINT_FILES
            if not (checkpoint_dir / name).exists()
        ]
        if missing_files:
            raise BAGELAdapterError(
                "BAGEL checkpoint is missing required files: "
                f"{missing_files}. Expected a local ByteDance-Seed/BAGEL-7B-MoT "
                "checkout with the official config and weight files."
            )

        missing_modules = [
            name for name in _BAGEL_REQUIRED_MODULES if _find_spec(name) is None
        ]
        if missing_modules:
            raise BAGELAdapterError(
                "BAGEL Python modules are not importable: "
                f"{missing_modules}. Add the official ByteDance-Seed/BAGEL repo "
                "to PYTHONPATH or vendor the required model code before enabling "
                "the real BAGEL backend."
            )

        raise BAGELAdapterError(
            "Real BAGEL checkpoint construction is not wired yet. "
            "BAGELInterleaveContextBackend can wrap an already loaded official "
            "InterleaveInferencer, but the loader still needs to construct that "
            "inferencer from checkpoint files inside SRT."
        )


class MockBAGELBackend:
    """Deterministic BAGEL-shaped backend for adapter and pipeline smoke tests."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.decode_counts: defaultdict[str, int] = defaultdict(int)
        self.closed_sessions: list[str] = []

    def prefill_interleaved(
        self, *, session, messages: list[UGInterleavedMessage]
    ) -> UGModelPrefillResult:
        self._record("prefill", session)
        token_count = 0
        for message in messages:
            if message.type == "text":
                token_count += len(str(message.content).split())
            elif message.type == "image":
                token_count += 2
            else:
                raise ValueError(f"Unsupported BAGEL message type: {message.type}")
        return UGModelPrefillResult(added_tokens=token_count)

    def decode_next_segment(self, *, session) -> UGDecodeResult:
        self._record("decode", session)
        session_id = session.handle.session_id
        decode_count = self.decode_counts[session_id]
        self.decode_counts[session_id] += 1
        if decode_count == 0:
            return UGDecodeResult(type="image_marker")
        if decode_count == 1:
            return UGDecodeResult(type="text", text="bagel_mock_text_after_image")
        return UGDecodeResult(type="done")

    def predict_velocity_from_session(
        self, *, session, request: UGVelocityRequest
    ) -> torch.Tensor:
        self._record("velocity", session)
        scale = 2.0 + session.srt_request_count * 0.1
        return request.latent_tokens + scale * request.timestep.reshape(-1, 1, 1).to(
            request.latent_tokens
        )

    def append_generated_image(
        self, *, session, image: Any | None
    ) -> UGModelAppendImageResult:
        del image
        self._record("append_image", session)
        return UGModelAppendImageResult(added_tokens=2)

    def close_session(self, *, session_id: str) -> None:
        self.events.append(("close", session_id))
        self.closed_sessions.append(session_id)

    def _record(self, event: str, session) -> None:
        self.events.append((event, session.handle.session_id))


def create_bagel_ug_model_adapter(model_path: str) -> BAGELUGModelAdapter:
    if "mock-bagel" in model_path.lower():
        return BAGELUGModelAdapter(model_path, backend=MockBAGELBackend())
    return BAGELUGModelAdapter(model_path)


def _require_keys(
    payload: dict[str, Any], required: tuple[str, ...], name: str
) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise BAGELDenoiseStepError(f"{name} is missing required keys: {missing}")


def _clone_context(context: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(context)


def _find_spec(module_name: str):
    try:
        return importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
