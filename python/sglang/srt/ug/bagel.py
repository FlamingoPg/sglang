# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from collections import defaultdict
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
        timesteps = timestep_shift * timesteps / (
            1 + (timestep_shift - 1) * timesteps
        )
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


class BAGELBackendProtocol(Protocol):
    def prefill_interleaved(
        self, *, session, messages: list[UGInterleavedMessage]
    ) -> UGModelPrefillResult:
        ...

    def decode_next_segment(self, *, session) -> UGDecodeResult:
        ...

    def predict_velocity_from_session(
        self, *, session, request: UGVelocityRequest
    ) -> torch.Tensor:
        ...

    def append_generated_image(
        self, *, session, image: Any | None
    ) -> UGModelAppendImageResult:
        ...

    def close_session(self, *, session_id: str) -> None:
        ...


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
            "Real BAGEL backend loading is not wired yet: official BAGEL "
            "InterleaveInferencer.gen_image owns the denoising loop, while "
            "SGLang UG requires predict_velocity_from_session for each G step."
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


def _require_keys(payload: dict[str, Any], required: tuple[str, ...], name: str) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise BAGELDenoiseStepError(f"{name} is missing required keys: {missing}")


def _find_spec(module_name: str):
    try:
        return importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
