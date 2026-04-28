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


class BAGELAdapterError(RuntimeError):
    """Raised when the BAGEL adapter cannot be constructed safely."""


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


def _find_spec(module_name: str):
    try:
        return importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
