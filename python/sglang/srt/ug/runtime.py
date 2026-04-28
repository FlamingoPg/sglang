# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol

import torch

from sglang.srt.ug.context import UGSessionHandle


class UGSegmentState(str, Enum):
    U_PREFILL = "u_prefill"
    U_DECODE = "u_decode"
    G_DENOISE = "g_denoise"
    APPEND_IMAGE = "append_image"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class UGInterleavedMessage:
    type: Literal["text", "image"]
    content: Any


@dataclass(frozen=True, slots=True)
class UGVelocityRequest:
    session: UGSessionHandle
    latent_tokens: torch.Tensor
    timestep: torch.Tensor
    latent_position_ids: torch.Tensor
    sampling_params: Any


@dataclass(frozen=True, slots=True)
class UGVelocityResponse:
    session: UGSessionHandle
    velocity: torch.Tensor


@dataclass(frozen=True, slots=True)
class UGDecodeResult:
    type: Literal["text", "image_marker", "done"]
    text: str | None = None


@dataclass(slots=True)
class UGSessionRecord:
    session_id: str
    state: UGSegmentState
    anchor_request_id: str
    context_length: int = 0
    context_version: int = 0
    prefill_count: int = 0
    velocity_count: int = 0
    append_image_count: int = 0
    decode_count: int = 0
    closed: bool = False

    def handle(self) -> UGSessionHandle:
        return UGSessionHandle(
            session_id=self.session_id,
            anchor_request_id=self.anchor_request_id,
            context_length=self.context_length,
            context_version=self.context_version,
        )


class UGModelRunnerProtocol(Protocol):
    def prefill_interleaved(
        self, *, session_id: str, messages: list[UGInterleavedMessage]
    ) -> int:
        ...

    def decode_next_segment(self, *, record: UGSessionRecord) -> UGDecodeResult:
        ...

    def predict_velocity_from_session(
        self, *, request: UGVelocityRequest, record: UGSessionRecord
    ) -> torch.Tensor:
        ...

    def append_generated_image(
        self, *, record: UGSessionRecord, image: Any | None
    ) -> int:
        ...

    def close_session(self, *, session_id: str) -> None:
        ...


class FakeUGModelRunner:
    """Deterministic UG model shell used to prove session/KV ownership plumbing."""

    def prefill_interleaved(
        self, *, session_id: str, messages: list[UGInterleavedMessage]
    ) -> int:
        del session_id
        token_count = 0
        for message in messages:
            if message.type == "text":
                token_count += len(str(message.content).split())
            elif message.type == "image":
                token_count += 2
            else:
                raise ValueError(f"Unsupported UG message type: {message.type}")
        return token_count

    def decode_next_segment(self, *, record: UGSessionRecord) -> UGDecodeResult:
        if record.append_image_count == 0 and record.decode_count == 0:
            return UGDecodeResult(type="image_marker")
        if record.append_image_count > 0 and record.decode_count == 1:
            return UGDecodeResult(type="text", text="generated_text_after_image")
        return UGDecodeResult(type="done")

    def predict_velocity_from_session(
        self, *, request: UGVelocityRequest, record: UGSessionRecord
    ) -> torch.Tensor:
        scale = 1.0 + record.context_length * 0.01 + record.context_version * 0.001
        return request.latent_tokens + scale * request.timestep.reshape(-1, 1, 1).to(
            request.latent_tokens
        )

    def append_generated_image(
        self, *, record: UGSessionRecord, image: Any | None
    ) -> int:
        del record, image
        return 2

    def close_session(self, *, session_id: str) -> None:
        del session_id


class UGSessionRuntime:
    """Lightweight UG state machine layered on top of SRT sessions."""

    def __init__(
        self,
        *,
        model_runner: UGModelRunnerProtocol | None = None,
        session_controller: Any | None = None,
        capacity_of_str_len: int = 4096,
    ) -> None:
        self.model_runner = model_runner or FakeUGModelRunner()
        self.session_controller = session_controller
        self.capacity_of_str_len = capacity_of_str_len
        self._records: dict[str, UGSessionRecord] = {}

    @staticmethod
    def normalize_messages(
        *, prompt: str | list[str] | None = None, image: Any | None = None
    ) -> list[UGInterleavedMessage]:
        messages: list[UGInterleavedMessage] = []
        if image is not None:
            messages.append(UGInterleavedMessage(type="image", content=image))
        if prompt is not None:
            prompt_text = " ".join(prompt) if isinstance(prompt, list) else prompt
            messages.append(UGInterleavedMessage(type="text", content=prompt_text))
        return messages

    def prefill_interleaved(
        self,
        messages: list[UGInterleavedMessage],
        *,
        session_id: str | None = None,
    ) -> UGSessionHandle:
        if not messages:
            raise ValueError("UG prefill requires at least one text or image message")
        session_id = session_id or uuid.uuid4().hex
        record = self._records.get(session_id)
        if record is None:
            self._ensure_srt_session(session_id)
            record = UGSessionRecord(
                session_id=session_id,
                state=UGSegmentState.U_PREFILL,
                anchor_request_id=f"{session_id}:u0",
            )
            self._records[session_id] = record
        elif record.closed:
            raise ValueError(f"UG session {session_id} is closed")
        elif record.state not in {UGSegmentState.U_DECODE, UGSegmentState.DONE}:
            raise ValueError(
                f"Cannot prefill UG session {session_id} from state {record.state}"
            )
        else:
            record.state = UGSegmentState.U_PREFILL

        added_tokens = self.model_runner.prefill_interleaved(
            session_id=session_id, messages=messages
        )
        record.context_length += added_tokens
        record.context_version += 1
        record.prefill_count += 1
        record.anchor_request_id = f"{session_id}:u{record.context_version}"
        record.state = UGSegmentState.U_DECODE
        return record.handle()

    def begin_g_denoise(self, handle: UGSessionHandle) -> UGSessionHandle:
        record = self._record_for(handle)
        if record.state != UGSegmentState.U_DECODE:
            raise ValueError(
                f"Cannot enter G denoise from state {record.state} "
                f"for UG session {handle.session_id}"
            )
        record.state = UGSegmentState.G_DENOISE
        return record.handle()

    def decode_next_segment(self, handle: UGSessionHandle) -> UGDecodeResult:
        record = self._record_for(handle)
        if record.state != UGSegmentState.U_DECODE:
            raise ValueError(
                f"Cannot decode U segment from state {record.state} "
                f"for UG session {handle.session_id}"
            )
        result = self.model_runner.decode_next_segment(record=record)
        record.decode_count += 1
        if result.type == "image_marker":
            record.state = UGSegmentState.G_DENOISE
        elif result.type == "done":
            record.state = UGSegmentState.DONE
        else:
            record.state = UGSegmentState.U_DECODE
        return result

    def predict_velocity(self, request: UGVelocityRequest) -> UGVelocityResponse:
        record = self._record_for(request.session)
        if record.state != UGSegmentState.G_DENOISE:
            raise ValueError(
                f"Cannot predict UG velocity from state {record.state} "
                f"for UG session {request.session.session_id}"
            )
        velocity = self.model_runner.predict_velocity_from_session(
            request=request, record=record
        )
        record.velocity_count += 1
        return UGVelocityResponse(session=record.handle(), velocity=velocity)

    def append_generated_image(
        self, handle: UGSessionHandle, image: Any | None
    ) -> UGSessionHandle:
        record = self._record_for(handle)
        if record.state != UGSegmentState.G_DENOISE:
            raise ValueError(
                f"Cannot append generated image from state {record.state} "
                f"for UG session {handle.session_id}"
            )
        record.state = UGSegmentState.APPEND_IMAGE
        added_tokens = self.model_runner.append_generated_image(
            record=record, image=image
        )
        record.context_length += added_tokens
        record.context_version += 1
        record.append_image_count += 1
        record.anchor_request_id = f"{record.session_id}:u{record.context_version}"
        record.state = UGSegmentState.U_DECODE
        return record.handle()

    def close_session(self, handle_or_session_id: UGSessionHandle | str) -> None:
        session_id = (
            handle_or_session_id.session_id
            if isinstance(handle_or_session_id, UGSessionHandle)
            else handle_or_session_id
        )
        record = self._records.get(session_id)
        if record is not None:
            record.closed = True
            record.state = UGSegmentState.DONE
        self.model_runner.close_session(session_id=session_id)
        self._close_srt_session(session_id)

    def get_state(self, handle_or_session_id: UGSessionHandle | str) -> UGSegmentState:
        return self._record_for(handle_or_session_id).state

    def get_debug_counters(self, handle_or_session_id: UGSessionHandle | str) -> dict:
        record = self._record_for(handle_or_session_id)
        return {
            "session_id": record.session_id,
            "state": record.state.value,
            "context_length": record.context_length,
            "context_version": record.context_version,
            "prefill_count": record.prefill_count,
            "velocity_count": record.velocity_count,
            "append_image_count": record.append_image_count,
            "decode_count": record.decode_count,
        }

    def _record_for(self, handle_or_session_id: UGSessionHandle | str) -> UGSessionRecord:
        session_id = (
            handle_or_session_id.session_id
            if isinstance(handle_or_session_id, UGSessionHandle)
            else handle_or_session_id
        )
        record = self._records.get(session_id)
        if record is None or record.closed:
            raise ValueError(f"Unknown or closed UG session: {session_id}")
        if isinstance(handle_or_session_id, UGSessionHandle):
            if handle_or_session_id.context_version != record.context_version:
                raise ValueError(
                    "Stale UG session handle: "
                    f"{handle_or_session_id.context_version} != "
                    f"{record.context_version}"
                )
        return record

    def _ensure_srt_session(self, session_id: str) -> None:
        if self.session_controller is None or session_id in self.session_controller:
            return
        from sglang.srt.managers.io_struct import OpenSessionReqInput

        output = self.session_controller.open(
            OpenSessionReqInput(
                session_id=session_id,
                capacity_of_str_len=self.capacity_of_str_len,
            )
        )
        if not getattr(output, "success", False):
            raise RuntimeError(f"Failed to open SRT session for UG: {session_id}")

    def _close_srt_session(self, session_id: str) -> None:
        if self.session_controller is None or session_id not in self.session_controller:
            return
        from sglang.srt.managers.io_struct import CloseSessionReqInput

        self.session_controller.close(CloseSessionReqInput(session_id=session_id))
