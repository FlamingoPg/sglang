# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


UGBoundaryType = Literal["text", "image_marker", "done"]
UGGeneratedSegmentType = Literal["text", "image"]


@dataclass(frozen=True, slots=True)
class UGContextRef:
    """Opaque coordinator reference to a backend-owned context.

    The coordinator may keep this reference to ask a backend for more work, but
    it must not expose raw KV allocator internals such as page or slot numbers.
    """

    context_id: str
    session_id: str | None = None
    version: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UGBoundary:
    """Result of U decoding until a generation boundary."""

    type: UGBoundaryType
    text: str | None = None
    token_ids: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GeneratedSegment:
    """Model-generated segment returned by a stateless G backend."""

    type: UGGeneratedSegmentType
    text: str | None = None
    image: Any | None = None
    commit_payload: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UGCoordinatorRequest:
    """Input to the internal UG coordinator."""

    messages: tuple[Any, ...]
    mode: str = "interleave"
    sampling_params: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UGCoordinatorResponse:
    """Ordered result assembled by the internal UG coordinator."""

    segments: tuple[GeneratedSegment, ...]
    context: UGContextRef | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class GContextOps(Protocol):
    """Backend-neutral access to temporary context-backed query execution."""

    def forward_queries(
        self,
        *,
        query_embeds: Any,
        position_ids: Any | None = None,
        attention_mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any: ...


class UBackend(Protocol):
    """Stateless UG-facing wrapper around U/session capabilities."""

    def prepare_context(self, request: UGCoordinatorRequest) -> UGContextRef: ...

    def decode_until_boundary(
        self,
        context: UGContextRef,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> UGBoundary: ...

    def append_generated_segment(
        self,
        context: UGContextRef,
        segment: GeneratedSegment,
    ) -> UGContextRef: ...

    def continue_decode(
        self,
        context: UGContextRef,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> UGBoundary: ...

    def release(self, context: UGContextRef) -> None: ...


class GBackend(Protocol):
    """Stateless G segment generator used by the UG coordinator."""

    def generate_segment(
        self,
        *,
        request: UGCoordinatorRequest,
        context_ops: GContextOps,
        params: Any | None = None,
    ) -> GeneratedSegment: ...
