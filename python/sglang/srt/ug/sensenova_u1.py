# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from PIL import Image

from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    SenseNovaU1SamplingParams,
    build_sensenova_u1_sampling_params,
)
from sglang.multimodal_gen.runtime.models.vision_utils import load_image
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1 import (
    SenseNovaU1PixelFlowGSegmentExecutor,
)
from sglang.srt.ug.sensenova_u1_context import (
    U1_EDIT_IMG_CONDITION_ROLE,
    U1_EDIT_UNCONDITION_ROLE,
    U1_IMG_START_TOKEN,
    U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
    U1_T2I_CFG_UNCONDITION_ROLE,
    _u1_decode_token_ids,
    _u1_eos_token_ids,
    _u1_needs_any_cfg,
    _u1_needs_text_cfg,
    _u1_token_id,
    build_u1_native_edit_img_condition_prepared_input,
    build_u1_native_edit_prepared_input,
    build_u1_native_edit_uncondition_prepared_input,
    build_u1_native_generated_image_commit_prepared_input,
    build_u1_native_interleave_prepared_input,
    build_u1_native_interleave_text_uncondition_marker_prepared_input,
    build_u1_native_interleave_text_uncondition_prepared_input,
    build_u1_native_t2i_cfg_uncondition_prepared_input,
    build_u1_native_t2i_prepared_input,
    build_u1_native_vlm_prepared_input,
)
from sglang.srt.ug.adapter import (
    UGModelAppendImageResult,
    UGModelPrefillResult,
    UGModelRunnerAdapter,
)
from sglang.srt.ug.context import UGContextBundle
from sglang.srt.ug.coordinator import UGInterleaveCoordinator
from sglang.srt.ug.interleaved import (
    DEFAULT_UG_TEXT_MAX_NEW_TOKENS,
    UGGKind,
    UGInputSegment,
    UGInterleavedRequest,
    UGInterleavedResponse,
    UGRuntimeStats,
    normalize_ug_generation_mode,
)
from sglang.srt.ug.middle import SRTBackedUGMiddleBridge, UGGSegmentExecutor
from sglang.srt.ug.runtime import (
    UGDecodeResult,
    UGInterleavedMessage,
    UGSegmentState,
    UGSessionRuntime,
    UGSRTPreparedInput,
    UGVLMTextGenerationResult,
)
from sglang.srt.ug.srt_executor import UGSRTSchedulerExecutor


def build_sensenova_u1_middle_bridge(
    *,
    scheduler: Any,
    srt_request_executor: Any | None = None,
    srt_u_decode_max_new_tokens: int | None = None,
) -> "U1SRTBackedUGMiddleBridge":
    """Build the SRT-owned middle bridge for SenseNova U1.

    Diffusion pipelines should consume the returned bridge as a module; SRT owns
    the scheduler/session/runtime wiring here.
    """

    if srt_request_executor is None:
        if scheduler is None:
            raise ValueError(
                "SenseNova U1 requires an attached SRT scheduler so U owns the session/KV"
            )
        srt_request_executor = UGSRTSchedulerExecutor(scheduler)
    if srt_u_decode_max_new_tokens is None:
        srt_u_decode_max_new_tokens = 0
    session_controller = srt_request_executor.session_controller
    model_config = getattr(scheduler, "model_config", None)
    runtime = UGSessionRuntime(
        model_runner=UGModelRunnerAdapter(
            U1UGModelAdapter(native_tokenizer=getattr(scheduler, "tokenizer", None))
        ),
        session_controller=session_controller,
        srt_request_executor=srt_request_executor,
        tokenizer=getattr(scheduler, "tokenizer", None),
        vocab_size=getattr(model_config, "vocab_size", 32000),
        srt_u_decode_max_new_tokens=srt_u_decode_max_new_tokens,
    )
    return U1SRTBackedUGMiddleBridge(runtime)


def build_sensenova_u1_coordinator(
    *,
    scheduler: Any,
    srt_request_executor: Any | None = None,
    srt_u_decode_max_new_tokens: int | None = None,
    g_segment_executor: Any | None = None,
) -> "SenseNovaU1UGCoordinator":
    return SenseNovaU1UGCoordinator(
        bridge=build_sensenova_u1_middle_bridge(
            scheduler=scheduler,
            srt_request_executor=srt_request_executor,
            srt_u_decode_max_new_tokens=srt_u_decode_max_new_tokens,
        ),
        g_segment_executor=g_segment_executor,
    )


class SenseNovaU1UGCoordinator:
    """Internal owner of SenseNova U1 U/G interleave flow."""

    def __init__(
        self,
        *,
        bridge: "U1SRTBackedUGMiddleBridge",
        g_segment_executor: Any | None = None,
    ) -> None:
        self.bridge = bridge
        self.g_segment_executor = (
            g_segment_executor
            if g_segment_executor is not None
            else SenseNovaU1PixelFlowGSegmentExecutor()
        )
        self._coordinator = UGInterleaveCoordinator(
            bridge=self.bridge,
            g_segment_executor=self.g_segment_executor,
        )

    def forward_interleaved(
        self,
        messages: UGInterleavedRequest | list[Any],
        sampling_params: SenseNovaU1SamplingParams | dict[str, Any] | None = None,
        server_args: Any | None = None,
        **sampling_kwargs: Any,
    ) -> UGInterleavedResponse:
        if server_args is None:
            raise ValueError("SenseNova U1 coordinator requires server_args")
        request = _normalize_sensenova_u1_request(
            messages,
            sampling_params,
            sampling_kwargs,
        )
        metadata = dict(request.metadata)
        metadata["mode"] = normalize_ug_generation_mode(
            metadata.get("mode"), default="interleave"
        )
        if metadata["mode"] == "vlm":
            return self.forward_vlm_request(request, metadata=metadata)

        setattr(request.sampling_params, "ug_generation_mode", metadata["mode"])
        batch = Req(
            sampling_params=request.sampling_params,
            extra={
                "ug_interleaved_messages": request.to_legacy_segments(),
                "ug_request_metadata": metadata,
                "ug_mode": metadata["mode"],
            },
        )
        contexts = None
        try:
            messages = _normalize_u1_interleaved_messages(request)
            think = _resolve_u1_think(request.sampling_params, metadata)
            think_max_new_tokens = _resolve_u1_think_max_new_tokens(
                request.sampling_params,
                metadata,
            )
            contexts = self.bridge.prepare_u_context_from_messages(
                messages=messages,
                think=think,
                think_max_new_tokens=think_max_new_tokens,
                sampling_params=request.sampling_params,
            )
            batch.extra["ug_contexts"] = contexts
            batch.extra["ug_pre_image_segments"] = contexts.full.metadata.get(
                "pre_image_segments", []
            )
            output_segments = self._coordinator.run_generation(
                batch=batch,
                contexts=contexts,
                server_args=server_args,
                metadata=metadata,
            )
            return UGInterleavedResponse.from_legacy_segments(
                output_segments,
                stats=_collect_interleaved_runtime_stats(self.bridge, contexts),
                metadata=metadata,
            )
        finally:
            if contexts is not None:
                self.bridge.release(contexts)

    def forward_interleaved_batch(
        self,
        requests: list[UGInterleavedRequest],
        server_args: Any | None = None,
    ) -> list[UGInterleavedResponse]:
        return [
            self.forward_interleaved(request, server_args=server_args)
            for request in requests
        ]

    def forward_vlm(
        self,
        messages: UGInterleavedRequest | list[Any],
        sampling_params: SenseNovaU1SamplingParams | dict[str, Any] | None = None,
        server_args: Any | None = None,
        max_new_tokens: int | None = None,
        **sampling_kwargs: Any,
    ) -> UGInterleavedResponse:
        del server_args
        request = _normalize_sensenova_u1_request(
            messages,
            sampling_params,
            sampling_kwargs,
        )
        metadata = dict(request.metadata)
        metadata["mode"] = "vlm"
        return self.forward_vlm_request(
            request,
            max_new_tokens=max_new_tokens,
            metadata=metadata,
        )

    def forward_vlm_batch(
        self,
        requests: list[UGInterleavedRequest],
        server_args: Any | None = None,
    ) -> list[UGInterleavedResponse]:
        return [
            self.forward_vlm(request, server_args=server_args) for request in requests
        ]

    def forward_vlm_request(
        self,
        request: UGInterleavedRequest,
        *,
        max_new_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UGInterleavedResponse:
        max_new_tokens = _resolve_vlm_max_new_tokens(
            request.metadata,
            explicit_max_new_tokens=max_new_tokens,
        )
        result = self.bridge.generate_vlm_text(
            messages=_normalize_u1_interleaved_messages(request),
            max_new_tokens=max_new_tokens,
        )
        runtime = getattr(self.bridge, "runtime", None)
        try:
            segment_metadata = {
                name: list(value)
                for name, value in (
                    ("token_ids", result.token_ids),
                    ("next_token_ids", result.next_token_ids),
                    ("position_ids", result.position_ids),
                )
                if value
            }
            return UGInterleavedResponse.from_legacy_segments(
                [
                    {
                        "type": "text",
                        "text": result.text,
                        "metadata": segment_metadata,
                    }
                ],
                stats=_collect_runtime_stats_from_session(self.bridge, result.session),
                metadata=metadata or {"mode": "vlm"},
            )
        finally:
            if runtime is not None:
                runtime.close_session(result.session)


class U1UGModelAdapter:
    """SenseNova U1 UG adapter shell for the UG middle protocol.

    U1 uses pixel-flow G mechanics; image-generation math stays in this backend
    instead of the common UG middle layer.
    """

    g_kind: UGGKind = "pixel_flow"

    def __init__(
        self,
        *,
        native_tokenizer: Any | None = None,
    ) -> None:
        self.native_tokenizer = native_tokenizer
        self.include_t2i_cfg_uncondition = False
        self.include_interleave_text_uncondition = False
        self.include_edit_img_condition = False
        self.include_edit_uncondition = False
        self.native_generation_mode: str | None = None
        self.native_interleave_think_mode = False

    def prepare_srt_u_interleaved_inputs(
        self,
        *,
        session: Any,
        messages: list[UGInterleavedMessage],
        state: Any,
    ) -> list[UGSRTPreparedInput] | None:
        if state != UGSegmentState.U_PREFILL or self.native_tokenizer is None:
            return None
        has_image = any(message.type == "image" for message in messages)
        has_text = any(message.type == "text" for message in messages)
        if not has_text:
            return None
        if self.native_generation_mode == "interleave":
            prepared = []
            if self.include_interleave_text_uncondition:
                prepared.append(
                    build_u1_native_interleave_text_uncondition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        messages=messages,
                        session=session,
                    )
                )
            if self.include_t2i_cfg_uncondition:
                prepared.append(
                    build_u1_native_t2i_cfg_uncondition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        session=session,
                    )
                )
            prepared.append(
                build_u1_native_interleave_prepared_input(
                    tokenizer=self.native_tokenizer,
                    messages=messages,
                    session=session,
                    think_mode=self.native_interleave_think_mode,
                )
            )
            return prepared
        if not has_image:
            prepared = []
            if self.include_t2i_cfg_uncondition:
                prepared.append(
                    build_u1_native_t2i_cfg_uncondition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        session=session,
                    )
                )
            prepared.append(
                build_u1_native_t2i_prepared_input(
                    tokenizer=self.native_tokenizer,
                    messages=messages,
                    session=session,
                )
            )
            return prepared
        if self.native_generation_mode == "edit":
            prepared = []
            if self.include_edit_img_condition:
                prepared.append(
                    build_u1_native_edit_img_condition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        messages=messages,
                        session=session,
                    )
                )
            if self.include_edit_uncondition:
                prepared.append(
                    build_u1_native_edit_uncondition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        session=session,
                    )
                )
            prepared.append(
                build_u1_native_edit_prepared_input(
                    tokenizer=self.native_tokenizer,
                    messages=messages,
                    session=session,
                )
            )
            return prepared
        return [
            build_u1_native_vlm_prepared_input(
                tokenizer=self.native_tokenizer,
                messages=messages,
                session=session,
            )
        ]

    def prepare_srt_u_message_inputs(
        self,
        *,
        session: Any,
        message: Any,
        state: Any,
    ) -> list[UGSRTPreparedInput] | None:
        if self.native_tokenizer is None:
            return None
        if message.type == "text":
            return None
        if message.type == "image":
            if state == UGSegmentState.APPEND_IMAGE:
                return [
                    build_u1_native_generated_image_commit_prepared_input(
                        tokenizer=self.native_tokenizer,
                        image=message.content,
                        session=session,
                    )
                ]
        return None

    def prefill_interleaved(
        self,
        *,
        session: Any,
        messages: list[Any],
    ) -> UGModelPrefillResult:
        return UGModelPrefillResult(
            added_tokens=self._added_tokens_from_srt_session_view(session)
        )

    def decode_next_segment(self, *, session: Any) -> Any:
        raise RuntimeError("SenseNova U1 decode requires the SRT-backed runtime path")

    def decode_next_segment_from_runtime(self, *, runtime: Any, session: Any) -> Any:
        session_view = self._runtime_session_view(runtime=runtime, session=session)
        u1_state = (
            (getattr(session_view, "metadata", {}) or {})
            .get("ug_model_state", {})
            .get("u1", {})
        )
        if bool(u1_state.get("native_interleave_prompt")):
            return self._decode_native_interleave_next_segment(
                runtime=runtime,
                session=session,
                u1_state=u1_state,
            )
        if not self._has_generated_image_commit(session_view):
            return UGDecodeResult(type="image_marker")
        if self.native_tokenizer is None or runtime is None:
            raise RuntimeError("SenseNova U1 decode requires a tokenizer and runtime")
        if getattr(runtime, "srt_request_executor", None) is None:
            raise RuntimeError("SenseNova U1 decode requires a SRT request executor")
        max_new_tokens = max(
            1,
            int(getattr(runtime, "srt_u_decode_max_new_tokens", 0) or 0),
        )
        decoded = runtime.decode_text(
            session,
            max_new_tokens=max_new_tokens,
            greedy=True,
        )
        return UGDecodeResult(
            type="text",
            text=decoded.text,
            token_ids=tuple(int(token_id) for token_id in decoded.output_ids),
        )

    def _decode_native_interleave_next_segment(
        self,
        *,
        runtime: Any,
        session: Any,
        u1_state: dict[str, Any],
    ) -> UGDecodeResult:
        if self.native_tokenizer is None or runtime is None:
            return UGDecodeResult(type="done")
        if bool(u1_state.get("interleave_pending_image_marker")):
            self._merge_runtime_u1_state(
                runtime,
                session,
                {
                    "interleave_pending_image_marker": False,
                    "open_image_marker": True,
                },
            )
            return UGDecodeResult(type="image_marker")
        if getattr(runtime, "srt_request_executor", None) is None:
            return UGDecodeResult(type="done")

        img_start_id = _u1_token_id(self.native_tokenizer, U1_IMG_START_TOKEN)
        eos_token_ids = _u1_eos_token_ids(self.native_tokenizer)
        max_new_tokens = max(
            1,
            int(getattr(runtime, "srt_u_decode_max_new_tokens", 0) or 0),
        )
        generated_text_ids: list[int] = []
        current_position = int(
            u1_state.get(
                "g_position_start",
                getattr(session, "context_length", 0) or 0,
            )
            or 0
        )

        for _ in range(max_new_tokens):
            decoded = runtime.decode_text(
                session,
                max_new_tokens=1,
                decode_position_id=current_position,
                greedy=True,
                model_state_updates={
                    "u1": {
                        "last_segment_type": "interleave",
                        "last_source": "native_interleave_decode",
                        "native_interleave_prompt": True,
                        "g_position_start": current_position,
                    }
                },
            )
            if not decoded.output_ids:
                self._merge_runtime_u1_state(
                    runtime,
                    session,
                    {"g_position_start": current_position},
                )
                break
            token_id = int(decoded.output_ids[-1])
            current_position += 1
            if token_id == img_start_id:
                state_updates = {
                    "last_segment_type": "interleave",
                    "last_source": "native_interleave_image_marker",
                    "native_interleave_prompt": True,
                    "open_image_marker": True,
                    "interleave_pending_image_marker": bool(generated_text_ids),
                    "g_position_start": current_position,
                }
                commit_decode_token = getattr(
                    runtime, "commit_u_decode_input_token", None
                )
                if callable(commit_decode_token):
                    session = commit_decode_token(
                        session,
                        token_id=token_id,
                        position_id=current_position - 1,
                        model_state_updates={"u1": state_updates},
                    )
                else:
                    self._merge_runtime_u1_state(runtime, session, state_updates)
                self._append_interleave_text_uncondition_marker(
                    runtime=runtime,
                    session=session,
                )
                if generated_text_ids:
                    return UGDecodeResult(
                        type="text",
                        text=_u1_decode_token_ids(
                            self.native_tokenizer,
                            generated_text_ids,
                        ),
                        token_ids=tuple(generated_text_ids),
                    )
                return UGDecodeResult(type="image_marker")
            if token_id in eos_token_ids:
                self._merge_runtime_u1_state(
                    runtime,
                    session,
                    {
                        "last_segment_type": "interleave",
                        "last_source": "native_interleave_eos",
                        "native_interleave_prompt": True,
                        "g_position_start": current_position,
                    },
                )
                if generated_text_ids:
                    return UGDecodeResult(
                        type="text",
                        text=_u1_decode_token_ids(
                            self.native_tokenizer,
                            generated_text_ids,
                        ),
                        token_ids=tuple(generated_text_ids),
                    )
                return UGDecodeResult(type="done")
            generated_text_ids.append(token_id)

        self._merge_runtime_u1_state(
            runtime,
            session,
            {
                "last_segment_type": "interleave",
                "last_source": "native_interleave_decode",
                "native_interleave_prompt": True,
                "g_position_start": current_position,
            },
        )
        if generated_text_ids:
            return UGDecodeResult(
                type="text",
                text=_u1_decode_token_ids(self.native_tokenizer, generated_text_ids),
                token_ids=tuple(generated_text_ids),
            )
        return UGDecodeResult(type="done")

    def _append_interleave_text_uncondition_marker(
        self,
        *,
        runtime: Any,
        session: Any,
    ) -> None:
        if self.native_tokenizer is None:
            return
        append_sidecar = getattr(runtime, "append_srt_sidecar_prepared_input", None)
        get_sidecar_state = getattr(runtime, "get_srt_sidecar_model_state", None)
        if not callable(append_sidecar) or not callable(get_sidecar_state):
            return
        sidecar_state = get_sidecar_state(
            session,
            U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        )
        u1_state = (sidecar_state or {}).get("u1") or {}
        if not u1_state:
            return
        if bool(u1_state.get("open_image_marker")):
            return
        logical_position = u1_state.get("g_position_start")
        if logical_position is None:
            return
        prepared = build_u1_native_interleave_text_uncondition_marker_prepared_input(
            tokenizer=self.native_tokenizer,
            session=session,
            logical_position=int(logical_position),
        )
        append_sidecar(session, prepared, state=UGSegmentState.U_DECODE)

    @staticmethod
    def _merge_runtime_u1_state(
        runtime: Any,
        session: Any,
        updates: dict[str, Any],
    ) -> None:
        record_for = getattr(runtime, "_record_for", None)
        merge = getattr(runtime, "_merge_ug_model_state_updates", None)
        if not callable(record_for) or not callable(merge):
            return
        try:
            record = record_for(session)
        except Exception:
            return
        merge(record, {"u1": dict(updates)})

    @staticmethod
    def _runtime_session_view(*, runtime: Any, session: Any) -> Any:
        if getattr(session, "metadata", None) is not None:
            return session
        metadata: dict[str, Any] = {}
        get_debug_counters = getattr(runtime, "get_debug_counters", None)
        if callable(get_debug_counters):
            try:
                counters = get_debug_counters(session)
                metadata["ug_model_state"] = counters.get("ug_model_state", {})
                metadata["srt_last_u_decode_output_ids"] = counters.get(
                    "srt_last_u_decode_output_ids",
                    (),
                )
            except Exception:
                metadata = {}
        return SimpleNamespace(handle=session, metadata=metadata)

    def decode_vlm_text(
        self,
        *,
        runtime: Any,
        session: Any,
        max_new_tokens: int,
    ) -> Any:
        if runtime is None:
            raise RuntimeError("SenseNova U1 VLM text generation requires SRT runtime")
        decoded = runtime.decode_text(
            session,
            max_new_tokens=max_new_tokens,
            greedy=True,
        )
        return UGVLMTextGenerationResult(
            session=decoded.session,
            text=decoded.text,
            token_ids=decoded.input_ids,
            next_token_ids=decoded.output_ids,
            position_ids=decoded.position_ids,
        )

    def append_generated_image(
        self,
        *,
        session: Any,
        image: Any | None,
    ) -> UGModelAppendImageResult:
        del image
        return UGModelAppendImageResult(
            added_tokens=self._added_tokens_from_srt_session_view(session)
        )

    def close_session(self, *, session_id: str) -> None:
        del session_id

    def _has_generated_image_commit(self, session: Any) -> bool:
        session_metadata = getattr(session, "metadata", {}) or {}
        model_state = session_metadata.get("ug_model_state") or {}
        u1_state = model_state.get("u1") or {}
        if bool(u1_state.get("last_generated_image_commit")):
            return True
        return any(
            bool(segment.get("generated_image_commit"))
            for segment in u1_state.get("segments", [])
        )

    def _added_tokens_from_srt_session_view(
        self,
        session: Any,
    ) -> int:
        handle = getattr(session, "handle", None)
        previous_length = int(getattr(handle, "context_length", 0) or 0)
        srt_length = int(getattr(session, "srt_last_origin_input_len", 0) or 0)
        if srt_length > previous_length:
            return srt_length - previous_length
        return 0


class U1SRTBackedUGMiddleBridge:
    """Pixel-flow U1 bridge shell backed by the common SRT UG session runtime."""

    g_kind: UGGKind = "pixel_flow"
    t2i_cfg_uncondition_role = U1_T2I_CFG_UNCONDITION_ROLE
    interleave_text_uncondition_role = U1_INTERLEAVE_TEXT_UNCONDITION_ROLE
    edit_img_condition_role = U1_EDIT_IMG_CONDITION_ROLE
    edit_uncondition_role = U1_EDIT_UNCONDITION_ROLE

    def __init__(
        self,
        runtime: UGSessionRuntime,
        *,
        max_pre_image_decode_steps: int = 128,
    ) -> None:
        self.runtime = runtime
        self._bridge = SRTBackedUGMiddleBridge(
            runtime,
            max_pre_image_decode_steps=max_pre_image_decode_steps,
        )

    def prepare_u_context(
        self,
        *,
        prompt: str | list[str] | None,
        image: Any | None,
        think: bool = False,
        think_max_new_tokens: int | None = None,
        sampling_params: Any | None = None,
    ) -> UGContextBundle:
        with self._temporary_generation_settings(sampling_params):
            bridge_think = (
                False
                if getattr(sampling_params, "ug_generation_mode", None) == "interleave"
                else think
            )
            contexts = self._bridge.prepare_u_context(
                prompt=prompt,
                image=image,
                think=bridge_think,
                think_max_new_tokens=think_max_new_tokens,
                sampling_params=sampling_params,
            )
        return contexts

    def prepare_u_context_from_messages(
        self,
        *,
        messages: list[UGInterleavedMessage | dict[str, Any]],
        think: bool = False,
        think_max_new_tokens: int | None = None,
        sampling_params: Any | None = None,
    ) -> UGContextBundle:
        with self._temporary_generation_settings(sampling_params):
            bridge_think = (
                False
                if getattr(sampling_params, "ug_generation_mode", None) == "interleave"
                else think
            )
            contexts = self._bridge.prepare_u_context_from_messages(
                messages=messages,
                think=bridge_think,
                think_max_new_tokens=think_max_new_tokens,
                sampling_params=sampling_params,
            )
        return contexts

    @contextmanager
    def _temporary_generation_settings(self, sampling_params: Any | None):
        adapter = getattr(self.runtime.model_runner, "adapter", None)
        old_cfg = getattr(adapter, "include_t2i_cfg_uncondition", False)
        old_interleave_text_uncondition = getattr(
            adapter,
            "include_interleave_text_uncondition",
            False,
        )
        old_edit_img_condition = getattr(adapter, "include_edit_img_condition", False)
        old_edit_uncondition = getattr(adapter, "include_edit_uncondition", False)
        old_mode = getattr(adapter, "native_generation_mode", None)
        old_interleave_think_mode = getattr(
            adapter,
            "native_interleave_think_mode",
            False,
        )
        mode = getattr(sampling_params, "ug_generation_mode", None)
        if adapter is not None:
            needs_cfg = _u1_needs_any_cfg(sampling_params)
            cfg_text_scale = float(getattr(sampling_params, "cfg_text_scale", 1.0))
            cfg_img_scale = float(getattr(sampling_params, "cfg_img_scale", 1.0))
            adapter.include_t2i_cfg_uncondition = (
                _u1_needs_text_cfg(sampling_params)
                and mode not in {"edit", "interleave"}
            ) or (mode == "interleave" and cfg_img_scale != 1.0)
            adapter.include_interleave_text_uncondition = (
                mode == "interleave" and _u1_needs_text_cfg(sampling_params)
            )
            adapter.include_edit_img_condition = (
                mode == "edit"
                and needs_cfg
                and (cfg_img_scale == 1.0 or cfg_text_scale != cfg_img_scale)
            )
            adapter.include_edit_uncondition = (
                mode == "edit" and needs_cfg and cfg_img_scale != 1.0
            )
            adapter.native_generation_mode = mode
            adapter.native_interleave_think_mode = bool(
                getattr(sampling_params, "think", False)
            )
        try:
            yield
        finally:
            if adapter is not None:
                adapter.include_t2i_cfg_uncondition = old_cfg
                adapter.include_interleave_text_uncondition = (
                    old_interleave_text_uncondition
                )
                adapter.include_edit_img_condition = old_edit_img_condition
                adapter.include_edit_uncondition = old_edit_uncondition
                adapter.native_generation_mode = old_mode
                adapter.native_interleave_think_mode = old_interleave_think_mode

    def run_g_segment(
        self,
        *,
        contexts: UGContextBundle,
        executor: UGGSegmentExecutor,
    ) -> Any:
        return self._bridge.run_g_segment(contexts=contexts, executor=executor)

    def commit_generated_segment(
        self,
        *,
        contexts: UGContextBundle,
        segment: Any,
    ) -> None:
        self._bridge.commit_generated_segment(contexts=contexts, segment=segment)
        self._commit_interleave_text_uncondition_sidecar(
            contexts=contexts,
            segment=segment,
        )

    def release(self, contexts: UGContextBundle) -> None:
        self._bridge.release(contexts)

    def continue_u_decode(self, *, contexts: UGContextBundle) -> UGDecodeResult:
        return self._bridge.continue_u_decode(contexts=contexts)

    def _commit_interleave_text_uncondition_sidecar(
        self,
        *,
        contexts: UGContextBundle,
        segment: Any,
    ) -> None:
        adapter = getattr(self.runtime.model_runner, "adapter", None)
        tokenizer = getattr(adapter, "native_tokenizer", None)
        if tokenizer is None or contexts.full.session is None:
            return
        image = getattr(segment, "commit_image", None)
        if image is None:
            image = getattr(segment, "image", None)
        if image is None:
            return
        get_handle = getattr(self.runtime, "get_srt_sidecar_handle", None)
        get_state = getattr(self.runtime, "get_srt_sidecar_model_state", None)
        append_sidecar = getattr(
            self.runtime, "append_srt_sidecar_prepared_input", None
        )
        if (
            not callable(get_handle)
            or not callable(get_state)
            or not callable(append_sidecar)
        ):
            return
        sidecar_handle = get_handle(
            contexts.full.session,
            U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        )
        if sidecar_handle is None:
            return
        sidecar_state = get_state(
            contexts.full.session,
            U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        )
        if not ((sidecar_state or {}).get("u1") or {}).get("open_image_marker"):
            return
        sidecar_session = SimpleNamespace(
            handle=sidecar_handle,
            metadata={"ug_model_state": sidecar_state},
        )
        prepared = build_u1_native_generated_image_commit_prepared_input(
            tokenizer=tokenizer,
            image=image,
            session=sidecar_session,
        )
        prepared.srt_sidecar_role = U1_INTERLEAVE_TEXT_UNCONDITION_ROLE
        prepared.srt_sidecar_session_id = sidecar_handle.session_id
        append_sidecar(
            contexts.full.session,
            prepared,
            state=UGSegmentState.APPEND_IMAGE,
        )

    def generate_vlm_text(
        self,
        *,
        messages: list[UGInterleavedMessage | dict[str, Any]],
        max_new_tokens: int,
    ) -> UGVLMTextGenerationResult:
        return self._bridge.generate_vlm_text(
            messages=messages,
            max_new_tokens=max_new_tokens,
        )


def _normalize_sensenova_u1_request(
    messages: UGInterleavedRequest | list[Any],
    sampling_params: SenseNovaU1SamplingParams | dict[str, Any] | None,
    sampling_kwargs: dict[str, Any],
) -> UGInterleavedRequest:
    if isinstance(messages, UGInterleavedRequest):
        if messages.sampling_params is not None and (
            sampling_params is not None or sampling_kwargs
        ):
            raise ValueError(
                "SenseNova U1 request already contains sampling_params; pass "
                "overrides by constructing a new UGInterleavedRequest"
            )
        return UGInterleavedRequest(
            messages=messages.messages,
            sampling_params=_normalize_sensenova_u1_sampling_params(
                (
                    messages.sampling_params
                    if sampling_params is None
                    else sampling_params
                ),
                sampling_kwargs,
            ),
            metadata=dict(messages.metadata),
        )
    return UGInterleavedRequest.from_segments(
        messages,
        sampling_params=_normalize_sensenova_u1_sampling_params(
            sampling_params,
            sampling_kwargs,
        ),
    )


def _normalize_sensenova_u1_sampling_params(
    sampling_params: SenseNovaU1SamplingParams | dict[str, Any] | None,
    sampling_kwargs: dict[str, Any],
) -> SenseNovaU1SamplingParams:
    if sampling_params is None:
        return build_sensenova_u1_sampling_params(sampling_kwargs)
    if isinstance(sampling_params, dict):
        values = dict(sampling_params)
        values.update(sampling_kwargs)
        return build_sensenova_u1_sampling_params(values)
    if sampling_kwargs:
        raise ValueError(
            "SenseNova U1 sampling keyword overrides require sampling_params "
            "to be omitted or passed as a dict"
        )
    return sampling_params


def _normalize_u1_interleaved_messages(
    request: UGInterleavedRequest,
) -> list[UGInterleavedMessage]:
    normalized: list[UGInterleavedMessage] = []
    for message in request.messages:
        if isinstance(message, UGInputSegment):
            segment = message.to_legacy_segment()
        elif isinstance(message, dict):
            segment = message
        else:
            segment = message.to_legacy_segment()
        message_type = segment.get("type")
        if message_type == "text":
            content = segment.get("text", segment.get("content"))
        elif message_type == "image":
            content = segment.get("image", segment.get("content"))
            if content is None:
                raise ValueError("SenseNova U1 image message is missing content")
            if isinstance(content, dict) and "image" in content:
                image_payload = dict(content)
                image = image_payload["image"]
                if not isinstance(image, Image.Image):
                    image = load_image(image)
                image_payload["image"] = image
                content = image_payload
            elif not isinstance(content, Image.Image):
                content = load_image(content)
        else:
            raise ValueError(f"Unsupported SenseNova U1 message type: {message_type!r}")
        if content is None:
            raise ValueError(f"SenseNova U1 {message_type} message is missing content")
        normalized.append(UGInterleavedMessage(type=message_type, content=content))
    if not normalized:
        raise ValueError("SenseNova U1 messages must not be empty")
    return normalized


def _resolve_vlm_max_new_tokens(
    metadata: dict[str, Any],
    *,
    explicit_max_new_tokens: int | None = None,
) -> int:
    value = explicit_max_new_tokens
    if value is None:
        value = metadata.get(
            "max_new_tokens",
            metadata.get("max_length", DEFAULT_UG_TEXT_MAX_NEW_TOKENS),
        )
    value = int(value)
    if value <= 0:
        raise ValueError(
            f"SenseNova U1 VLM max_new_tokens must be positive, got {value}"
        )
    return value


def _resolve_u1_think(sampling_params: Any, metadata: dict[str, Any]) -> bool:
    if "think" in metadata:
        return _coerce_u1_bool(metadata["think"], name="think")
    return bool(getattr(sampling_params, "think", False))


def _resolve_u1_think_max_new_tokens(
    sampling_params: Any,
    metadata: dict[str, Any],
) -> int | None:
    value = metadata.get(
        "think_max_new_tokens",
        getattr(sampling_params, "think_max_new_tokens", None),
    )
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"think_max_new_tokens must be positive when set, got {value}")
    return value


def _coerce_u1_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a bool, got {value!r}")


def _collect_interleaved_runtime_stats(
    bridge: Any,
    contexts: Any | None,
) -> UGRuntimeStats | None:
    if contexts is None or contexts.full.session is None:
        return None
    runtime = getattr(bridge, "runtime", None)
    if runtime is None:
        return None
    return UGRuntimeStats.from_debug_counters(
        runtime.get_debug_counters(contexts.full.session)
    )


def _collect_runtime_stats_from_session(
    bridge: Any,
    session: Any | None,
) -> UGRuntimeStats | None:
    if session is None:
        return None
    runtime = getattr(bridge, "runtime", None)
    if runtime is None:
        return None
    return UGRuntimeStats.from_debug_counters(runtime.get_debug_counters(session))
