# SPDX-License-Identifier: Apache-2.0

from typing import Any

from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    SenseNovaU1SamplingParams,
    build_sensenova_u1_sampling_params,
)
from sglang.multimodal_gen.runtime.pipelines_core import ComposedPipelineBase
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1 import (
    SenseNovaU1PixelFlowGSegmentExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_context import (
    U1SRTBackedUGMiddleBridge,
    U1UGModelAdapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.sensenova_u1 import (
    SenseNovaU1ContextStage,
    SenseNovaU1DecodeStage,
    SenseNovaU1GSegmentStage,
    _normalize_pipeline_interleaved_messages,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.ug.adapter import UGModelRunnerAdapter
from sglang.srt.ug.interleaved import (
    DEFAULT_UG_TEXT_MAX_NEW_TOKENS,
    UGInterleavedRequest,
    UGInterleavedResponse,
    UGRuntimeStats,
    normalize_ug_generation_mode,
)
from sglang.srt.ug.middle import UGMiddleBridge
from sglang.srt.ug.runtime import UGSessionRuntime
from sglang.srt.ug.srt_executor import UGSRTSchedulerExecutor


def _build_srt_owned_session_runtime(
    model_runner=None,
    *,
    scheduler=None,
    srt_request_executor=None,
    srt_u_decode_max_new_tokens: int = 0,
) -> UGSessionRuntime:
    if srt_request_executor is None:
        srt_request_executor = _build_srt_request_executor(scheduler)
    session_controller = srt_request_executor.session_controller
    model_config = getattr(scheduler, "model_config", None)
    return UGSessionRuntime(
        model_runner=model_runner,
        session_controller=session_controller,
        srt_request_executor=srt_request_executor,
        tokenizer=getattr(scheduler, "tokenizer", None),
        vocab_size=getattr(model_config, "vocab_size", 32000),
        srt_u_decode_max_new_tokens=srt_u_decode_max_new_tokens,
    )


def _build_srt_request_executor(scheduler=None):
    if scheduler is None:
        raise ValueError(
            "SenseNovaU1Pipeline requires an attached SRT scheduler so U owns the session/KV"
        )
    return UGSRTSchedulerExecutor(scheduler)


def _build_sensenova_u1_middle_bridge(
    scheduler,
    srt_request_executor,
    srt_u_decode_max_new_tokens: int | None,
) -> UGMiddleBridge:
    if srt_u_decode_max_new_tokens is None:
        srt_u_decode_max_new_tokens = 0
    return U1SRTBackedUGMiddleBridge(
        _build_srt_owned_session_runtime(
            UGModelRunnerAdapter(
                U1UGModelAdapter(native_tokenizer=getattr(scheduler, "tokenizer", None))
            ),
            scheduler=scheduler,
            srt_request_executor=srt_request_executor,
            srt_u_decode_max_new_tokens=srt_u_decode_max_new_tokens,
        )
    )


def _build_sensenova_u1_g_segment_executor(bridge: UGMiddleBridge):
    g_kind = getattr(bridge, "g_kind", None)
    if g_kind == "pixel_flow":
        return SenseNovaU1PixelFlowGSegmentExecutor()
    raise ValueError(f"Unsupported SenseNova U1 G kind: {g_kind!r}")


class SenseNovaU1Pipeline(ComposedPipelineBase):
    pipeline_name = "SenseNovaU1Pipeline"
    _required_config_modules: list[str] = []

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        modules = dict(loaded_modules or {})
        if "srt_middle_bridge" not in modules:
            srt_request_executor = _build_srt_request_executor(
                getattr(server_args, "ug_srt_scheduler", None)
            )
            modules["srt_middle_bridge"] = _build_sensenova_u1_middle_bridge(
                getattr(server_args, "ug_srt_scheduler", None),
                srt_request_executor,
                getattr(server_args, "ug_srt_u_decode_max_new_tokens", None),
            )
        if "g_segment_executor" not in modules:
            modules["g_segment_executor"] = _build_sensenova_u1_g_segment_executor(
                modules["srt_middle_bridge"]
            )
        return modules

    def create_pipeline_stages(self, server_args: ServerArgs):
        bridge = self.get_module("srt_middle_bridge")
        g_segment_executor = self.get_module("g_segment_executor")
        self.add_stage(SenseNovaU1ContextStage(bridge))
        self.add_stage(SenseNovaU1GSegmentStage(bridge, g_segment_executor))
        self.add_stage(SenseNovaU1DecodeStage(bridge))

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ):
        _apply_bridge_sampling_defaults(self.get_module("srt_middle_bridge"), batch)
        return super().forward(batch, server_args)

    def forward_interleaved(
        self,
        messages: UGInterleavedRequest | list[Any],
        sampling_params: SenseNovaU1SamplingParams | dict[str, Any] | None = None,
        server_args: ServerArgs | None = None,
        **sampling_kwargs: Any,
    ) -> UGInterleavedResponse:
        """Experimental SenseNova U1 interleaved API.

        This is intentionally Python-only and internal for now. It accepts a
        single interleaved request and returns ordered output segments without
        promising OpenAI-compatible request or response shapes.
        """

        server_args = server_args or self.server_args
        if server_args is None:
            raise ValueError("SenseNova U1 interleaved API requires server_args")
        request = _normalize_interleaved_request(
            messages, sampling_params, sampling_kwargs
        )
        metadata = dict(request.metadata)
        metadata["mode"] = normalize_ug_generation_mode(
            metadata.get("mode"), default="interleave"
        )
        if metadata["mode"] == "vlm":
            return self._forward_vlm_request(request, metadata=metadata)
        batch = Req(
            sampling_params=request.sampling_params,
            extra={
                "ug_interleaved_messages": request.to_legacy_segments(),
                "ug_request_metadata": metadata,
                "ug_mode": metadata["mode"],
            },
        )
        try:
            if metadata["mode"] == "interleave":
                result = self._forward_interleave_loop(
                    batch,
                    server_args,
                    metadata=metadata,
                )
            else:
                result = self.forward(batch, server_args)
            contexts = result.extra.get("ug_contexts")
            stats = _collect_interleaved_runtime_stats(
                self.get_module("srt_middle_bridge"), contexts
            )
            return UGInterleavedResponse.from_legacy_segments(
                list(result.extra["ug_output_segments"]),
                stats=stats,
                metadata=metadata,
            )
        finally:
            contexts = batch.extra.get("ug_contexts")
            if contexts is not None:
                self.get_module("srt_middle_bridge").release(contexts)

    def _forward_interleave_loop(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        metadata: dict[str, Any],
    ) -> Req:
        bridge = self.get_module("srt_middle_bridge")
        g_segment_executor = self.get_module("g_segment_executor")
        context_stage = SenseNovaU1ContextStage(bridge)
        g_stage = SenseNovaU1GSegmentStage(bridge, g_segment_executor)

        _apply_bridge_sampling_defaults(bridge, batch)
        batch = context_stage.forward(batch, server_args)
        contexts = batch.extra.get("ug_contexts")
        if contexts is None:
            raise ValueError(
                "SenseNova U1 interleave loop requires prepared SRT middle contexts"
            )

        output_segments = list(batch.extra.get("ug_pre_image_segments", []))
        max_images = _resolve_positive_metadata_int(
            metadata,
            "max_interleave_images",
            default=1,
        )
        max_text_segments = _resolve_positive_metadata_int(
            metadata,
            "max_interleave_text_segments",
            default=1,
        )

        for _ in range(max_images):
            batch = g_stage.forward(batch, server_args)
            generated_segment = batch.extra.get("ug_generated_segment")
            if generated_segment is None:
                raise ValueError(
                    "SenseNova U1 interleave loop expected a generated image segment"
                )
            image_for_append = generated_segment.image
            output_segments.append(
                {
                    "type": "image",
                    "image": image_for_append,
                    "metadata": dict(generated_segment.metadata),
                }
            )
            bridge.commit_generated_segment(
                contexts=contexts,
                segment=generated_segment,
            )
            batch.extra.pop("ug_generated_segment", None)

            next_image_requested = False
            for _ in range(max_text_segments):
                post_segment = bridge.continue_u_decode(contexts=contexts)
                if post_segment.type == "text":
                    segment = {"type": "text", "text": post_segment.text or ""}
                    if post_segment.token_ids:
                        segment["metadata"] = {
                            "token_ids": [
                                int(token_id) for token_id in post_segment.token_ids
                            ]
                        }
                    output_segments.append(segment)
                    continue
                if post_segment.type == "image_marker":
                    next_image_requested = True
                    break
                if post_segment.type == "done":
                    batch.extra["ug_output_segments"] = output_segments
                    return batch
                raise ValueError(
                    "SenseNova U1 interleave loop expected U text, "
                    "image marker, or done, "
                    f"got {post_segment.type}"
                )
            if not next_image_requested:
                break

        batch.extra["ug_output_segments"] = output_segments
        return batch

    def forward_interleaved_batch(
        self,
        requests: list[UGInterleavedRequest],
        server_args: ServerArgs | None = None,
    ) -> list[UGInterleavedResponse]:
        return [
            self.forward_interleaved(request, server_args=server_args)
            for request in requests
        ]

    def forward_vlm(
        self,
        messages: UGInterleavedRequest | list[Any],
        sampling_params: SenseNovaU1SamplingParams | dict[str, Any] | None = None,
        server_args: ServerArgs | None = None,
        max_new_tokens: int | None = None,
        **sampling_kwargs: Any,
    ) -> UGInterleavedResponse:
        """Experimental SenseNova U1 VLM-only API.

        This path runs only SRT-owned U prefill and U text decode. It must not
        enter G preparation, G execution, image decode, or append-image stages.
        """

        server_args = server_args or self.server_args
        if server_args is None:
            raise ValueError("SenseNova U1 VLM API requires server_args")
        request = _normalize_interleaved_request(
            messages, sampling_params, sampling_kwargs
        )
        metadata = dict(request.metadata)
        metadata["mode"] = "vlm"
        return self._forward_vlm_request(
            request,
            max_new_tokens=max_new_tokens,
            metadata=metadata,
        )

    def _forward_vlm_request(
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
        bridge = self.get_module("srt_middle_bridge")
        generate_vlm_text = getattr(bridge, "generate_vlm_text", None)
        if not callable(generate_vlm_text):
            raise RuntimeError(
                f"{bridge.__class__.__name__} does not support "
                "SenseNova U1 VLM text generation"
            )

        result = generate_vlm_text(
            messages=_normalize_pipeline_interleaved_messages(request),
            max_new_tokens=max_new_tokens,
        )
        runtime = getattr(bridge, "runtime", None)
        try:
            stats = _collect_runtime_stats_from_session(bridge, result.session)
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
                stats=stats,
                metadata=metadata or {"mode": "vlm"},
            )
        finally:
            if runtime is not None:
                runtime.close_session(result.session)

    def forward_vlm_batch(
        self,
        requests: list[UGInterleavedRequest],
        server_args: ServerArgs | None = None,
    ) -> list[UGInterleavedResponse]:
        return [
            self.forward_vlm(request, server_args=server_args) for request in requests
        ]


EntryClass = SenseNovaU1Pipeline


def _normalize_interleaved_request(
    messages: UGInterleavedRequest | list[Any],
    sampling_params: SenseNovaU1SamplingParams | dict[str, Any] | None,
    sampling_kwargs: dict[str, Any],
) -> UGInterleavedRequest:
    if isinstance(messages, UGInterleavedRequest):
        if messages.sampling_params is not None and (
            sampling_params is not None or sampling_kwargs
        ):
            raise ValueError(
                "SenseNova U1 interleaved request already contains "
                "sampling_params; pass "
                "overrides by constructing a new UGInterleavedRequest"
            )
        return UGInterleavedRequest(
            messages=messages.messages,
            sampling_params=_normalize_interleaved_sampling_params(
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
        sampling_params=_normalize_interleaved_sampling_params(
            sampling_params, sampling_kwargs
        ),
    )


def _normalize_interleaved_sampling_params(
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
            "SenseNova U1 interleaved sampling keyword overrides require "
            "sampling_params "
            "to be omitted or passed as a dict"
        )
    return sampling_params


def _apply_bridge_sampling_defaults(bridge: UGMiddleBridge, batch: Req) -> None:
    if batch.sampling_params is None:
        return
    apply_defaults = getattr(bridge, "apply_sampling_defaults", None)
    if not callable(apply_defaults):
        return
    interleaved_messages = batch.extra.get("ug_interleaved_messages")
    request_metadata = dict(batch.extra.get("ug_request_metadata") or {})
    mode = _resolve_generation_mode_for_defaults(
        batch, request_metadata, interleaved_messages
    )
    setattr(batch.sampling_params, "ug_generation_mode", mode)
    apply_defaults(
        batch.sampling_params,
        mode=mode,
        has_input_image=_has_input_image(batch, interleaved_messages),
        explicit_fields=set(batch.extra.get("explicit_fields", [])),
    )


def _resolve_generation_mode_for_defaults(
    batch: Req,
    metadata: dict[str, Any],
    interleaved_messages,
):
    if "ug_mode" in batch.extra:
        return normalize_ug_generation_mode(
            batch.extra["ug_mode"], default="interleave"
        )
    if "mode" in metadata:
        return normalize_ug_generation_mode(metadata["mode"], default="interleave")
    if interleaved_messages is not None:
        return "interleave"
    return "edit" if batch.condition_image is not None or batch.image_path else "t2i"


def _has_input_image(batch: Req, interleaved_messages) -> bool:
    if interleaved_messages is not None:
        return any(
            (
                isinstance(message, dict)
                and message.get("type") == "image"
                or getattr(message, "type", None) == "image"
            )
            for message in interleaved_messages
        )
    return batch.condition_image is not None or batch.image_path is not None


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


def _resolve_positive_metadata_int(
    metadata: dict[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    value = int(metadata.get(key, default))
    if value <= 0:
        raise ValueError(f"SenseNova U1 metadata {key} must be positive, got {value}")
    return value


def _collect_interleaved_runtime_stats(
    bridge: UGMiddleBridge,
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
    bridge: UGMiddleBridge,
    session: Any | None,
) -> UGRuntimeStats | None:
    if session is None:
        return None
    runtime = getattr(bridge, "runtime", None)
    if runtime is None:
        return None
    return UGRuntimeStats.from_debug_counters(runtime.get_debug_counters(session))
