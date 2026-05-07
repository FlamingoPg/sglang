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
from sglang.multimodal_gen.runtime.pipelines_core.stages.sensenova_u1 import (
    SenseNovaU1CommitStage,
    SenseNovaU1DecodeStage,
    SenseNovaU1GSegmentStage,
    SenseNovaU1InputStage,
    SenseNovaU1UContextStage,
    _normalize_pipeline_interleaved_messages,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.ug.interleaved import (
    DEFAULT_UG_TEXT_MAX_NEW_TOKENS,
    UGInterleavedRequest,
    UGInterleavedResponse,
    UGRuntimeStats,
    normalize_ug_generation_mode,
)
from sglang.srt.ug.middle import UGMiddleBridge
from sglang.srt.ug.sensenova_u1 import build_sensenova_u1_middle_bridge


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
            modules["srt_middle_bridge"] = build_sensenova_u1_middle_bridge(
                scheduler=getattr(server_args, "ug_srt_scheduler", None),
                srt_u_decode_max_new_tokens=getattr(
                    server_args, "ug_srt_u_decode_max_new_tokens", None
                ),
            )
        if "g_segment_executor" not in modules:
            modules["g_segment_executor"] = SenseNovaU1PixelFlowGSegmentExecutor()
        return modules

    def create_pipeline_stages(self, server_args: ServerArgs):
        bridge = self.get_module("srt_middle_bridge")
        g_segment_executor = self.get_module("g_segment_executor")
        self.add_stage(SenseNovaU1InputStage(bridge))
        self.add_stage(SenseNovaU1UContextStage(bridge))
        self.add_stage(SenseNovaU1GSegmentStage(bridge, g_segment_executor))
        self.add_stage(SenseNovaU1CommitStage(bridge))
        self.add_stage(SenseNovaU1DecodeStage(bridge))

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ):
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
        input_stage = SenseNovaU1InputStage(bridge)
        context_stage = SenseNovaU1UContextStage(bridge)
        g_stage = SenseNovaU1GSegmentStage(bridge, g_segment_executor)
        commit_stage = SenseNovaU1CommitStage(bridge)

        batch = input_stage.forward(batch, server_args)
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
            batch = commit_stage.forward(batch, server_args)
            batch.extra.pop("ug_generated_segment", None)
            batch.extra.pop("ug_generated_segment_committed", None)

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
