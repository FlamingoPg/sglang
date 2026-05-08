# SPDX-License-Identifier: Apache-2.0

from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_decode import (
    SenseNovaU1PixelFlowDecodeStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_denoise import (
    SenseNovaU1PixelFlowDenoiseStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_prepare import (
    SenseNovaU1PixelFlowPrepareStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_types import (
    U1_EDIT_IMG_CONDITION_ROLE,
    U1_EDIT_UNCONDITION_ROLE,
    U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
    U1_T2I_CFG_UNCONDITION_ROLE,
    SRTGContext,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_utils import (
    require_sidecar_srt_context,
    require_srt_context,
    resolve_pixel_flow_cfg,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

class SenseNovaU1PixelFlowGSegmentExecutor:
    """Run SenseNova U1 pixel-flow G through the model-specific diffusion stages."""

    required_g_kind: str = "pixel_flow"

    def __call__(
        self,
        *,
        bridge: Any,
        contexts: Any,
        batch: Req,
        server_args: ServerArgs,
    ) -> Any:
        if getattr(bridge, "g_kind", None) != self.required_g_kind:
            raise ValueError(
                "SenseNova U1 pixel-flow executor requires g_kind='pixel_flow', got "
                f"{getattr(bridge, 'g_kind', None)!r}"
            )
        runtime = getattr(bridge, "runtime", None)
        if runtime is None or getattr(runtime, "srt_request_executor", None) is None:
            raise RuntimeError(
                "SenseNova U1 pixel-flow requires a SRT-backed UG runtime"
            )
        srt_executor = runtime.srt_request_executor
        get_srt_model = getattr(srt_executor, "get_srt_model", None)
        if not callable(get_srt_model):
            raise RuntimeError(
                "SenseNova U1 pixel-flow requires SRT executor model access"
            )
        forward_batch_provider = getattr(
            srt_executor,
            "build_ug_g_forward_batch_for_session",
            None,
        )
        if not callable(forward_batch_provider):
            raise RuntimeError(
                "SenseNova U1 pixel-flow requires SRT temporary G forward batches"
            )
        (
            srt_context,
            cfg_img_condition_srt_context,
            cfg_uncondition_srt_context,
        ) = _resolve_srt_contexts(
            bridge=bridge,
            contexts=contexts,
            batch=batch,
            srt_executor=srt_executor,
        )
        native_runner = _SenseNovaU1NativePixelFlowRunner(
            get_srt_model(),
            forward_batch_provider=forward_batch_provider,
        )
        return native_runner.generate(
            contexts=contexts,
            batch=batch,
            server_args=server_args,
            srt_context=srt_context,
            cfg_img_condition_srt_context=cfg_img_condition_srt_context,
            cfg_uncondition_srt_context=cfg_uncondition_srt_context,
        )


def _resolve_srt_contexts(
    *,
    bridge: Any,
    contexts: Any,
    batch: Req,
    srt_executor: Any,
) -> tuple[SRTGContext, SRTGContext | None, SRTGContext | None]:
    if contexts.full.session is None:
        raise ValueError("SenseNova U1 pixel-flow requires a SRT UG session")
    get_position_count = getattr(
        srt_executor,
        "get_latest_ug_session_position_count",
        None,
    )
    if not callable(get_position_count):
        raise RuntimeError(
            "SenseNova U1 pixel-flow requires latest SRT session position count"
        )

    session_id = contexts.full.session.session_id
    srt_context = require_srt_context(
        get_position_count,
        session_id,
        "SenseNova U1 pixel-flow has no SRT session position count",
    )
    cfg_img_condition_context = None
    cfg_uncondition_context = None
    sampling_params = batch.sampling_params
    mode = getattr(sampling_params, "ug_generation_mode", None)
    cfg = resolve_pixel_flow_cfg(sampling_params)

    t2i_uncondition_role = getattr(
        bridge,
        "t2i_cfg_uncondition_role",
        U1_T2I_CFG_UNCONDITION_ROLE,
    )
    interleave_text_uncondition_role = getattr(
        bridge,
        "interleave_text_uncondition_role",
        U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
    )
    edit_img_condition_role = getattr(
        bridge,
        "edit_img_condition_role",
        U1_EDIT_IMG_CONDITION_ROLE,
    )
    edit_uncondition_role = getattr(
        bridge,
        "edit_uncondition_role",
        U1_EDIT_UNCONDITION_ROLE,
    )

    if mode == "edit":
        if cfg.needs_img_condition:
            cfg_img_condition_context = require_sidecar_srt_context(
                get_position_count,
                session_id,
                edit_img_condition_role,
                "SenseNova U1 edit image CFG requires sidecar SRT position count",
            )
        if cfg.needs_uncondition:
            cfg_uncondition_context = require_sidecar_srt_context(
                get_position_count,
                session_id,
                edit_uncondition_role,
                "SenseNova U1 edit uncondition CFG requires sidecar SRT position count",
            )
    elif mode == "interleave":
        if cfg.needs_img_condition:
            cfg_img_condition_context = require_sidecar_srt_context(
                get_position_count,
                session_id,
                interleave_text_uncondition_role,
                "SenseNova U1 interleave text CFG requires sidecar SRT position count",
            )
        if cfg.needs_uncondition:
            cfg_uncondition_context = require_sidecar_srt_context(
                get_position_count,
                session_id,
                t2i_uncondition_role,
                "SenseNova U1 interleave image CFG requires sidecar SRT position count",
            )
    elif cfg.text_scale > 1.0:
        cfg_img_condition_context = require_sidecar_srt_context(
            get_position_count,
            session_id,
            t2i_uncondition_role,
            "SenseNova U1 pixel-flow CFG requires sidecar SRT position count",
        )
    return srt_context, cfg_img_condition_context, cfg_uncondition_context


class _SenseNovaU1NativePixelFlowRunner:
    """Model-specific pixel-flow runner; SRT only supplies KV-backed forwards."""

    def __init__(
        self,
        srt_model: Any,
        *,
        forward_batch_provider: Any,
    ) -> None:
        self.prepare_stage = SenseNovaU1PixelFlowPrepareStage(srt_model)
        self.denoise_stage = SenseNovaU1PixelFlowDenoiseStage(
            srt_model,
            forward_batch_provider=forward_batch_provider,
        )
        self.decode_stage = SenseNovaU1PixelFlowDecodeStage()

    def generate(
        self,
        *,
        contexts: Any,
        batch: Any,
        server_args: Any,
        srt_context: SRTGContext,
        cfg_img_condition_srt_context: SRTGContext | None = None,
        cfg_uncondition_srt_context: SRTGContext | None = None,
    ) -> Any:
        import torch

        del server_args
        with torch.inference_mode():
            prepared = self.prepare_stage.forward(
                contexts=contexts,
                batch=batch,
                srt_context=srt_context,
                cfg_img_condition_srt_context=cfg_img_condition_srt_context,
                cfg_uncondition_srt_context=cfg_uncondition_srt_context,
            )
            image_prediction = self.denoise_stage.forward(prepared)
            return self.decode_stage.forward(prepared, image_prediction)
