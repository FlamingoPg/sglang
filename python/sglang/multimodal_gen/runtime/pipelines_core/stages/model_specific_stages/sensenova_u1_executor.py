# SPDX-License-Identifier: Apache-2.0

from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_decode import (
    SenseNovaU1PixelFlowDecoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_denoise import (
    SenseNovaU1PixelFlowDenoiser,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_prepare import (
    SenseNovaU1PixelFlowPreparer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_types import (
    U1_EDIT_IMG_CONDITION_ROLE,
    U1_EDIT_UNCONDITION_ROLE,
    U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
    U1_T2I_CFG_UNCONDITION_ROLE,
    U1GContext,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_utils import (
    resolve_pixel_flow_cfg,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class SenseNovaU1PixelFlowGSegmentExecutor:
    """Run SenseNova U1 pixel-flow G through the model-specific diffusion stages."""

    required_g_kind: str = "pixel_flow"

    def __call__(
        self,
        *,
        context_ops: Any,
        batch: Req,
        server_args: ServerArgs,
    ) -> Any:
        if getattr(context_ops, "g_kind", None) != self.required_g_kind:
            raise ValueError(
                "SenseNova U1 pixel-flow executor requires g_kind='pixel_flow', got "
                f"{getattr(context_ops, 'g_kind', None)!r}"
            )
        get_model = getattr(context_ops, "get_model", None)
        if not callable(get_model):
            raise RuntimeError("SenseNova U1 pixel-flow requires model access")
        forward_batch_provider = getattr(
            context_ops,
            "build_temporary_forward_batch",
            None,
        )
        if not callable(forward_batch_provider):
            raise RuntimeError(
                "SenseNova U1 pixel-flow requires temporary query forward batches"
            )
        (
            u1_context,
            cfg_img_condition_u1_context,
            cfg_uncondition_u1_context,
        ) = _resolve_u1_contexts(
            context_ops=context_ops,
            batch=batch,
        )
        native_runner = _SenseNovaU1NativePixelFlowRunner(
            get_model(),
            forward_batch_provider=forward_batch_provider,
        )
        return native_runner.generate(
            context_metadata=dict(getattr(context_ops, "metadata", {}) or {}),
            batch=batch,
            server_args=server_args,
            u1_context=u1_context,
            cfg_img_condition_u1_context=cfg_img_condition_u1_context,
            cfg_uncondition_u1_context=cfg_uncondition_u1_context,
        )


def _resolve_u1_contexts(
    *,
    context_ops: Any,
    batch: Req,
) -> tuple[U1GContext, U1GContext | None, U1GContext | None]:
    get_position_count = getattr(context_ops, "get_position_count", None)
    if not callable(get_position_count):
        raise RuntimeError(
            "SenseNova U1 pixel-flow requires latest context position count"
        )

    u1_context = _require_context(
        context_ops,
        "SenseNova U1 pixel-flow has no context position count",
    )
    cfg_img_condition_context = None
    cfg_uncondition_context = None
    sampling_params = batch.sampling_params
    mode = getattr(sampling_params, "ug_generation_mode", None)
    cfg = resolve_pixel_flow_cfg(sampling_params)

    t2i_uncondition_role = context_ops.get_role(
        "t2i_cfg_uncondition_role",
        U1_T2I_CFG_UNCONDITION_ROLE,
    )
    interleave_text_uncondition_role = context_ops.get_role(
        "interleave_text_uncondition_role",
        U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
    )
    edit_img_condition_role = context_ops.get_role(
        "edit_img_condition_role",
        U1_EDIT_IMG_CONDITION_ROLE,
    )
    edit_uncondition_role = context_ops.get_role(
        "edit_uncondition_role",
        U1_EDIT_UNCONDITION_ROLE,
    )

    if mode == "edit":
        if cfg.needs_img_condition:
            cfg_img_condition_context = _require_context(
                context_ops,
                "SenseNova U1 edit image CFG requires sidecar context position count",
                edit_img_condition_role,
            )
        if cfg.needs_uncondition:
            cfg_uncondition_context = _require_context(
                context_ops,
                "SenseNova U1 edit uncondition CFG requires sidecar context position count",
                edit_uncondition_role,
            )
    elif mode == "interleave":
        if cfg.needs_img_condition:
            cfg_img_condition_context = _require_context(
                context_ops,
                "SenseNova U1 interleave text CFG requires sidecar context position count",
                interleave_text_uncondition_role,
            )
        if cfg.needs_uncondition:
            cfg_uncondition_context = _require_context(
                context_ops,
                "SenseNova U1 interleave image CFG requires sidecar context position count",
                t2i_uncondition_role,
            )
    elif cfg.text_scale > 1.0:
        cfg_img_condition_context = _require_context(
            context_ops,
            "SenseNova U1 pixel-flow CFG requires sidecar context position count",
            t2i_uncondition_role,
        )
    return u1_context, cfg_img_condition_context, cfg_uncondition_context


def _require_context(
    context_ops: Any,
    message: str,
    sidecar_role: str | None = None,
) -> U1GContext:
    position_count = context_ops.get_position_count(sidecar_role=sidecar_role)
    if position_count is None:
        suffix = f" sidecar {sidecar_role}" if sidecar_role is not None else ""
        raise RuntimeError(f"{message} for context {context_ops.session_id}{suffix}")
    return U1GContext(
        session_id=context_ops.session_id,
        sidecar_role=sidecar_role,
        position_count=int(position_count),
    )


class _SenseNovaU1NativePixelFlowRunner:
    """Model-specific pixel-flow runner fed by generic context ops."""

    def __init__(
        self,
        model: Any,
        *,
        forward_batch_provider: Any,
    ) -> None:
        self.preparer = SenseNovaU1PixelFlowPreparer(model)
        self.denoiser = SenseNovaU1PixelFlowDenoiser(
            model,
            forward_batch_provider=forward_batch_provider,
        )
        self.decoder = SenseNovaU1PixelFlowDecoder()

    def generate(
        self,
        *,
        context_metadata: dict[str, Any],
        batch: Any,
        server_args: Any,
        u1_context: U1GContext,
        cfg_img_condition_u1_context: U1GContext | None = None,
        cfg_uncondition_u1_context: U1GContext | None = None,
    ) -> Any:
        import torch

        del server_args
        with torch.inference_mode():
            prepared = self.preparer.forward(
                context_metadata=context_metadata,
                batch=batch,
                u1_context=u1_context,
                cfg_img_condition_u1_context=cfg_img_condition_u1_context,
                cfg_uncondition_u1_context=cfg_uncondition_u1_context,
            )
            image_prediction = self.denoiser.forward(prepared)
            return self.decoder.forward(prepared, image_prediction)
