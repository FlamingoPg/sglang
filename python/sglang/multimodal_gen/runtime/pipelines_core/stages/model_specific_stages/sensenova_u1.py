# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.ug.context import UGContextBundle
from sglang.srt.ug.interleaved import UGGKind, UGGSegmentResult
from sglang.srt.ug.middle import UGMiddleBridge

U1_T2I_CFG_UNCONDITION_ROLE = "u1_t2i_cfg_uncondition"
U1_INTERLEAVE_TEXT_UNCONDITION_ROLE = "u1_interleave_text_uncondition"
U1_EDIT_IMG_CONDITION_ROLE = "u1_edit_img_condition"
U1_EDIT_UNCONDITION_ROLE = "u1_edit_uncondition"


@dataclass(frozen=True, slots=True)
class _SRTGContext:
    session_id: str
    position_count: int
    sidecar_role: str | None = None


class SenseNovaU1PixelFlowGSegmentExecutor:
    """Run SenseNova U1 pixel-flow G through the model-specific diffusion stage."""

    required_g_kind: UGGKind = "pixel_flow"

    def __call__(
        self,
        *,
        bridge: UGMiddleBridge,
        contexts: UGContextBundle,
        batch: Req,
        server_args: ServerArgs,
    ) -> UGGSegmentResult:
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
    bridge: UGMiddleBridge,
    contexts: UGContextBundle,
    batch: Req,
    srt_executor: Any,
) -> tuple[_SRTGContext, _SRTGContext | None, _SRTGContext | None]:
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
    srt_context = _require_srt_context(
        get_position_count,
        session_id,
        "SenseNova U1 pixel-flow has no SRT session position count",
    )
    cfg_img_condition_context = None
    cfg_uncondition_context = None
    sampling_params = batch.sampling_params
    mode = getattr(sampling_params, "ug_generation_mode", None)
    cfg_text_scale = float(getattr(sampling_params, "cfg_text_scale", 1.0))
    cfg_img_scale = float(getattr(sampling_params, "cfg_img_scale", 1.0))
    needs_cfg = not (cfg_text_scale == 1.0 and cfg_img_scale == 1.0)
    needs_img_condition = needs_cfg and (
        cfg_img_scale == 1.0 or cfg_text_scale != cfg_img_scale
    )
    needs_uncondition = needs_cfg and cfg_img_scale != 1.0

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
        if needs_img_condition:
            cfg_img_condition_context = _require_sidecar_srt_context(
                get_position_count,
                session_id,
                edit_img_condition_role,
                "SenseNova U1 edit image CFG requires sidecar SRT position count",
            )
        if needs_uncondition:
            cfg_uncondition_context = _require_sidecar_srt_context(
                get_position_count,
                session_id,
                edit_uncondition_role,
                "SenseNova U1 edit uncondition CFG requires sidecar SRT position count",
            )
    elif mode == "interleave":
        if needs_img_condition:
            cfg_img_condition_context = _require_sidecar_srt_context(
                get_position_count,
                session_id,
                interleave_text_uncondition_role,
                "SenseNova U1 interleave text CFG requires sidecar SRT position count",
            )
        if needs_uncondition:
            cfg_uncondition_context = _require_sidecar_srt_context(
                get_position_count,
                session_id,
                t2i_uncondition_role,
                "SenseNova U1 interleave image CFG requires sidecar SRT position count",
            )
    elif cfg_text_scale > 1.0:
        cfg_img_condition_context = _require_sidecar_srt_context(
            get_position_count,
            session_id,
            t2i_uncondition_role,
            "SenseNova U1 pixel-flow CFG requires sidecar SRT position count",
        )
    return srt_context, cfg_img_condition_context, cfg_uncondition_context


class _SenseNovaU1NativePixelFlowRunner:
    """Model-specific pixel-flow loop; SRT only supplies KV-backed forwards."""

    def __init__(
        self,
        srt_model: Any,
        *,
        forward_batch_provider: Any,
    ) -> None:
        self.srt_model = srt_model
        self.forward_batch_provider = forward_batch_provider

    def generate(
        self,
        *,
        contexts: UGContextBundle,
        batch: Any,
        server_args: Any,
        srt_context: _SRTGContext,
        cfg_img_condition_srt_context: _SRTGContext | None = None,
        cfg_uncondition_srt_context: _SRTGContext | None = None,
    ) -> UGGSegmentResult:
        import torch

        with torch.inference_mode():
            return self._generate_impl(
                contexts=contexts,
                batch=batch,
                server_args=server_args,
                srt_context=srt_context,
                cfg_img_condition_srt_context=cfg_img_condition_srt_context,
                cfg_uncondition_srt_context=cfg_uncondition_srt_context,
            )

    def _generate_impl(
        self,
        *,
        contexts: UGContextBundle,
        batch: Any,
        server_args: Any,
        srt_context: _SRTGContext,
        cfg_img_condition_srt_context: _SRTGContext | None = None,
        cfg_uncondition_srt_context: _SRTGContext | None = None,
    ) -> UGGSegmentResult:
        del server_args
        import numpy as np
        import torch
        from PIL import Image

        if contexts.full.session is None:
            raise ValueError("SenseNova U1 pixel-flow requires contexts.full.session")
        sampling_params = batch.sampling_params
        cfg_text_scale = float(getattr(sampling_params, "cfg_text_scale", 1.0))
        cfg_img_scale = float(getattr(sampling_params, "cfg_img_scale", 1.0))
        needs_cfg = not (cfg_text_scale == 1.0 and cfg_img_scale == 1.0)
        needs_img_condition = needs_cfg and (
            cfg_img_scale == 1.0 or cfg_text_scale != cfg_img_scale
        )
        needs_uncondition = needs_cfg and cfg_img_scale != 1.0
        if (
            needs_img_condition
            and cfg_img_condition_srt_context is None
            and not needs_uncondition
            and cfg_uncondition_srt_context is not None
        ):
            cfg_img_condition_srt_context = cfg_uncondition_srt_context
            cfg_uncondition_srt_context = None
        if needs_img_condition and cfg_img_condition_srt_context is None:
            raise RuntimeError(
                "SenseNova U1 pixel-flow CFG requires an image-condition " "SRT context"
            )
        if needs_uncondition and cfg_uncondition_srt_context is None:
            raise RuntimeError(
                "SenseNova U1 pixel-flow CFG requires an uncondition SRT context"
            )

        image_size = _batch_image_size(batch)
        width, height = image_size
        patch_size = int(self.srt_model.config.vision_config.patch_size)
        merge_size = int(1 / float(self.srt_model.config.downsample_ratio))
        divisor = patch_size * merge_size
        if width % divisor or height % divisor:
            raise ValueError(
                "SenseNova U1 pixel-flow image size must be divisible by "
                f"{divisor}, got {width}x{height}"
            )

        token_h = height // divisor
        token_w = width // divisor
        grid_h = height // patch_size
        grid_w = width // patch_size
        steps = int(getattr(sampling_params, "num_inference_steps", None) or 0)
        if steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {steps}")

        device = _model_device(self.srt_model)
        dtype = _model_dtype(self.srt_model)
        seed = int(getattr(batch, "seed", None) or 0)
        generator = _session_torch_generator(
            contexts.full.metadata,
            seed=seed,
            device=device,
        )
        noise_scale = float(
            _u1_noise_scale_for_image(self.srt_model, grid_h=grid_h, grid_w=grid_w)
        )
        image_prediction = noise_scale * torch.randn(
            (1, 3, height, width),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=device, dtype=torch.long)
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)
        timesteps = _u1_apply_time_schedule(
            self.srt_model,
            timesteps,
            image_seq_len=token_h * token_w,
            timestep_shift=float(getattr(sampling_params, "timestep_shift", 1.0)),
        )
        g_position_start = int(srt_context.position_count)
        indexes_image = _u1_build_t2i_image_indexes(
            token_h=token_h,
            token_w=token_w,
            text_len=g_position_start,
            device=device,
        )
        cfg_img_condition_position_count = None
        indexes_image_img_condition = None
        if cfg_img_condition_srt_context is not None:
            cfg_img_condition_position_count = int(
                cfg_img_condition_srt_context.position_count
            )
            indexes_image_img_condition = _u1_build_t2i_image_indexes(
                token_h=token_h,
                token_w=token_w,
                text_len=cfg_img_condition_position_count,
                device=device,
            )
        cfg_uncondition_position_count = None
        indexes_image_uncondition = None
        if cfg_uncondition_srt_context is not None:
            cfg_uncondition_position_count = int(
                cfg_uncondition_srt_context.position_count
            )
            indexes_image_uncondition = _u1_build_t2i_image_indexes(
                token_h=token_h,
                token_w=token_w,
                text_len=cfg_uncondition_position_count,
                device=device,
            )
        generation_input = {
            "packed_seqlens": torch.tensor(
                [token_h * token_w], dtype=torch.int32, device=device
            ),
            "packed_position_ids": indexes_image,
        }
        prepared = SimpleNamespace(
            generation_input=generation_input,
            srt_session_id=srt_context.session_id,
            srt_sidecar_role=srt_context.sidecar_role,
        )
        prepared_img_condition = None
        if cfg_img_condition_srt_context is not None:
            prepared_img_condition = SimpleNamespace(
                generation_input={
                    "packed_seqlens": generation_input["packed_seqlens"],
                    "packed_position_ids": indexes_image_img_condition,
                },
                srt_session_id=cfg_img_condition_srt_context.session_id,
                srt_sidecar_role=cfg_img_condition_srt_context.sidecar_role,
            )
        prepared_uncondition = None
        if cfg_uncondition_srt_context is not None:
            prepared_uncondition = SimpleNamespace(
                generation_input={
                    "packed_seqlens": generation_input["packed_seqlens"],
                    "packed_position_ids": indexes_image_uncondition,
                },
                srt_session_id=cfg_uncondition_srt_context.session_id,
                srt_sidecar_role=cfg_uncondition_srt_context.sidecar_role,
            )
        cfg_interval = list(getattr(sampling_params, "cfg_interval", [0.0, 1.0]))
        if len(cfg_interval) != 2:
            raise ValueError(
                "SenseNova U1 pixel-flow cfg_interval must have two values"
            )
        cfg_start, cfg_end = float(cfg_interval[0]), float(cfg_interval[1])
        cfg_renorm_type = str(getattr(sampling_params, "cfg_renorm_type", "none"))

        for step_i in range(steps):
            timestep = timesteps[step_i]
            next_timestep = timesteps[step_i + 1]
            use_cfg = (float(timestep) > cfg_start and float(timestep) < cfg_end) or (
                cfg_start == 0.0
            )
            z = _u1_patchify(image_prediction, patch_size * merge_size)
            image_input = _u1_patchify(
                image_prediction,
                patch_size,
                channel_first=True,
            )
            image_embeds = self.srt_model.extract_feature(
                image_input.view(grid_h * grid_w, -1),
                gen_model=True,
                grid_hw=gen_grid_hw,
            ).view(1, token_h * token_w, -1)
            timestep_values = timestep.expand(token_h * token_w)
            timestep_embeddings = self.srt_model.fm_modules["timestep_embedder"](
                timestep_values
            ).view(1, token_h * token_w, -1)
            if getattr(self.srt_model.config, "add_noise_scale_embedding", False):
                noise_values = torch.full_like(
                    timestep_values,
                    noise_scale / float(self.srt_model.config.noise_scale_max_value),
                )
                timestep_embeddings = timestep_embeddings + self.srt_model.fm_modules[
                    "noise_scale_embedder"
                ](noise_values).view(1, token_h * token_w, -1)
            image_embeds = image_embeds + timestep_embeddings

            v_condition = self._predict_v(
                prepared=prepared,
                image_embeds=image_embeds,
                indexes_image=indexes_image,
                timestep=timestep,
                z=z,
            )
            if not use_cfg or not needs_cfg:
                v_pred = v_condition
            elif cfg_img_scale == 1.0:
                v_img_condition = self._predict_v(
                    prepared=prepared_img_condition,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image_img_condition,
                    timestep=timestep,
                    z=z,
                )
                v_pred = v_img_condition + cfg_text_scale * (
                    v_condition - v_img_condition
                )
            elif cfg_text_scale == cfg_img_scale:
                v_uncondition = self._predict_v(
                    prepared=prepared_uncondition,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image_uncondition,
                    timestep=timestep,
                    z=z,
                )
                v_pred = v_uncondition + cfg_text_scale * (v_condition - v_uncondition)
            else:
                v_img_condition = self._predict_v(
                    prepared=prepared_img_condition,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image_img_condition,
                    timestep=timestep,
                    z=z,
                )
                v_uncondition = self._predict_v(
                    prepared=prepared_uncondition,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image_uncondition,
                    timestep=timestep,
                    z=z,
                )
                v_pred = (
                    v_uncondition
                    + cfg_text_scale * (v_condition - v_img_condition)
                    + cfg_img_scale * (v_img_condition - v_uncondition)
                )
            if needs_cfg and use_cfg:
                v_pred = self._apply_cfg_renorm(
                    v_condition=v_condition,
                    v_pred=v_pred,
                    cfg_renorm_type=cfg_renorm_type,
                )

            z = z + (next_timestep - timestep) * v_pred
            image_prediction = _u1_unpatchify(
                z,
                patch_size * merge_size,
                height,
                width,
            )

        array = (
            (image_prediction[0].float() * 0.5 + 0.5)
            .clamp(0, 1)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        image = Image.fromarray((array * 255.0).round().astype(np.uint8), "RGB")
        commit_image = {
            "pixel_values": image_prediction.detach().to(torch.bfloat16).cpu(),
            "value_range": "minus_one_to_one",
            "grid_hw": gen_grid_hw[:1].detach().cpu(),
        }
        return UGGSegmentResult(
            type="image",
            image=image,
            metadata={
                "g_kind": "pixel_flow",
                "native_srt_pixel_flow": True,
                "temporary_g_kv": True,
                "timesteps": steps,
                "seed": seed,
                "width": width,
                "height": height,
                "grid": (token_h, token_w),
                "g_position_start": g_position_start,
                "condition_position_count": g_position_start,
                "cfg_img_condition_position_count": (cfg_img_condition_position_count),
                "cfg_uncondition_position_count": cfg_uncondition_position_count,
                "noise_scale": noise_scale,
                "cfg_text_scale": cfg_text_scale,
                "cfg_img_scale": cfg_img_scale,
                "cfg_renorm_type": (cfg_renorm_type if needs_cfg else "none"),
            },
            commit_image=commit_image,
        )

    def _predict_v(
        self,
        *,
        prepared: Any,
        image_embeds: Any,
        indexes_image: Any,
        timestep: Any,
        z: Any,
    ) -> Any:
        forward_batch_context = self.forward_batch_provider(
            prepared=prepared,
            g_query_embeds=image_embeds,
            timestep=timestep,
        )
        forward_batch = getattr(
            forward_batch_context,
            "forward_batch",
            forward_batch_context,
        )
        try:
            return _predict_u1_pixel_flow_from_srt(
                self.srt_model,
                image_embeds=image_embeds,
                indexes_image=indexes_image,
                forward_batch=forward_batch,
                timestep=timestep,
                z=z,
            )
        finally:
            release = getattr(forward_batch_context, "release", None)
            if callable(release):
                release()

    @staticmethod
    def _apply_cfg_renorm(
        *,
        v_condition: Any,
        v_pred: Any,
        cfg_renorm_type: str,
    ) -> Any:
        if cfg_renorm_type == "none":
            return v_pred
        if cfg_renorm_type == "global":
            norm_v_condition = v_condition.norm(dim=(1, 2), keepdim=True)
            norm_v_cfg = v_pred.norm(dim=(1, 2), keepdim=True)
        elif cfg_renorm_type == "channel":
            norm_v_condition = v_condition.norm(dim=-1, keepdim=True)
            norm_v_cfg = v_pred.norm(dim=-1, keepdim=True)
        else:
            raise ValueError(
                "Unsupported SenseNova U1 pixel-flow CFG renorm type: "
                f"{cfg_renorm_type}"
            )
        scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
        return v_pred * scale


def _batch_image_size(batch: Any) -> tuple[int, int]:
    sampling_params = batch.sampling_params
    height = _first_int(
        getattr(batch, "height", None),
        getattr(sampling_params, "height", None),
        default=1024,
    )
    width = _first_int(
        getattr(batch, "width", None),
        getattr(sampling_params, "width", None),
        default=1024,
    )
    return width, height


def _require_srt_context(
    get_position_count: Any,
    session_id: str,
    message: str,
    *,
    sidecar_role: str | None = None,
) -> _SRTGContext:
    position_count = get_position_count(session_id, sidecar_role=sidecar_role)
    if position_count is None:
        suffix = f" sidecar {sidecar_role}" if sidecar_role is not None else ""
        raise RuntimeError(f"{message} for session {session_id}{suffix}")
    return _SRTGContext(
        session_id=session_id,
        sidecar_role=sidecar_role,
        position_count=int(position_count),
    )


def _require_sidecar_srt_context(
    get_position_count: Any,
    session_id: str,
    role: str,
    message: str,
) -> _SRTGContext:
    return _require_srt_context(
        get_position_count,
        session_id,
        message,
        sidecar_role=role,
    )


def _u1_patchify(images: Any, patch_size: int, *, channel_first: bool = False) -> Any:
    import torch

    h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
    x = images.reshape(images.shape[0], 3, h, patch_size, w, patch_size)
    if channel_first:
        x = torch.einsum("nchpwq->nhwcpq", x)
    else:
        x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(images.shape[0], h * w, patch_size**2 * 3)


def _u1_unpatchify(
    x: Any,
    patch_size: int,
    h: int | None = None,
    w: int | None = None,
) -> Any:
    import torch

    if h is None or w is None:
        h = w = int(x.shape[1] ** 0.5)
    else:
        h = h // patch_size
        w = w // patch_size
    x = x.reshape(x.shape[0], h, w, patch_size, patch_size, 3)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(x.shape[0], 3, h * patch_size, w * patch_size)


def _u1_build_t2i_image_indexes(
    *,
    token_h: int,
    token_w: int,
    text_len: int,
    device: Any,
) -> Any:
    import torch

    t_image = torch.full(
        (token_h * token_w,),
        int(text_len),
        dtype=torch.long,
        device=device,
    )
    idx = torch.arange(token_h * token_w, device=device, dtype=torch.long)
    h_image = idx // token_w
    w_image = idx % token_w
    return torch.stack([t_image, h_image, w_image], dim=0)


def _u1_apply_time_schedule(
    model: Any,
    timesteps: Any,
    *,
    image_seq_len: int,
    timestep_shift: float,
) -> Any:
    import torch

    sigma = 1 - timesteps
    cfg = model.config
    schedule = str(cfg.time_schedule)
    if timestep_shift != 1:
        schedule = "standard"
    if schedule == "standard":
        shift = float(timestep_shift)
        sigma = shift * sigma / (1 + (shift - 1) * sigma)
    elif schedule == "dynamic":
        mu = _u1_calculate_dynamic_mu(model, image_seq_len)
        mu_t = timesteps.new_tensor(mu)
        time_shift_type = str(cfg.time_shift_type)
        if time_shift_type == "exponential":
            shift = torch.exp(mu_t)
            sigma = shift * sigma / (1 + (shift - 1) * sigma)
        elif time_shift_type == "linear":
            sigma = mu_t / (mu_t + (1 / sigma - 1))
        else:
            raise ValueError(
                f"Unsupported SenseNova U1 time_shift_type: {time_shift_type}"
            )
    else:
        raise ValueError(f"Unsupported SenseNova U1 time_schedule: {schedule}")
    return 1 - sigma


def _u1_noise_scale_for_image(model: Any, *, grid_h: int, grid_w: int) -> float:
    import math

    cfg = model.config
    merge_size = int(1 / float(cfg.downsample_ratio))
    noise_scale = float(cfg.noise_scale)
    noise_scale_mode = str(cfg.noise_scale_mode)
    if noise_scale_mode in {"resolution", "dynamic", "dynamic_sqrt"}:
        base = float(cfg.noise_scale_base_image_seq_len)
        scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / base)
        noise_scale = scale * float(cfg.noise_scale)
        if noise_scale_mode == "dynamic_sqrt":
            noise_scale = math.sqrt(noise_scale)
    return min(noise_scale, float(cfg.noise_scale_max_value))


def _u1_calculate_dynamic_mu(model: Any, image_seq_len: int) -> float:
    cfg = model.config
    denom = int(cfg.max_image_seq_len) - int(cfg.base_image_seq_len)
    if denom == 0:
        return float(cfg.base_shift)
    slope = (float(cfg.max_shift) - float(cfg.base_shift)) / denom
    bias = float(cfg.base_shift) - slope * int(cfg.base_image_seq_len)
    return float(image_seq_len) * slope + bias


def _predict_u1_pixel_flow_from_srt(
    model: Any,
    *,
    image_embeds: Any,
    indexes_image: Any,
    forward_batch: Any,
    timestep: Any,
    z: Any,
) -> Any:
    batch_size, image_token_num = image_embeds.shape[:2]
    hidden_states = model.language_model.forward_u1_gen_embeds(
        input_embeds=image_embeds.reshape(-1, image_embeds.shape[-1]),
        positions=indexes_image,
        forward_batch=forward_batch,
    ).view(batch_size, image_token_num, -1)
    x_pred = model.fm_modules["fm_head"](hidden_states).view(
        batch_size,
        image_token_num,
        -1,
    )

    t = timestep.to(device=z.device, dtype=z.dtype)
    return (x_pred - z) / (1 - t).clamp_min(float(getattr(model.config, "t_eps", 0.02)))


def _session_torch_generator(
    metadata: dict[str, Any],
    *,
    seed: int,
    device: Any,
) -> Any:
    import torch

    key = "_u1_pixel_flow_generator"
    device_str = str(device)
    state = metadata.get(key)
    if (
        isinstance(state, dict)
        and state.get("seed") == int(seed)
        and state.get("device") == device_str
        and isinstance(state.get("generator"), torch.Generator)
    ):
        return state["generator"]
    generator = torch.Generator(device=device).manual_seed(int(seed))
    metadata[key] = {
        "seed": int(seed),
        "device": device_str,
        "generator": generator,
    }
    return generator


def _first_int(*values: Any, default: int) -> int:
    for value in values:
        if value is not None:
            return int(value)
    return int(default)


def _model_device(srt_model: Any) -> Any:
    vision_model = getattr(srt_model, "vision_model", None)
    device = getattr(vision_model, "device", None)
    if device is not None:
        return device
    return next(srt_model.parameters()).device


def _model_dtype(srt_model: Any) -> Any:
    vision_model = getattr(srt_model, "vision_model", None)
    dtype = getattr(vision_model, "dtype", None)
    if dtype is not None:
        return dtype
    return next(srt_model.parameters()).dtype
