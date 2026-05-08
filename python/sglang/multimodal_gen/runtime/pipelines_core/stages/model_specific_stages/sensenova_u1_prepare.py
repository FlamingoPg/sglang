# SPDX-License-Identifier: Apache-2.0

from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_types import (
    SRTGContext,
    U1PixelFlowPrepared,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_utils import (
    batch_image_size,
    build_pixel_flow_forward_context,
    model_device,
    model_dtype,
    resolve_pixel_flow_cfg,
    session_torch_generator,
    u1_apply_time_schedule,
    u1_noise_scale_for_image,
)

class SenseNovaU1PixelFlowPrepareStage:
    def __init__(self, srt_model: Any) -> None:
        self.srt_model = srt_model

    def forward(
        self,
        *,
        contexts: Any,
        batch: Any,
        srt_context: SRTGContext,
        cfg_img_condition_srt_context: SRTGContext | None = None,
        cfg_uncondition_srt_context: SRTGContext | None = None,
    ) -> U1PixelFlowPrepared:
        import torch

        if contexts.full.session is None:
            raise ValueError("SenseNova U1 pixel-flow requires contexts.full.session")
        sampling_params = batch.sampling_params
        cfg = resolve_pixel_flow_cfg(sampling_params)
        if (
            cfg.needs_img_condition
            and cfg_img_condition_srt_context is None
            and not cfg.needs_uncondition
            and cfg_uncondition_srt_context is not None
        ):
            cfg_img_condition_srt_context = cfg_uncondition_srt_context
            cfg_uncondition_srt_context = None
        if cfg.needs_img_condition and cfg_img_condition_srt_context is None:
            raise RuntimeError(
                "SenseNova U1 pixel-flow CFG requires an image-condition SRT context"
            )
        if cfg.needs_uncondition and cfg_uncondition_srt_context is None:
            raise RuntimeError(
                "SenseNova U1 pixel-flow CFG requires an uncondition SRT context"
            )

        width, height = batch_image_size(batch)
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

        device = model_device(self.srt_model)
        dtype = model_dtype(self.srt_model)
        seed = int(getattr(batch, "seed", None) or 0)
        generator = session_torch_generator(
            contexts.full.metadata,
            seed=seed,
            device=device,
        )
        noise_scale = float(
            u1_noise_scale_for_image(self.srt_model, grid_h=grid_h, grid_w=grid_w)
        )
        image_prediction = noise_scale * torch.randn(
            (1, 3, height, width),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=device, dtype=torch.long)
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)
        timesteps = u1_apply_time_schedule(
            self.srt_model,
            timesteps,
            image_seq_len=token_h * token_w,
            timestep_shift=float(getattr(sampling_params, "timestep_shift", 1.0)),
        )
        packed_seqlens = torch.tensor(
            [token_h * token_w], dtype=torch.int32, device=device
        )
        condition = build_pixel_flow_forward_context(
            srt_context,
            token_h=token_h,
            token_w=token_w,
            packed_seqlens=packed_seqlens,
            device=device,
        )
        img_condition = None
        if cfg_img_condition_srt_context is not None:
            img_condition = build_pixel_flow_forward_context(
                cfg_img_condition_srt_context,
                token_h=token_h,
                token_w=token_w,
                packed_seqlens=packed_seqlens,
                device=device,
            )
        uncondition = None
        if cfg_uncondition_srt_context is not None:
            uncondition = build_pixel_flow_forward_context(
                cfg_uncondition_srt_context,
                token_h=token_h,
                token_w=token_w,
                packed_seqlens=packed_seqlens,
                device=device,
            )

        return U1PixelFlowPrepared(
            width=width,
            height=height,
            patch_size=patch_size,
            merge_size=merge_size,
            token_h=token_h,
            token_w=token_w,
            grid_h=grid_h,
            grid_w=grid_w,
            steps=steps,
            seed=seed,
            noise_scale=noise_scale,
            image_prediction=image_prediction,
            gen_grid_hw=gen_grid_hw,
            timesteps=timesteps,
            cfg=cfg,
            condition=condition,
            img_condition=img_condition,
            uncondition=uncondition,
        )
