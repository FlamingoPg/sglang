# SPDX-License-Identifier: Apache-2.0

from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.model_specific.sensenova_u1 import (
    batch_image_size,
    build_pixel_flow_forward_context,
    model_device,
    model_dtype,
    resolve_pixel_flow_cfg,
    session_torch_generator,
    U1GContext,
    u1_apply_time_schedule,
    u1_noise_scale_for_image,
    U1PixelFlowPrepared,
)


class SenseNovaU1PixelFlowPreparer:
    def __init__(self, model: Any) -> None:
        self.model = model

    def forward(
        self,
        *,
        context_metadata: dict[str, Any],
        batch: Any,
        u1_context: U1GContext,
        cfg_img_condition_u1_context: U1GContext | None = None,
        cfg_uncondition_u1_context: U1GContext | None = None,
    ) -> U1PixelFlowPrepared:
        import torch

        sampling_params = batch.sampling_params
        cfg = resolve_pixel_flow_cfg(sampling_params)
        if (
            cfg.needs_img_condition
            and cfg_img_condition_u1_context is None
            and not cfg.needs_uncondition
            and cfg_uncondition_u1_context is not None
        ):
            cfg_img_condition_u1_context = cfg_uncondition_u1_context
            cfg_uncondition_u1_context = None
        if cfg.needs_img_condition and cfg_img_condition_u1_context is None:
            raise RuntimeError(
                "SenseNova U1 pixel-flow CFG requires an image-condition context"
            )
        if cfg.needs_uncondition and cfg_uncondition_u1_context is None:
            raise RuntimeError(
                "SenseNova U1 pixel-flow CFG requires an uncondition context"
            )

        width, height = batch_image_size(batch)
        patch_size = int(self.model.config.vision_config.patch_size)
        merge_size = int(1 / float(self.model.config.downsample_ratio))
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

        device = model_device(self.model)
        dtype = model_dtype(self.model)
        seed = int(getattr(batch, "seed", None) or 0)
        generator = session_torch_generator(
            context_metadata,
            seed=seed,
            device=device,
        )
        noise_scale = float(
            u1_noise_scale_for_image(self.model, grid_h=grid_h, grid_w=grid_w)
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
            self.model,
            timesteps,
            image_seq_len=token_h * token_w,
            timestep_shift=float(getattr(sampling_params, "timestep_shift", 1.0)),
        )
        packed_seqlens = torch.tensor(
            [token_h * token_w], dtype=torch.int32, device=device
        )
        condition = build_pixel_flow_forward_context(
            u1_context,
            token_h=token_h,
            token_w=token_w,
            packed_seqlens=packed_seqlens,
            device=device,
        )
        img_condition = None
        if cfg_img_condition_u1_context is not None:
            img_condition = build_pixel_flow_forward_context(
                cfg_img_condition_u1_context,
                token_h=token_h,
                token_w=token_w,
                packed_seqlens=packed_seqlens,
                device=device,
            )
        uncondition = None
        if cfg_uncondition_u1_context is not None:
            uncondition = build_pixel_flow_forward_context(
                cfg_uncondition_u1_context,
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
