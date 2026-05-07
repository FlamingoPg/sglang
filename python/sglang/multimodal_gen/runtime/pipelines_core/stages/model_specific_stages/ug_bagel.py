# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import cast

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.sample.ug import (
    UGSamplingParams,
    get_ug_explicit_sampling_fields,
    mark_ug_explicit_sampling_fields,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.ug.context import UGContextBundle
from sglang.srt.ug.denoiser import UGLatentFlowMiddleBridge, UGMiddleBridge
from sglang.srt.ug.interleaved import UGGSegmentResult

BAGEL_OFFICIAL_T2I_DEFAULTS = {
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 1.5,
    "cfg_interval": [0.4, 1.0],
    "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "global",
    "timestep_shift": 3.0,
    "num_inference_steps": 50,
}

BAGEL_OFFICIAL_EDIT_DEFAULTS = {
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 2.0,
    "cfg_interval": [0.0, 1.0],
    "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "text_channel",
    "timestep_shift": 3.0,
    "num_inference_steps": 50,
}


def apply_bagel_official_sampling_defaults(
    params: UGSamplingParams | None,
    *,
    mode: str,
    has_input_image: bool,
    explicit_fields: set[str] | None = None,
) -> UGSamplingParams | None:
    """Apply BAGEL demo defaults without clobbering user-provided fields."""

    if params is None or mode == "vlm":
        return params

    explicit = get_ug_explicit_sampling_fields(params)
    if explicit_fields is not None:
        explicit |= set(explicit_fields)

    if mode == "edit" or (mode == "interleave" and has_input_image):
        defaults = BAGEL_OFFICIAL_EDIT_DEFAULTS
    else:
        defaults = BAGEL_OFFICIAL_T2I_DEFAULTS

    for name, value in defaults.items():
        if name not in explicit:
            setattr(params, name, deepcopy(value))

    params._validate_ug_fields()
    mark_ug_explicit_sampling_fields(params, explicit)
    return params


class BAGELLatentFlowGSegmentExecutor:
    """BAGEL G mechanics for the UG middle protocol.

    The generic UG stages only pass the SRT-owned U context into this executor.
    Latent preparation, schedule stepping, velocity calls, and image decoding
    stay here on the BAGEL/diffusion side of the boundary.
    """

    required_g_kind = "latent_flow"

    def __call__(
        self,
        *,
        bridge: UGMiddleBridge,
        contexts: UGContextBundle,
        batch: Req,
        server_args: ServerArgs,
    ) -> UGGSegmentResult:
        if getattr(bridge, "g_kind", None) != "latent_flow":
            raise ValueError(
                "BAGEL latent-flow executor requires g_kind='latent_flow', got "
                f"{getattr(bridge, 'g_kind', None)!r}"
            )
        latent_bridge = cast(UGLatentFlowMiddleBridge, bridge)
        self._prepare_latents(
            bridge=latent_bridge,
            contexts=contexts,
            batch=batch,
            server_args=server_args,
        )
        self._denoise(bridge=latent_bridge, contexts=contexts, batch=batch)
        image = self._decode_image(
            bridge=latent_bridge,
            contexts=contexts,
            batch=batch,
        )
        metadata = {"g_kind": "latent_flow"}
        if "ug_latent_shape" in batch.extra:
            metadata["latent_shape"] = batch.extra["ug_latent_shape"]
        return UGGSegmentResult(type="image", image=image, metadata=metadata)

    @staticmethod
    def _prepare_latents(
        *,
        bridge: UGLatentFlowMiddleBridge,
        contexts: UGContextBundle,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        del server_args
        prepared = bridge.prepare_g_latents(
            contexts=contexts,
            sampling_params=batch.sampling_params,
            seed=batch.seed,
        )
        if prepared is not None:
            batch.latents = prepared.latent_tokens
            batch.extra["ug_latent_position_ids"] = prepared.latent_position_ids
            batch.extra["ug_latent_shape"] = prepared.latent_shape
            return

        raise RuntimeError(
            "BAGEL latent-flow requires backend-prepared VAE latents from "
            "SRT-owned U context"
        )

    @staticmethod
    def _denoise(
        *,
        bridge: UGLatentFlowMiddleBridge,
        contexts: UGContextBundle,
        batch: Req,
    ) -> None:
        params = batch.sampling_params
        x_t = batch.latents
        if x_t is None:
            raise ValueError("UG G segment requires latents from latent preparation")
        num_steps = int(params.num_inference_steps)
        if num_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {num_steps}")

        timesteps = torch.linspace(1, 0, num_steps, device=x_t.device)
        shift = float(params.timestep_shift)
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        trajectory_latents = []
        trajectory_timesteps = []

        for i, timestep in enumerate(timesteps[:-1]):
            trajectory_latents.append(x_t)
            trajectory_timesteps.append(timestep)
            velocity = bridge.predict_g_velocity(
                contexts=contexts,
                latent_tokens=x_t,
                timestep=timestep.reshape(1),
                latent_position_ids=batch.extra["ug_latent_position_ids"],
                sampling_params=params,
            )
            x_t = x_t - velocity.to(x_t) * dts[i].to(x_t)

        batch.latents = x_t
        if batch.return_trajectory_latents:
            if trajectory_latents:
                batch.trajectory_latents = torch.stack(trajectory_latents)
                batch.trajectory_timesteps = torch.stack(trajectory_timesteps)
            else:
                batch.trajectory_latents = x_t[:0]
                batch.trajectory_timesteps = timesteps[:0]

    @staticmethod
    def _decode_image(
        *,
        bridge: UGLatentFlowMiddleBridge,
        contexts: UGContextBundle,
        batch: Req,
    ) -> Image.Image:
        image = bridge.decode_g_latents(
            contexts=contexts,
            latent_tokens=batch.latents,
            sampling_params=batch.sampling_params,
        )
        if image is None:
            raise RuntimeError(
                "BAGEL latent-flow requires backend image decode from SRT-owned "
                "U context"
            )
        if isinstance(image, Image.Image):
            return image
        array = np.asarray(image)
        if array.ndim == 4:
            array = array[0]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)
