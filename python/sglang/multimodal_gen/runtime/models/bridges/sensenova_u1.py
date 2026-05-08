# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

# SenseNova U1 keeps image generation inside the unified model. These helpers
# adapt SRT-owned U context to U1's pixel-flow query path.

U1_T2I_CFG_UNCONDITION_ROLE = "u1_t2i_cfg_uncondition"
U1_INTERLEAVE_TEXT_UNCONDITION_ROLE = "u1_interleave_text_uncondition"
U1_EDIT_IMG_CONDITION_ROLE = "u1_edit_img_condition"
U1_EDIT_UNCONDITION_ROLE = "u1_edit_uncondition"


@dataclass(frozen=True, slots=True)
class U1GContext:
    session_id: str
    position_count: int
    sidecar_role: str | None = None


@dataclass(frozen=True, slots=True)
class U1PixelFlowCFG:
    text_scale: float
    img_scale: float
    needs_cfg: bool
    needs_img_condition: bool
    needs_uncondition: bool
    start: float
    end: float
    renorm_type: str


@dataclass(frozen=True, slots=True)
class U1PixelFlowForwardContext:
    prepared: Any
    indexes_image: Any
    position_count: int


@dataclass(slots=True)
class U1PixelFlowPrepared:
    width: int
    height: int
    patch_size: int
    merge_size: int
    token_h: int
    token_w: int
    grid_h: int
    grid_w: int
    steps: int
    seed: int
    noise_scale: float
    image_prediction: Any
    gen_grid_hw: Any
    timesteps: Any
    cfg: U1PixelFlowCFG
    condition: U1PixelFlowForwardContext
    img_condition: U1PixelFlowForwardContext | None
    uncondition: U1PixelFlowForwardContext | None


@dataclass(frozen=True, slots=True)
class U1GeneratedSegment:
    type: str
    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    commit_image: Any | None = None


def resolve_pixel_flow_cfg(sampling_params: Any) -> U1PixelFlowCFG:
    text_scale = float(getattr(sampling_params, "cfg_text_scale", 1.0))
    img_scale = float(getattr(sampling_params, "cfg_img_scale", 1.0))
    needs_cfg = not (text_scale == 1.0 and img_scale == 1.0)
    cfg_interval = list(getattr(sampling_params, "cfg_interval", [0.0, 1.0]))
    if len(cfg_interval) != 2:
        raise ValueError("SenseNova U1 pixel-flow cfg_interval must have two values")
    return U1PixelFlowCFG(
        text_scale=text_scale,
        img_scale=img_scale,
        needs_cfg=needs_cfg,
        needs_img_condition=needs_cfg and (img_scale == 1.0 or text_scale != img_scale),
        needs_uncondition=needs_cfg and img_scale != 1.0,
        start=float(cfg_interval[0]),
        end=float(cfg_interval[1]),
        renorm_type=str(getattr(sampling_params, "cfg_renorm_type", "none")),
    )


def build_pixel_flow_forward_context(
    context: U1GContext,
    *,
    token_h: int,
    token_w: int,
    packed_seqlens: Any,
    device: Any,
) -> U1PixelFlowForwardContext:
    position_count = int(context.position_count)
    indexes_image = u1_build_t2i_image_indexes(
        token_h=token_h,
        token_w=token_w,
        text_len=position_count,
        device=device,
    )
    prepared = SimpleNamespace(
        generation_input={
            "packed_seqlens": packed_seqlens,
            "packed_position_ids": indexes_image,
        },
        session_id=context.session_id,
        sidecar_role=context.sidecar_role,
    )
    return U1PixelFlowForwardContext(
        prepared=prepared,
        indexes_image=indexes_image,
        position_count=position_count,
    )


def should_apply_cfg(cfg: U1PixelFlowCFG, timestep: Any) -> bool:
    return (float(timestep) > cfg.start and float(timestep) < cfg.end) or (
        cfg.start == 0.0
    )


def require_forward_context(
    context: U1PixelFlowForwardContext | None,
) -> U1PixelFlowForwardContext:
    if context is None:
        raise RuntimeError("SenseNova U1 pixel-flow CFG forward context is missing")
    return context


def forward_context_position(context: U1PixelFlowForwardContext | None) -> int | None:
    if context is None:
        return None
    return context.position_count


def batch_image_size(batch: Any) -> tuple[int, int]:
    sampling_params = batch.sampling_params
    height = first_int(
        getattr(batch, "height", None),
        getattr(sampling_params, "height", None),
        default=1024,
    )
    width = first_int(
        getattr(batch, "width", None),
        getattr(sampling_params, "width", None),
        default=1024,
    )
    return width, height


def u1_patchify(images: Any, patch_size: int, *, channel_first: bool = False) -> Any:
    import torch

    h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
    x = images.reshape(images.shape[0], 3, h, patch_size, w, patch_size)
    if channel_first:
        x = torch.einsum("nchpwq->nhwcpq", x)
    else:
        x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(images.shape[0], h * w, patch_size**2 * 3)


def u1_unpatchify(
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


def u1_build_t2i_image_indexes(
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


def u1_apply_time_schedule(
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
        mu = u1_calculate_dynamic_mu(model, image_seq_len)
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


def u1_noise_scale_for_image(model: Any, *, grid_h: int, grid_w: int) -> float:
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


def u1_calculate_dynamic_mu(model: Any, image_seq_len: int) -> float:
    cfg = model.config
    denom = int(cfg.max_image_seq_len) - int(cfg.base_image_seq_len)
    if denom == 0:
        return float(cfg.base_shift)
    slope = (float(cfg.max_shift) - float(cfg.base_shift)) / denom
    bias = float(cfg.base_shift) - slope * int(cfg.base_image_seq_len)
    return float(image_seq_len) * slope + bias


def predict_u1_pixel_flow_from_srt(
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


def session_torch_generator(
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


def first_int(*values: Any, default: int) -> int:
    for value in values:
        if value is not None:
            return int(value)
    return int(default)


def model_device(model: Any) -> Any:
    vision_model = getattr(model, "vision_model", None)
    device = getattr(vision_model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def model_dtype(model: Any) -> Any:
    vision_model = getattr(model, "vision_model", None)
    dtype = getattr(vision_model, "dtype", None)
    if dtype is not None:
        return dtype
    return next(model.parameters()).dtype
