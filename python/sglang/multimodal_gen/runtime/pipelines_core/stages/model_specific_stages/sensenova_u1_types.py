# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

U1_T2I_CFG_UNCONDITION_ROLE = "u1_t2i_cfg_uncondition"
U1_INTERLEAVE_TEXT_UNCONDITION_ROLE = "u1_interleave_text_uncondition"
U1_EDIT_IMG_CONDITION_ROLE = "u1_edit_img_condition"
U1_EDIT_UNCONDITION_ROLE = "u1_edit_uncondition"


@dataclass(frozen=True, slots=True)
class SRTGContext:
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
