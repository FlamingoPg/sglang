# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class UGContextHandle:
    request_id: str
    token_count: int
    kv_indices: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UGContextBundle:
    full: UGContextHandle
    text_cfg: UGContextHandle
    image_cfg: UGContextHandle
