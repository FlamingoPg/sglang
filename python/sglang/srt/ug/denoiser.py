# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Protocol

import torch

from sglang.srt.ug.context import UGContextBundle, UGContextHandle


class UGDenoiserBridge(Protocol):
    def build_contexts(
        self, *, prompt: str | list[str] | None, image: Any | None
    ) -> UGContextBundle:
        ...

    def predict_velocity(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
        latent_position_ids: torch.Tensor,
        sampling_params: Any,
    ) -> torch.Tensor:
        ...

    def release_contexts(self, contexts: UGContextBundle) -> None:
        ...


class FakeUGDenoiserBridge:
    def build_contexts(
        self, *, prompt: str | list[str] | None, image: Any | None
    ) -> UGContextBundle:
        prompt_text = " ".join(prompt) if isinstance(prompt, list) else prompt or ""
        text_tokens = len(prompt_text.split())
        image_tokens = 2 if image is not None else 0
        return UGContextBundle(
            full=UGContextHandle("full", text_tokens + image_tokens),
            text_cfg=UGContextHandle("text_cfg", image_tokens),
            image_cfg=UGContextHandle("image_cfg", text_tokens),
        )

    def predict_velocity(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
        latent_position_ids: torch.Tensor,
        sampling_params: Any,
    ) -> torch.Tensor:
        del latent_position_ids, sampling_params
        scale = 1.0 + contexts.full.token_count * 0.01
        return latent_tokens + scale * timestep.reshape(-1, 1, 1).to(latent_tokens)

    def release_contexts(self, contexts: UGContextBundle) -> None:
        del contexts
