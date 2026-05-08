# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
from PIL import Image

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class ContextConditionedImageGenerationStage(PipelineStage):
    """Generic image generation stage fed by an external context handle.

    This stage is intentionally model-agnostic: it does not know how the context
    was created, what model owns it, or whether the generator uses latents,
    pixels, or another internal representation. It only normalizes the
    multimodal_gen-side lifecycle for a single generated image segment.
    """

    def __init__(
        self,
        segment_generator: Any,
        *,
        context_ops_key: str,
        output_extra_key: str = "generated_segment",
    ) -> None:
        super().__init__()
        self.segment_generator = segment_generator
        self.context_ops_key = context_ops_key
        self.output_extra_key = output_extra_key

    @property
    def role_affinity(self):
        return RoleType.DENOISER

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        context_ops = batch.extra.get(self.context_ops_key)
        if context_ops is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires batch.extra"
                f"[{self.context_ops_key!r}]"
            )

        segment = self._generate_segment(
            context_ops=context_ops,
            batch=batch,
            server_args=server_args,
        )
        if getattr(segment, "type", None) != "image":
            raise ValueError(
                "Context-conditioned image generation expected an image segment, "
                f"got {getattr(segment, 'type', None)!r}"
            )

        batch.extra[self.output_extra_key] = segment
        batch.output = image_to_numpy_batch(segment.image)
        return batch

    def _generate_segment(
        self,
        *,
        context_ops: Any,
        batch: Req,
        server_args: ServerArgs,
    ) -> Any:
        generate = getattr(self.segment_generator, "generate_segment", None)
        if callable(generate):
            return generate(
                context_ops=context_ops,
                batch=batch,
                server_args=server_args,
            )
        return self.segment_generator(
            context_ops=context_ops,
            batch=batch,
            server_args=server_args,
        )


def image_to_numpy_batch(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[None, ...]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array
