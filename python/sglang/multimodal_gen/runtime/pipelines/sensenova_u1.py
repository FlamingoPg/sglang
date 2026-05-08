# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
from PIL import Image

from sglang.multimodal_gen.runtime.pipelines_core import ComposedPipelineBase
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1 import (
    SenseNovaU1PixelFlowGSegmentExecutor,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class SenseNovaU1Pipeline(ComposedPipelineBase):
    """Stateless SenseNova U1 G segment generator.

    UG interleave orchestration is intentionally outside multimodal_gen. This
    pipeline owns only the model-private pixel-flow generation call.
    """

    pipeline_name = "SenseNovaU1Pipeline"
    _required_config_modules: list[str] = []

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del server_args
        modules = dict(loaded_modules or {})
        if "g_segment_executor" not in modules:
            modules["g_segment_executor"] = SenseNovaU1PixelFlowGSegmentExecutor()
        return modules

    def create_pipeline_stages(self, server_args: ServerArgs):
        del server_args
        return None

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        context_ops = batch.extra.get("sensenova_u1_context_ops")
        if context_ops is None:
            raise RuntimeError(
                "SenseNovaU1Pipeline is a stateless G segment generator. "
                "Use the internal UG coordinator to prepare G context ops and "
                "call generate_segment()."
            )
        segment = self.generate_segment(
            context_ops=context_ops,
            batch=batch,
            server_args=server_args,
        )
        batch.extra["sensenova_u1_generated_segment"] = segment
        batch.output = _image_to_numpy_batch(segment.image)
        return batch

    def generate_segment(
        self,
        *,
        context_ops: Any,
        batch: Req,
        server_args: ServerArgs,
    ) -> Any:
        executor = self.get_module("g_segment_executor")
        return executor(
            context_ops=context_ops,
            batch=batch,
            server_args=server_args,
        )

    def forward_interleaved(self, *args, **kwargs):
        raise RuntimeError(
            "SenseNova U1 interleave is owned by the internal UG coordinator, "
            "not by multimodal_gen. Use sglang.srt.ug.sensenova_u1 entrypoints."
        )

    def forward_interleaved_batch(self, *args, **kwargs):
        raise RuntimeError(
            "SenseNova U1 interleave batching is owned by the internal UG coordinator, "
            "not by multimodal_gen."
        )

    def forward_vlm(self, *args, **kwargs):
        raise RuntimeError(
            "SenseNova U1 VLM is owned by the internal UG coordinator, "
            "not by multimodal_gen."
        )

    def forward_vlm_batch(self, *args, **kwargs):
        raise RuntimeError(
            "SenseNova U1 VLM batching is owned by the internal UG coordinator, "
            "not by multimodal_gen."
        )


EntryClass = SenseNovaU1Pipeline


def _image_to_numpy_batch(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[None, ...]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array
