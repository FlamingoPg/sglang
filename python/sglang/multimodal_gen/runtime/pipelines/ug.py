# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core import ComposedPipelineBase
from sglang.multimodal_gen.runtime.pipelines_core.stages.ug import (
    UGContextStage,
    UGDecodeStage,
    UGDenoiseStage,
    UGLatentStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.ug.denoiser import SRTBackedUGDenoiserBridge, UGDenoiserBridge


def _load_ug_bridge(model_path: str) -> UGDenoiserBridge:
    model_path_lower = model_path.lower()
    if "fake-ug" in model_path_lower:
        return SRTBackedUGDenoiserBridge()
    if "bagel" in model_path_lower:
        raise NotImplementedError(
            "BAGEL UG bridge is not wired yet. Use sglang-internal/fake-ug "
            "for diffusion runtime tests until the SRT UG model path lands."
        )
    raise ValueError(f"Unsupported UG model path: {model_path}")


class UGPipeline(ComposedPipelineBase):
    pipeline_name = "UGPipeline"
    _required_config_modules: list[str] = []

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if loaded_modules and "ug_bridge" in loaded_modules:
            return loaded_modules
        return {"ug_bridge": _load_ug_bridge(self.model_path)}

    def create_pipeline_stages(self, server_args: ServerArgs):
        bridge = self.get_module("ug_bridge")
        self.add_stage(UGContextStage(bridge))
        self.add_stage(UGLatentStage())
        self.add_stage(UGDenoiseStage(bridge))
        self.add_stage(UGDecodeStage(bridge))


EntryClass = UGPipeline
