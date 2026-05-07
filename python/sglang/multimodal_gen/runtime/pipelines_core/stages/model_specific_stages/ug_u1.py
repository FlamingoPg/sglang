# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.ug.context import UGContextBundle
from sglang.srt.ug.middle import UGMiddleBridge
from sglang.srt.ug.interleaved import UGGKind, UGGSegmentResult


class U1PixelFlowGSegmentExecutor:
    """SenseNova U1 pixel-flow G executor skeleton."""

    required_g_kind: UGGKind = "pixel_flow"
    patch_size: int = 16

    def __call__(
        self,
        *,
        bridge: UGMiddleBridge,
        contexts: UGContextBundle,
        batch: Req,
        server_args: ServerArgs,
    ) -> UGGSegmentResult:
        if getattr(bridge, "g_kind", None) != self.required_g_kind:
            raise ValueError(
                "SenseNova U1 pixel-flow executor requires g_kind='pixel_flow', got "
                f"{getattr(bridge, 'g_kind', None)!r}"
            )
        run_native = getattr(bridge, "run_native_pixel_flow_g_segment", None)
        if not callable(run_native):
            raise RuntimeError(
                "SenseNova U1 pixel-flow requires a native SRT pixel-flow bridge"
            )
        native_segment = run_native(
            contexts=contexts,
            batch=batch,
            server_args=server_args,
        )
        if native_segment is None:
            raise RuntimeError("SenseNova U1 native pixel-flow did not return an image")
        return native_segment
