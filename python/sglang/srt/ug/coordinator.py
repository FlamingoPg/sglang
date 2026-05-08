# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable

GSegmentCallable = Callable[..., Any]


class UGInterleaveCoordinator:
    """Small owner of the UG interleave control loop.

    SRT remains the session/KV capability provider through ``bridge``.
    multimodal_gen remains the stateless G segment provider through
    ``g_segment_executor``. This coordinator only decides when to run U, G, and
    commit generated segments.
    """

    def __init__(
        self,
        *,
        bridge: Any,
        g_segment_executor: GSegmentCallable,
    ) -> None:
        self.bridge = bridge
        self.g_segment_executor = g_segment_executor

    def run_generation(
        self,
        *,
        batch: Any,
        contexts: Any,
        server_args: Any,
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        mode = metadata.get("mode", "interleave")
        if mode == "interleave":
            return self.run_interleave(
                batch=batch,
                contexts=contexts,
                server_args=server_args,
                metadata=metadata,
            )

        generated_segment = self.run_g_segment(
            batch=batch,
            contexts=contexts,
            server_args=server_args,
        )
        return [self._image_output_segment(generated_segment)]

    def run_interleave(
        self,
        *,
        batch: Any,
        contexts: Any,
        server_args: Any,
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        output_segments = list(batch.extra.get("ug_pre_image_segments", []))
        max_images = _resolve_positive_metadata_int(
            metadata,
            "max_interleave_images",
            default=1,
        )
        max_text_segments = _resolve_positive_metadata_int(
            metadata,
            "max_interleave_text_segments",
            default=1,
        )

        for _ in range(max_images):
            generated_segment = self.run_g_segment(
                batch=batch,
                contexts=contexts,
                server_args=server_args,
            )
            output_segments.append(self._image_output_segment(generated_segment))
            self.bridge.commit_generated_segment(
                contexts=contexts,
                segment=generated_segment,
            )

            next_image_requested = False
            for _ in range(max_text_segments):
                post_segment = self.bridge.continue_u_decode(contexts=contexts)
                if post_segment.type == "text":
                    output_segments.append(self._text_output_segment(post_segment))
                    continue
                if post_segment.type == "image_marker":
                    next_image_requested = True
                    break
                if post_segment.type == "done":
                    return output_segments
                raise ValueError(
                    "UG interleave expected U text, image marker, or done, "
                    f"got {post_segment.type}"
                )
            if not next_image_requested:
                break
        return output_segments

    def run_g_segment(
        self,
        *,
        batch: Any,
        contexts: Any,
        server_args: Any,
    ) -> Any:
        return self.bridge.run_g_segment(
            contexts=contexts,
            executor=lambda context_ops: self.g_segment_executor(
                context_ops=context_ops,
                batch=batch,
                server_args=server_args,
            ),
        )

    @staticmethod
    def _image_output_segment(segment: Any) -> dict[str, Any]:
        return {
            "type": "image",
            "image": segment.image,
            "metadata": dict(getattr(segment, "metadata", {}) or {}),
        }

    @staticmethod
    def _text_output_segment(segment: Any) -> dict[str, Any]:
        output = {"type": "text", "text": segment.text or ""}
        token_ids = getattr(segment, "token_ids", ())
        if token_ids:
            output["metadata"] = {
                "token_ids": [int(token_id) for token_id in token_ids]
            }
        return output


def _resolve_positive_metadata_int(
    metadata: dict[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    value = int(metadata.get(key, default))
    if value <= 0:
        raise ValueError(f"UG metadata {key} must be positive, got {value}")
    return value
