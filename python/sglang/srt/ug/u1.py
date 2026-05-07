# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from sglang.srt.ug.adapter import UGModelAppendImageResult, UGModelPrefillResult
from sglang.srt.ug.context import UGContextBundle
from sglang.srt.ug.denoiser import (
    SRTBackedUGMiddleBridge,
    UGGSegmentExecutor,
)
from sglang.srt.ug.interleaved import UGGKind, UGGSegmentResult
from sglang.srt.ug.runtime import (
    UGDecodeResult,
    UGInterleavedMessage,
    UGSessionRuntime,
    UGSegmentState,
    UGSRTPreparedInput,
    UGVLMTextGenerationResult,
)

U1_IMG_START_TOKEN = "<img>"
U1_IMG_END_TOKEN = "</img>"
U1_IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
U1_IMAGE_PLACEHOLDER = "<image>"
U1_T2I_CFG_UNCONDITION_ROLE = "u1_t2i_cfg_uncondition"
U1_INTERLEAVE_TEXT_UNCONDITION_ROLE = "u1_interleave_text_uncondition"
U1_EDIT_IMG_CONDITION_ROLE = "u1_edit_img_condition"
U1_EDIT_UNCONDITION_ROLE = "u1_edit_uncondition"
U1_IMAGENET_MEAN = (0.485, 0.456, 0.406)
U1_IMAGENET_STD = (0.229, 0.224, 0.225)
U1_SYSTEM_MESSAGE_FOR_GEN = (
    "You are an image generation and editing assistant that accurately understands "
    "and executes user intent.\n\n"
    "You support two modes:\n\n"
    "1. Think Mode:\n"
    "If the task requires reasoning, you MUST start with a <think></think> block. "
    "Put all reasoning inside the block using plain text. DO NOT include any image "
    "tags. Keep it reasonable and directly useful for producing the final image.\n\n"
    "2. Non-Think Mode:\n"
    "If no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\n"
    "A. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or "
    "transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the "
    "rendered text in the user's description, not the language of the prompt. If no "
    "language is specified, use the user's input language."
)
U1_INTERLEAVE_SYSTEM_MESSAGE = (
    "You are a multimodal assistant capable of reasoning with both text and "
    "images. You support two modes:\n\n"
    "Think Mode: When reasoning is needed, you MUST start with a "
    "<think></think> block and place all reasoning inside it. You MUST "
    "interleave text with generated images using tags like <image1>, <image2>. "
    "Images can ONLY be generated between <think> and </think>, and may be "
    "referenced in the final answer.\n\n"
    "Non-Think Mode: When no reasoning is needed, directly provide the answer "
    "without reasoning. Do not use tags like <image1>, <image2>; present any "
    "images naturally alongside the text.\n\n"
    "After the think block, always provide a concise, user-facing final answer. "
    "The answer may include text, images, or both. Match the user's language in "
    "both reasoning and the final answer."
)


@dataclass(frozen=True, slots=True)
class U1VLMBackendResult:
    text: str
    token_ids: tuple[int, ...] = ()
    next_token_ids: tuple[int, ...] = ()
    position_ids: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class U1VLMBackend(Protocol):
    def generate_text(
        self,
        *,
        messages: list[UGInterleavedMessage],
        max_new_tokens: int,
    ) -> U1VLMBackendResult: ...


class U1SubprocessVLMBackend:
    """Opt-in external VLM runner for U1 parity.

    This backend keeps the official/compatibility implementation out of the
    SGLang runtime process. It is a parity bridge, not native SRT ModelRunner
    execution.
    """

    def __init__(
        self,
        *,
        python: str | Path,
        repo: str | Path,
        model_path: str | Path,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_backend: str = "sdpa",
        timeout: int = 600,
        cuda_visible_devices: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.python = Path(python)
        self.repo = Path(repo)
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.attn_backend = attn_backend
        self.timeout = int(timeout)
        self.cuda_visible_devices = cuda_visible_devices
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def generate_text(
        self,
        *,
        messages: list[UGInterleavedMessage],
        max_new_tokens: int,
    ) -> U1VLMBackendResult:
        image_path = _first_u1_image_path(messages)
        question = _u1_question_text(messages)
        output_dir = self._make_output_dir()
        output_path = output_dir / "u1_vlm_candidate.txt"
        cmd = [
            str(self.python),
            str(self.repo / "examples/vqa/inference.py"),
            "--model_path",
            self.model_path,
            "--image",
            str(image_path),
            "--question",
            question,
            "--output",
            str(output_path),
            "--max_new_tokens",
            str(int(max_new_tokens)),
            "--device",
            self.device,
            "--dtype",
            self.dtype,
            "--attn_backend",
            self.attn_backend,
        ]
        run_env = os.environ.copy()
        if self.cuda_visible_devices is not None:
            run_env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        completed = subprocess.run(
            cmd,
            cwd=self.repo,
            env=run_env,
            text=True,
            capture_output=True,
            timeout=self.timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "U1 VLM subprocess backend failed: "
                f"returncode={completed.returncode}, stderr={_tail(completed.stderr)}"
            )
        if not output_path.exists():
            raise RuntimeError("U1 VLM subprocess backend did not write output text")
        return U1VLMBackendResult(
            text=output_path.read_text(),
            metadata={
                "backend": "external_subprocess",
                "native_srt_model_runner": False,
                "command": cmd,
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
            },
        )

    def _make_output_dir(self) -> Path:
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return self.output_dir
        return Path(tempfile.mkdtemp(prefix="u1-vlm-backend-"))


def is_sensenova_u1_ug_model(
    model_path: str | None,
    model_id: str | None = None,
) -> bool:
    identifier = f"{model_path or ''} {model_id or ''}".lower()
    return "sensenova-u1" in identifier or "sensenova_u1" in identifier


class U1UGModelAdapter:
    """SenseNova U1 UG adapter shell for the UG middle protocol.

    U1 uses pixel-flow G mechanics, so it intentionally does not expose BAGEL
    latent-flow methods such as velocity prediction or latent decode.
    """

    g_kind: UGGKind = "pixel_flow"

    bos_token_id = 1
    text_token_base = 1000
    image_token_id = 31001
    generated_image_token_id = 31003

    def __init__(
        self,
        *,
        vlm_backend: U1VLMBackend | None = None,
        native_tokenizer: Any | None = None,
    ) -> None:
        self.observed_u_forwards: list[dict[str, Any]] = []
        self._pending_segments_by_session: dict[str, list[dict[str, Any]]] = {}
        self._messages_by_session: dict[str, list[UGInterleavedMessage]] = {}
        self.vlm_backend = vlm_backend
        self.native_tokenizer = native_tokenizer
        self.include_t2i_cfg_uncondition = False
        self.include_interleave_text_uncondition = False
        self.include_edit_img_condition = False
        self.include_edit_uncondition = False
        self.native_generation_mode: str | None = None
        self.native_interleave_think_mode = False

    def prepare_srt_u_interleaved_inputs(
        self,
        *,
        session: Any,
        messages: list[UGInterleavedMessage],
        state: Any,
    ) -> list[UGSRTPreparedInput] | None:
        if state != UGSegmentState.U_PREFILL or self.native_tokenizer is None:
            return None
        has_image = any(message.type == "image" for message in messages)
        has_text = any(message.type == "text" for message in messages)
        if not has_text:
            return None
        if self.native_generation_mode == "interleave":
            prepared = []
            if self.include_interleave_text_uncondition:
                prepared.append(
                    build_u1_native_interleave_text_uncondition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        messages=messages,
                        session=session,
                    )
                )
            if self.include_t2i_cfg_uncondition:
                prepared.append(
                    build_u1_native_t2i_cfg_uncondition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        session=session,
                    )
                )
            prepared.append(
                build_u1_native_interleave_prepared_input(
                    tokenizer=self.native_tokenizer,
                    messages=messages,
                    session=session,
                    think_mode=self.native_interleave_think_mode,
                )
            )
            return prepared
        if not has_image:
            prepared = []
            if self.include_t2i_cfg_uncondition:
                prepared.append(
                    build_u1_native_t2i_cfg_uncondition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        session=session,
                    )
                )
            prepared.append(
                build_u1_native_t2i_prepared_input(
                    tokenizer=self.native_tokenizer,
                    messages=messages,
                    session=session,
                )
            )
            return prepared
        if self.native_generation_mode == "edit":
            prepared = []
            if self.include_edit_img_condition:
                prepared.append(
                    build_u1_native_edit_img_condition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        messages=messages,
                        session=session,
                    )
                )
            if self.include_edit_uncondition:
                prepared.append(
                    build_u1_native_edit_uncondition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        session=session,
                    )
                )
            prepared.append(
                build_u1_native_edit_prepared_input(
                    tokenizer=self.native_tokenizer,
                    messages=messages,
                    session=session,
                )
            )
            return prepared
        return [
            build_u1_native_vlm_prepared_input(
                tokenizer=self.native_tokenizer,
                messages=messages,
                session=session,
            )
        ]

    def prepare_srt_u_message_inputs(
        self,
        *,
        session: Any,
        message: Any,
        state: Any,
    ) -> list[UGSRTPreparedInput] | None:
        if message.type == "text":
            return [self._prepare_text_input(session=session, message=message)]
        if message.type == "image":
            if (
                state == UGSegmentState.APPEND_IMAGE
                and self.native_tokenizer is not None
            ):
                return [
                    build_u1_native_generated_image_commit_prepared_input(
                        tokenizer=self.native_tokenizer,
                        image=message.content,
                        session=session,
                    )
                ]
            return [
                self._prepare_image_input(
                    session=session,
                    message=message,
                    generated_image_commit=state == UGSegmentState.APPEND_IMAGE,
                )
            ]
        return None

    def observe_srt_u_forward(
        self,
        *,
        session: Any,
        request: Any,
        messages: list[Any],
    ) -> None:
        del session
        self.observed_u_forwards.append(
            {
                "request_id": request.request_id,
                "state": request.state,
                "origin_input_len": request.origin_input_len,
                "metadata": request.metadata,
                "message_types": [message.type for message in messages],
            }
        )

    def prefill_interleaved(
        self,
        *,
        session: Any,
        messages: list[Any],
    ) -> UGModelPrefillResult:
        try:
            self._remember_session_messages(session, messages)
            return UGModelPrefillResult(
                added_tokens=self._added_tokens_from_srt_session_view(session, messages)
            )
        finally:
            self._clear_pending_segments(session)

    def decode_next_segment(self, *, session: Any) -> Any:
        if self._has_generated_image_commit(session):
            return UGDecodeResult(type="text", text="u1_pixel_flow_text_after_image")
        return UGDecodeResult(type="image_marker")

    def decode_next_segment_from_runtime(self, *, runtime: Any, session: Any) -> Any:
        session_view = self._runtime_session_view(runtime=runtime, session=session)
        u1_state = (
            (getattr(session_view, "metadata", {}) or {})
            .get("ug_model_state", {})
            .get("u1", {})
        )
        if bool(u1_state.get("native_interleave_prompt")):
            return self._decode_native_interleave_next_segment(
                runtime=runtime,
                session=session,
                u1_state=u1_state,
            )
        if not self._has_generated_image_commit(session_view):
            return UGDecodeResult(type="image_marker")
        if self.native_tokenizer is None or runtime is None:
            return UGDecodeResult(type="text", text="u1_pixel_flow_text_after_image")
        if getattr(runtime, "srt_request_executor", None) is None:
            return UGDecodeResult(type="text", text="u1_pixel_flow_text_after_image")
        max_new_tokens = max(
            1,
            int(getattr(runtime, "srt_u_decode_max_new_tokens", 0) or 0),
        )
        decoded = runtime.decode_text(
            session,
            max_new_tokens=max_new_tokens,
            greedy=True,
        )
        return UGDecodeResult(
            type="text",
            text=decoded.text,
            token_ids=tuple(int(token_id) for token_id in decoded.output_ids),
        )

    def _decode_native_interleave_next_segment(
        self,
        *,
        runtime: Any,
        session: Any,
        u1_state: dict[str, Any],
    ) -> UGDecodeResult:
        if self.native_tokenizer is None or runtime is None:
            return UGDecodeResult(type="done")
        if bool(u1_state.get("interleave_pending_image_marker")):
            self._merge_runtime_u1_state(
                runtime,
                session,
                {
                    "interleave_pending_image_marker": False,
                    "open_image_marker": True,
                },
            )
            return UGDecodeResult(type="image_marker")
        if getattr(runtime, "srt_request_executor", None) is None:
            return UGDecodeResult(type="done")

        img_start_id = _u1_token_id(self.native_tokenizer, U1_IMG_START_TOKEN)
        eos_token_ids = _u1_eos_token_ids(self.native_tokenizer)
        max_new_tokens = max(
            1,
            int(getattr(runtime, "srt_u_decode_max_new_tokens", 0) or 0),
        )
        generated_text_ids: list[int] = []
        current_position = int(
            u1_state.get(
                "g_position_start",
                getattr(session, "context_length", 0) or 0,
            )
            or 0
        )

        for _ in range(max_new_tokens):
            decoded = runtime.decode_text(
                session,
                max_new_tokens=1,
                decode_position_id=current_position,
                greedy=True,
                model_state_updates={
                    "u1": {
                        "last_segment_type": "interleave",
                        "last_source": "native_interleave_decode",
                        "native_interleave_prompt": True,
                        "g_position_start": current_position,
                    }
                },
            )
            if not decoded.output_ids:
                self._merge_runtime_u1_state(
                    runtime,
                    session,
                    {"g_position_start": current_position},
                )
                break
            token_id = int(decoded.output_ids[-1])
            current_position += 1
            if token_id == img_start_id:
                state_updates = {
                    "last_segment_type": "interleave",
                    "last_source": "native_interleave_image_marker",
                    "native_interleave_prompt": True,
                    "open_image_marker": True,
                    "interleave_pending_image_marker": bool(generated_text_ids),
                    "g_position_start": current_position,
                }
                commit_decode_token = getattr(
                    runtime, "commit_u_decode_input_token", None
                )
                if callable(commit_decode_token):
                    session = commit_decode_token(
                        session,
                        token_id=token_id,
                        position_id=current_position - 1,
                        model_state_updates={"u1": state_updates},
                    )
                else:
                    self._merge_runtime_u1_state(runtime, session, state_updates)
                self._append_interleave_text_uncondition_marker(
                    runtime=runtime,
                    session=session,
                )
                if generated_text_ids:
                    return UGDecodeResult(
                        type="text",
                        text=_u1_decode_token_ids(
                            self.native_tokenizer,
                            generated_text_ids,
                        ),
                        token_ids=tuple(generated_text_ids),
                    )
                return UGDecodeResult(type="image_marker")
            if token_id in eos_token_ids:
                self._merge_runtime_u1_state(
                    runtime,
                    session,
                    {
                        "last_segment_type": "interleave",
                        "last_source": "native_interleave_eos",
                        "native_interleave_prompt": True,
                        "g_position_start": current_position,
                    },
                )
                if generated_text_ids:
                    return UGDecodeResult(
                        type="text",
                        text=_u1_decode_token_ids(
                            self.native_tokenizer,
                            generated_text_ids,
                        ),
                        token_ids=tuple(generated_text_ids),
                    )
                return UGDecodeResult(type="done")
            generated_text_ids.append(token_id)

        self._merge_runtime_u1_state(
            runtime,
            session,
            {
                "last_segment_type": "interleave",
                "last_source": "native_interleave_decode",
                "native_interleave_prompt": True,
                "g_position_start": current_position,
            },
        )
        if generated_text_ids:
            return UGDecodeResult(
                type="text",
                text=_u1_decode_token_ids(self.native_tokenizer, generated_text_ids),
                token_ids=tuple(generated_text_ids),
            )
        return UGDecodeResult(type="done")

    def _append_interleave_text_uncondition_marker(
        self,
        *,
        runtime: Any,
        session: Any,
    ) -> None:
        if self.native_tokenizer is None:
            return
        append_sidecar = getattr(runtime, "append_srt_sidecar_prepared_input", None)
        get_sidecar_state = getattr(runtime, "get_srt_sidecar_model_state", None)
        if not callable(append_sidecar) or not callable(get_sidecar_state):
            return
        sidecar_state = get_sidecar_state(
            session,
            U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        )
        u1_state = (sidecar_state or {}).get("u1") or {}
        if not u1_state:
            return
        if bool(u1_state.get("open_image_marker")):
            return
        logical_position = u1_state.get("g_position_start")
        if logical_position is None:
            return
        prepared = build_u1_native_interleave_text_uncondition_marker_prepared_input(
            tokenizer=self.native_tokenizer,
            session=session,
            logical_position=int(logical_position),
        )
        append_sidecar(session, prepared, state=UGSegmentState.U_DECODE)

    @staticmethod
    def _merge_runtime_u1_state(
        runtime: Any,
        session: Any,
        updates: dict[str, Any],
    ) -> None:
        record_for = getattr(runtime, "_record_for", None)
        merge = getattr(runtime, "_merge_ug_model_state_updates", None)
        if not callable(record_for) or not callable(merge):
            return
        try:
            record = record_for(session)
        except Exception:
            return
        merge(record, {"u1": dict(updates)})

    @staticmethod
    def _runtime_session_view(*, runtime: Any, session: Any) -> Any:
        if getattr(session, "metadata", None) is not None:
            return session
        metadata: dict[str, Any] = {}
        get_debug_counters = getattr(runtime, "get_debug_counters", None)
        if callable(get_debug_counters):
            try:
                counters = get_debug_counters(session)
                metadata["ug_model_state"] = counters.get("ug_model_state", {})
                metadata["srt_last_u_decode_output_ids"] = counters.get(
                    "srt_last_u_decode_output_ids",
                    (),
                )
            except Exception:
                metadata = {}
        return SimpleNamespace(handle=session, metadata=metadata)

    def decode_vlm_text(
        self,
        *,
        runtime: Any,
        session: Any,
        max_new_tokens: int,
    ) -> Any:
        if self.vlm_backend is None:
            if runtime is None:
                raise _not_wired()
            decoded = runtime.decode_text(
                session,
                max_new_tokens=max_new_tokens,
                greedy=True,
            )
            return UGVLMTextGenerationResult(
                session=decoded.session,
                text=decoded.text,
                token_ids=decoded.input_ids,
                next_token_ids=decoded.output_ids,
                position_ids=decoded.position_ids,
            )
        messages = self._messages_for_session(session)
        result = self.vlm_backend.generate_text(
            messages=messages,
            max_new_tokens=max_new_tokens,
        )
        return UGVLMTextGenerationResult(
            session=session,
            text=result.text,
            token_ids=result.token_ids,
            next_token_ids=result.next_token_ids,
            position_ids=result.position_ids,
        )

    def append_generated_image(
        self,
        *,
        session: Any,
        image: Any | None,
    ) -> UGModelAppendImageResult:
        del image
        try:
            return UGModelAppendImageResult(
                added_tokens=self._added_tokens_from_srt_session_view(session, [])
            )
        finally:
            self._clear_pending_segments(session)

    def close_session(self, *, session_id: str) -> None:
        self._messages_by_session.pop(str(session_id), None)
        self._pending_segments_by_session.pop(str(session_id), None)

    def _prepare_text_input(
        self,
        *,
        session: Any,
        message: UGInterleavedMessage,
    ) -> UGSRTPreparedInput:
        text = str(message.content)
        text_token_ids = self._text_token_ids(text)
        input_ids = [self.bos_token_id] + text_token_ids
        token_indices = list(range(1, len(input_ids)))
        metadata = self._segment_metadata(
            session=session,
            segment_type="text",
            source="user_text",
            token_indices=token_indices,
            attention="causal",
            generated_image_commit=False,
        )
        return UGSRTPreparedInput(
            input_ids=input_ids,
            input_text=text,
            messages=[message],
            mot_text_token_indices=token_indices,
            adapter_metadata=metadata,
        )

    def _prepare_image_input(
        self,
        *,
        session: Any,
        message: UGInterleavedMessage,
        generated_image_commit: bool,
    ) -> UGSRTPreparedInput:
        image_token_id = (
            self.generated_image_token_id
            if generated_image_commit
            else self.image_token_id
        )
        input_ids = [self.bos_token_id, image_token_id, image_token_id + 1]
        token_indices = [1, 2]
        source = "generated_image" if generated_image_commit else "input_image"
        metadata = self._segment_metadata(
            session=session,
            segment_type="image",
            source=source,
            token_indices=token_indices,
            attention="hybrid",
            generated_image_commit=generated_image_commit,
        )
        return UGSRTPreparedInput(
            input_ids=input_ids,
            input_text=f"<u1:{source}>",
            messages=[message],
            non_causal_query_attention=True,
            mot_image_token_indices=token_indices,
            adapter_metadata=metadata,
        )

    def _segment_metadata(
        self,
        *,
        session: Any,
        segment_type: str,
        source: str,
        token_indices: list[int],
        attention: str,
        generated_image_commit: bool,
    ) -> dict[str, Any]:
        u1_segment = {
            "segment_type": segment_type,
            "source": source,
            "token_indices": list(token_indices),
            "attention_rows": [
                {
                    "kind": segment_type,
                    "attention": attention,
                    "start": min(token_indices) if token_indices else 0,
                    "end": (max(token_indices) + 1) if token_indices else 0,
                }
            ],
            "generated_image_commit": bool(generated_image_commit),
        }
        previous_segments = self._previous_u1_segments(session)
        u1_state = {
            "segments": previous_segments + [u1_segment],
            "last_segment_type": segment_type,
            "last_source": source,
            "last_generated_image_commit": bool(generated_image_commit),
        }
        self._remember_pending_segment(session, u1_segment)
        return {
            "u1": u1_segment,
            "ug_model_state_updates": {"u1": u1_state},
        }

    def _previous_u1_segments(self, session: Any) -> list[dict[str, Any]]:
        session_metadata = getattr(session, "metadata", {}) or {}
        model_state = session_metadata.get("ug_model_state") or {}
        u1_state = model_state.get("u1") or {}
        segments = [dict(segment) for segment in u1_state.get("segments", [])]
        session_key = self._session_key(session)
        if session_key is not None:
            segments.extend(
                dict(segment)
                for segment in self._pending_segments_by_session.get(session_key, [])
            )
        return segments

    def _has_generated_image_commit(self, session: Any) -> bool:
        return any(
            bool(segment.get("generated_image_commit"))
            for segment in self._previous_u1_segments(session)
        )

    def _remember_pending_segment(self, session: Any, segment: dict[str, Any]) -> None:
        session_key = self._session_key(session)
        if session_key is None:
            return
        self._pending_segments_by_session.setdefault(session_key, []).append(
            dict(segment)
        )

    def _clear_pending_segments(self, session: Any) -> None:
        session_key = self._session_key(session)
        if session_key is not None:
            self._pending_segments_by_session.pop(session_key, None)

    def _remember_session_messages(
        self,
        session: Any,
        messages: list[UGInterleavedMessage],
    ) -> None:
        session_key = self._session_key(session)
        if session_key is None:
            return
        stored = self._messages_by_session.setdefault(session_key, [])
        stored.extend(messages)

    def _messages_for_session(self, session: Any) -> list[UGInterleavedMessage]:
        session_key = getattr(session, "session_id", None)
        if session_key is None:
            session_key = self._session_key(session)
        if session_key is None:
            raise RuntimeError("U1 VLM decode requires a UG session id")
        messages = self._messages_by_session.get(str(session_key), [])
        if not messages:
            raise RuntimeError(
                f"U1 VLM decode has no messages for session {session_key}"
            )
        return list(messages)

    @staticmethod
    def _session_key(session: Any) -> str | None:
        handle = getattr(session, "handle", None)
        session_id = getattr(handle, "session_id", None)
        return str(session_id) if session_id is not None else None

    def _text_token_ids(self, text: str) -> list[int]:
        words = text.split() or [text]
        return [
            self.text_token_base + (sum(word.encode("utf-8")) % 1000) for word in words
        ]

    def _added_tokens_from_srt_session_view(
        self,
        session: Any,
        messages: list[Any],
    ) -> int:
        handle = getattr(session, "handle", None)
        previous_length = int(getattr(handle, "context_length", 0) or 0)
        srt_length = int(getattr(session, "srt_last_origin_input_len", 0) or 0)
        if srt_length > previous_length:
            return srt_length - previous_length
        return sum(self._message_token_count(message) for message in messages)

    def _message_token_count(self, message: Any) -> int:
        if message.type == "text":
            return 1 + len(self._text_token_ids(str(message.content)))
        if message.type == "image":
            return 3
        return 0


class U1SRTBackedUGMiddleBridge:
    """Pixel-flow U1 bridge shell backed by the common SRT UG session runtime."""

    g_kind: UGGKind = "pixel_flow"

    def __init__(
        self,
        runtime: UGSessionRuntime,
        *,
        max_pre_image_decode_steps: int = 128,
    ) -> None:
        self.runtime = runtime
        self._bridge = SRTBackedUGMiddleBridge(
            runtime,
            max_pre_image_decode_steps=max_pre_image_decode_steps,
        )

    def prepare_u_context(
        self,
        *,
        prompt: str | list[str] | None,
        image: Any | None,
        think: bool = False,
        think_max_new_tokens: int | None = None,
        sampling_params: Any | None = None,
    ) -> UGContextBundle:
        with self._temporary_generation_settings(sampling_params):
            bridge_think = (
                False
                if getattr(sampling_params, "ug_generation_mode", None) == "interleave"
                else think
            )
            contexts = self._bridge.prepare_u_context(
                prompt=prompt,
                image=image,
                think=bridge_think,
                think_max_new_tokens=think_max_new_tokens,
                sampling_params=sampling_params,
            )
        self._attach_u1_context_metadata(contexts)
        return contexts

    def prepare_u_context_from_messages(
        self,
        *,
        messages: list[UGInterleavedMessage | dict[str, Any]],
        think: bool = False,
        think_max_new_tokens: int | None = None,
        sampling_params: Any | None = None,
    ) -> UGContextBundle:
        with self._temporary_generation_settings(sampling_params):
            bridge_think = (
                False
                if getattr(sampling_params, "ug_generation_mode", None) == "interleave"
                else think
            )
            contexts = self._bridge.prepare_u_context_from_messages(
                messages=messages,
                think=bridge_think,
                think_max_new_tokens=think_max_new_tokens,
                sampling_params=sampling_params,
            )
        self._attach_u1_context_metadata(contexts)
        return contexts

    @contextmanager
    def _temporary_generation_settings(self, sampling_params: Any | None):
        adapter = getattr(self.runtime.model_runner, "adapter", None)
        old_cfg = getattr(adapter, "include_t2i_cfg_uncondition", False)
        old_interleave_text_uncondition = getattr(
            adapter,
            "include_interleave_text_uncondition",
            False,
        )
        old_edit_img_condition = getattr(adapter, "include_edit_img_condition", False)
        old_edit_uncondition = getattr(adapter, "include_edit_uncondition", False)
        old_mode = getattr(adapter, "native_generation_mode", None)
        old_interleave_think_mode = getattr(
            adapter,
            "native_interleave_think_mode",
            False,
        )
        mode = getattr(sampling_params, "ug_generation_mode", None)
        if adapter is not None:
            needs_cfg = _u1_needs_any_cfg(sampling_params)
            cfg_text_scale = float(getattr(sampling_params, "cfg_text_scale", 1.0))
            cfg_img_scale = float(getattr(sampling_params, "cfg_img_scale", 1.0))
            adapter.include_t2i_cfg_uncondition = (
                _u1_needs_text_cfg(sampling_params)
                and mode not in {"edit", "interleave"}
            ) or (mode == "interleave" and cfg_img_scale != 1.0)
            adapter.include_interleave_text_uncondition = (
                mode == "interleave" and _u1_needs_text_cfg(sampling_params)
            )
            adapter.include_edit_img_condition = (
                mode == "edit"
                and needs_cfg
                and (cfg_img_scale == 1.0 or cfg_text_scale != cfg_img_scale)
            )
            adapter.include_edit_uncondition = (
                mode == "edit" and needs_cfg and cfg_img_scale != 1.0
            )
            adapter.native_generation_mode = mode
            adapter.native_interleave_think_mode = bool(
                getattr(sampling_params, "think", False)
            )
        try:
            yield
        finally:
            if adapter is not None:
                adapter.include_t2i_cfg_uncondition = old_cfg
                adapter.include_interleave_text_uncondition = (
                    old_interleave_text_uncondition
                )
                adapter.include_edit_img_condition = old_edit_img_condition
                adapter.include_edit_uncondition = old_edit_uncondition
                adapter.native_generation_mode = old_mode
                adapter.native_interleave_think_mode = old_interleave_think_mode

    def _attach_u1_context_metadata(self, contexts: UGContextBundle) -> None:
        session = contexts.full.session
        if session is None:
            return
        counters = self.runtime.get_debug_counters(session)
        u1_state = counters.get("ug_model_state", {}).get("u1", {})
        g_position_start = u1_state.get("g_position_start")
        if g_position_start is not None:
            contexts.full.metadata["u1_g_position_start"] = int(g_position_start)

    def run_g_segment(
        self,
        *,
        contexts: UGContextBundle,
        executor: UGGSegmentExecutor,
    ) -> Any:
        return self._bridge.run_g_segment(contexts=contexts, executor=executor)

    def run_native_pixel_flow_g_segment(
        self,
        *,
        contexts: UGContextBundle,
        batch: Any,
        server_args: Any,
    ) -> UGGSegmentResult | None:
        srt_executor = self.runtime.srt_request_executor
        create_executor = getattr(
            srt_executor,
            "create_u1_native_srt_pixel_flow_executor",
            None,
        )
        if not callable(create_executor):
            return None
        if contexts.full.session is None:
            raise ValueError("U1 native pixel-flow requires a SRT UG session")
        get_binding = getattr(srt_executor, "get_latest_ug_session_token_binding", None)
        if not callable(get_binding):
            raise RuntimeError(
                "U1 native pixel-flow requires latest SRT session token binding"
            )
        binding = get_binding(contexts.full.session.session_id)
        if binding is None:
            raise RuntimeError(
                "U1 native pixel-flow has no SRT KV token binding for session "
                f"{contexts.full.session.session_id}"
            )
        cfg_img_condition_binding = None
        cfg_uncondition_binding = None
        sampling_params = batch.sampling_params
        mode = getattr(sampling_params, "ug_generation_mode", None)
        cfg_text_scale = float(getattr(sampling_params, "cfg_text_scale", 1.0))
        cfg_img_scale = float(getattr(sampling_params, "cfg_img_scale", 1.0))
        needs_cfg = _u1_needs_any_cfg(sampling_params)
        needs_img_condition = needs_cfg and (
            cfg_img_scale == 1.0 or cfg_text_scale != cfg_img_scale
        )
        needs_uncondition = needs_cfg and cfg_img_scale != 1.0
        if mode == "edit":
            if needs_img_condition:
                sidecar_session_id = (
                    f"{contexts.full.session.session_id}:"
                    f"{U1_EDIT_IMG_CONDITION_ROLE}"
                )
                cfg_img_condition_binding = get_binding(sidecar_session_id)
                if cfg_img_condition_binding is None:
                    raise RuntimeError(
                        "U1 native edit image CFG requires sidecar SRT KV token "
                        f"binding for session {sidecar_session_id}"
                    )
            if needs_uncondition:
                sidecar_session_id = (
                    f"{contexts.full.session.session_id}:" f"{U1_EDIT_UNCONDITION_ROLE}"
                )
                cfg_uncondition_binding = get_binding(sidecar_session_id)
                if cfg_uncondition_binding is None:
                    raise RuntimeError(
                        "U1 native edit uncondition CFG requires sidecar SRT KV "
                        f"token binding for session {sidecar_session_id}"
                    )
        elif mode == "interleave":
            if needs_img_condition:
                sidecar_session_id = (
                    f"{contexts.full.session.session_id}:"
                    f"{U1_INTERLEAVE_TEXT_UNCONDITION_ROLE}"
                )
                cfg_img_condition_binding = get_binding(sidecar_session_id)
                if cfg_img_condition_binding is None:
                    raise RuntimeError(
                        "U1 native interleave text CFG requires sidecar SRT KV "
                        f"token binding for session {sidecar_session_id}"
                    )
            if needs_uncondition:
                sidecar_session_id = (
                    f"{contexts.full.session.session_id}:"
                    f"{U1_T2I_CFG_UNCONDITION_ROLE}"
                )
                cfg_uncondition_binding = get_binding(sidecar_session_id)
                if cfg_uncondition_binding is None:
                    raise RuntimeError(
                        "U1 native interleave image CFG requires sidecar SRT KV "
                        f"token binding for session {sidecar_session_id}"
                    )
        elif cfg_text_scale > 1.0:
            sidecar_session_id = (
                f"{contexts.full.session.session_id}:" f"{U1_T2I_CFG_UNCONDITION_ROLE}"
            )
            cfg_img_condition_binding = get_binding(sidecar_session_id)
            if cfg_img_condition_binding is None:
                raise RuntimeError(
                    "U1 native pixel-flow CFG requires sidecar SRT KV token "
                    f"binding for session {sidecar_session_id}"
                )
        native_executor = create_executor()
        debug_dump_dir = getattr(self, "debug_tensor_dump_dir", None)
        if debug_dump_dir is not None:
            native_executor.debug_tensor_dump_dir = debug_dump_dir
            native_executor.debug_tensor_dump_max_g_calls = int(
                getattr(self, "debug_tensor_dump_max_g_calls", 32)
            )
            native_executor.debug_g_sublayer_layers = tuple(
                int(layer_id)
                for layer_id in getattr(self, "debug_g_sublayer_layers", (0,))
            )
        return native_executor.generate(
            contexts=contexts,
            batch=batch,
            server_args=server_args,
            srt_kv_token_binding=binding,
            cfg_img_condition_srt_kv_token_binding=cfg_img_condition_binding,
            cfg_uncondition_srt_kv_token_binding=cfg_uncondition_binding,
        )

    def commit_generated_segment(
        self,
        *,
        contexts: UGContextBundle,
        segment: Any,
    ) -> None:
        self._bridge.commit_generated_segment(contexts=contexts, segment=segment)
        self._commit_interleave_text_uncondition_sidecar(
            contexts=contexts,
            segment=segment,
        )

    def release(self, contexts: UGContextBundle) -> None:
        self._bridge.release(contexts)

    def continue_u_decode(self, *, contexts: UGContextBundle) -> UGDecodeResult:
        result = self._bridge.continue_u_decode(contexts=contexts)
        self._attach_u1_context_metadata(contexts)
        return result

    def _commit_interleave_text_uncondition_sidecar(
        self,
        *,
        contexts: UGContextBundle,
        segment: Any,
    ) -> None:
        adapter = getattr(self.runtime.model_runner, "adapter", None)
        tokenizer = getattr(adapter, "native_tokenizer", None)
        if tokenizer is None or contexts.full.session is None:
            return
        image = getattr(segment, "commit_image", None)
        if image is None:
            image = getattr(segment, "image", None)
        if image is None:
            return
        get_handle = getattr(self.runtime, "get_srt_sidecar_handle", None)
        get_state = getattr(self.runtime, "get_srt_sidecar_model_state", None)
        append_sidecar = getattr(
            self.runtime, "append_srt_sidecar_prepared_input", None
        )
        if (
            not callable(get_handle)
            or not callable(get_state)
            or not callable(append_sidecar)
        ):
            return
        sidecar_handle = get_handle(
            contexts.full.session,
            U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        )
        if sidecar_handle is None:
            return
        sidecar_state = get_state(
            contexts.full.session,
            U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        )
        if not ((sidecar_state or {}).get("u1") or {}).get("open_image_marker"):
            return
        sidecar_session = SimpleNamespace(
            handle=sidecar_handle,
            metadata={"ug_model_state": sidecar_state},
        )
        prepared = build_u1_native_generated_image_commit_prepared_input(
            tokenizer=tokenizer,
            image=image,
            session=sidecar_session,
        )
        prepared.srt_sidecar_role = U1_INTERLEAVE_TEXT_UNCONDITION_ROLE
        prepared.srt_sidecar_session_id = sidecar_handle.session_id
        append_sidecar(
            contexts.full.session,
            prepared,
            state=UGSegmentState.APPEND_IMAGE,
        )

    def generate_vlm_text(
        self,
        *,
        messages: list[UGInterleavedMessage | dict[str, Any]],
        max_new_tokens: int,
    ) -> UGVLMTextGenerationResult:
        return self._bridge.generate_vlm_text(
            messages=messages,
            max_new_tokens=max_new_tokens,
        )


def build_u1_native_vlm_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
) -> UGSRTPreparedInput:
    image = _first_u1_image_content(messages)
    question = _u1_question_text(messages)
    pixel_values, grid_hw = load_u1_native_image(image)
    input_ids, image_offsets, prompt = build_u1_vlm_input_ids_and_offsets(
        tokenizer=tokenizer,
        grid_hw=grid_hw,
        question=question,
    )

    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
    )

    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=pixel_values,
        model_specific_data={"image_grid_hws": grid_hw},
        offsets=image_offsets,
    )
    item.set_pad_value()
    mm_inputs = MultimodalInputs(mm_items=[item])
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=list(messages),
        mm_inputs=mm_inputs,
        adapter_metadata={
            "u1": {
                "segment_type": "vlm",
                "source": "native_vlm_input",
                "image_grid_hw": [list(map(int, row)) for row in grid_hw.tolist()],
                "image_offsets": list(image_offsets),
            },
            "ug_model_state_updates": {
                "u1": {
                    "last_segment_type": "vlm",
                    "last_source": "native_vlm_input",
                    "native_vlm_prompt": True,
                    "session_id": getattr(
                        getattr(session, "handle", None), "session_id", None
                    ),
                }
            },
        },
    )


def build_u1_native_interleave_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
    think_mode: bool = False,
    system_message: str = U1_INTERLEAVE_SYSTEM_MESSAGE,
) -> UGSRTPreparedInput:
    prompt_text = _u1_prompt_with_image_placeholders(
        _u1_question_text(messages),
        image_count=len(_u1_image_contents(messages)),
    )
    prompt = build_u1_interleave_prompt(
        prompt=prompt_text,
        system_message=system_message,
        think_mode=think_mode,
    )
    return _build_u1_native_interleave_like_prepared_input(
        tokenizer=tokenizer,
        prompt=prompt,
        messages=list(messages),
        images=_u1_image_contents(messages),
        session=session,
        role=None,
        source="native_interleave_prompt",
        segment_type="interleave",
        model_state_updates={
            "last_segment_type": "interleave",
            "last_source": "native_interleave_prompt",
            "native_interleave_prompt": True,
            "open_image_marker": False,
            "interleave_pending_image_marker": False,
            "interleave_image_count": 0,
        },
    )


def build_u1_native_interleave_text_uncondition_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
) -> UGSRTPreparedInput:
    images = _u1_image_contents(messages)
    prompt = build_u1_t2i_plain_query(prompt=U1_IMAGE_PLACEHOLDER * len(images))
    return _build_u1_native_interleave_like_prepared_input(
        tokenizer=tokenizer,
        prompt=prompt,
        messages=[UGInterleavedMessage(type="text", content="")],
        images=images,
        session=session,
        role=U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        source="native_interleave_text_uncondition_prompt",
        segment_type="interleave_text_uncondition",
        model_state_updates={
            "last_segment_type": "interleave_text_uncondition",
            "last_source": "native_interleave_text_uncondition_prompt",
            "native_interleave_text_uncondition_prompt": True,
            "open_image_marker": False,
        },
    )


def build_u1_native_interleave_text_uncondition_marker_prepared_input(
    *,
    tokenizer: Any,
    session: Any | None = None,
    logical_position: int,
) -> UGSRTPreparedInput:
    img_start_id = tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN)
    session_id = getattr(getattr(session, "handle", None), "session_id", None)
    next_position = int(logical_position) + 1
    return UGSRTPreparedInput(
        input_ids=[int(img_start_id)],
        input_text=U1_IMG_START_TOKEN,
        messages=[],
        position_ids=[[int(logical_position), 0, 0]],
        srt_sidecar_role=U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        srt_sidecar_session_id=(
            f"{session_id}:{U1_INTERLEAVE_TEXT_UNCONDITION_ROLE}"
            if session_id is not None
            else None
        ),
        adapter_metadata={
            "u1": {
                "segment_type": "interleave_text_uncondition_image_marker",
                "source": "native_interleave_text_uncondition_image_marker",
                "g_position_start": next_position,
            },
            "ug_model_state_updates": {
                "u1": {
                    "last_segment_type": "interleave_text_uncondition",
                    "last_source": "native_interleave_text_uncondition_image_marker",
                    "native_interleave_text_uncondition_prompt": True,
                    "open_image_marker": True,
                    "g_position_start": next_position,
                    "session_id": session_id,
                }
            },
        },
    )


def _build_u1_native_interleave_like_prepared_input(
    *,
    tokenizer: Any,
    prompt: str,
    messages: list[UGInterleavedMessage],
    images: list[Any],
    session: Any | None,
    role: str | None,
    source: str,
    segment_type: str,
    model_state_updates: dict[str, Any] | None,
) -> UGSRTPreparedInput:
    mm_inputs = None
    image_offsets: list[tuple[int, int]] = []
    g_position_start = None
    if images:
        import torch

        pixel_values_list = []
        grid_hw_list = []
        for image in images:
            pixel_values, grid_hw = load_u1_native_image(
                image,
                min_pixels=512 * 512,
                max_pixels=min(2048 * 2048, (4096 * 4096) // max(1, len(images))),
                upscale=False,
            )
            pixel_values_list.append(pixel_values)
            grid_hw_list.append(grid_hw)
        pixel_values = torch.cat(pixel_values_list, dim=0)
        grid_hw = torch.cat(grid_hw_list, dim=0)
        prompt = _replace_u1_image_placeholders(prompt, grid_hw)

    input_ids = _u1_tokenize_to_ids(
        tokenizer,
        prompt,
    )
    if not input_ids:
        raise RuntimeError(f"U1 native {segment_type} prompt produced no input ids")
    if images:
        from sglang.srt.managers.schedule_batch import (
            Modality,
            MultimodalDataItem,
            MultimodalInputs,
        )
        from sglang.srt.models.neo_chat import build_u1_vlm_thw_indexes

        context_id = tokenizer.convert_tokens_to_ids(U1_IMG_CONTEXT_TOKEN)
        image_offsets = _u1_image_context_offsets(
            input_ids,
            context_token_id=context_id,
        )
        if len(image_offsets) != len(images):
            raise RuntimeError(
                "U1 native interleave prompt image context count mismatch: "
                f"{len(image_offsets)} != {len(images)}"
            )
        positions = build_u1_vlm_thw_indexes(
            input_ids,
            grid_hw=grid_hw,
            img_start_token_id=tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN),
            img_context_token_id=context_id,
        )
        g_position_start = int(positions[0].max().item()) + 1
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=pixel_values,
            model_specific_data={"image_grid_hws": grid_hw},
            offsets=image_offsets,
        )
        item.set_pad_value()
        mm_inputs = MultimodalInputs(mm_items=[item])
    else:
        g_position_start = len(input_ids)

    session_id = getattr(getattr(session, "handle", None), "session_id", None)
    u1_metadata = {
        "segment_type": segment_type,
        "source": source,
        "g_position_start": g_position_start,
    }
    if image_offsets:
        u1_metadata["image_offsets"] = list(image_offsets)
        u1_metadata["image_count"] = len(images)
    adapter_metadata = {"u1": u1_metadata}
    if model_state_updates is not None:
        state_updates = dict(model_state_updates)
        state_updates["g_position_start"] = g_position_start
        state_updates["session_id"] = session_id
        adapter_metadata["ug_model_state_updates"] = {"u1": state_updates}
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=messages,
        mm_inputs=mm_inputs,
        srt_sidecar_role=role,
        srt_sidecar_session_id=(
            f"{session_id}:{role}"
            if role is not None and session_id is not None
            else None
        ),
        adapter_metadata=adapter_metadata,
    )


def build_u1_native_t2i_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
) -> UGSRTPreparedInput:
    prompt_text = _u1_question_text(messages)
    prompt = build_u1_t2i_prompt(prompt=prompt_text)
    input_ids = _u1_tokenize_to_ids(
        tokenizer,
        prompt,
        add_special_tokens=False,
    )
    if not input_ids:
        raise RuntimeError("U1 native T2I prompt produced no input ids")
    img_start_id = tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN)
    if img_start_id not in input_ids:
        raise RuntimeError("U1 native T2I prompt did not contain <img> token")
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=list(messages),
        adapter_metadata={
            "u1": {
                "segment_type": "t2i",
                "source": "native_t2i_prompt",
                "prompt_ends_with_image_marker": input_ids[-1] == img_start_id,
            },
            "ug_model_state_updates": {
                "u1": {
                    "last_segment_type": "t2i",
                    "last_source": "native_t2i_prompt",
                    "native_t2i_prompt": True,
                    "open_image_marker": input_ids[-1] == img_start_id,
                    "session_id": getattr(
                        getattr(session, "handle", None), "session_id", None
                    ),
                }
            },
        },
    )


def build_u1_native_edit_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
) -> UGSRTPreparedInput:
    image = _first_u1_image_content(messages)
    prompt_text = _u1_question_text(messages)
    if U1_IMAGE_PLACEHOLDER not in prompt_text:
        prompt_text = f"{U1_IMAGE_PLACEHOLDER}\n{prompt_text}"
    pixel_values, grid_hw = load_u1_native_image(
        image,
        min_pixels=512 * 512,
        max_pixels=2048 * 2048,
        upscale=False,
    )
    prompt = build_u1_t2i_prompt(prompt=prompt_text)
    prompt = _replace_u1_image_placeholders(prompt, grid_hw)
    input_ids = _u1_tokenize_to_ids(
        tokenizer,
        prompt,
        add_special_tokens=False,
    )
    if not input_ids:
        raise RuntimeError("U1 native edit prompt produced no input ids")
    context_id = tokenizer.convert_tokens_to_ids(U1_IMG_CONTEXT_TOKEN)
    selected = [
        index for index, token_id in enumerate(input_ids) if token_id == context_id
    ]
    if not selected:
        raise RuntimeError("U1 native edit prompt did not contain image context tokens")

    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
    )
    from sglang.srt.models.neo_chat import build_u1_vlm_thw_indexes

    positions = build_u1_vlm_thw_indexes(
        input_ids,
        grid_hw=grid_hw,
        img_start_token_id=tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN),
        img_context_token_id=context_id,
    )
    g_position_start = int(positions[0].max().item()) + 1
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=pixel_values,
        model_specific_data={"image_grid_hws": grid_hw},
        offsets=[(selected[0], selected[-1])],
    )
    item.set_pad_value()
    mm_inputs = MultimodalInputs(mm_items=[item])
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=list(messages),
        mm_inputs=mm_inputs,
        adapter_metadata={
            "u1": {
                "segment_type": "edit",
                "source": "native_edit_prompt",
                "image_grid_hw": [list(map(int, row)) for row in grid_hw.tolist()],
                "image_offsets": [(selected[0], selected[-1])],
                "g_position_start": g_position_start,
            },
            "ug_model_state_updates": {
                "u1": {
                    "last_segment_type": "edit",
                    "last_source": "native_edit_prompt",
                    "native_edit_prompt": True,
                    "g_position_start": g_position_start,
                    "session_id": getattr(
                        getattr(session, "handle", None), "session_id", None
                    ),
                }
            },
        },
    )


def build_u1_native_edit_img_condition_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
) -> UGSRTPreparedInput:
    image = _first_u1_image_content(messages)
    pixel_values, grid_hw = load_u1_native_image(
        image,
        min_pixels=512 * 512,
        max_pixels=2048 * 2048,
        upscale=False,
    )
    prompt = build_u1_t2i_plain_query(
        prompt=U1_IMAGE_PLACEHOLDER,
        append_text=U1_IMG_START_TOKEN,
    )
    prompt = _replace_u1_image_placeholders(prompt, grid_hw)
    prepared = _build_u1_native_image_sidecar_prepared_input(
        tokenizer=tokenizer,
        prompt=prompt,
        image=image,
        pixel_values=pixel_values,
        grid_hw=grid_hw,
        role=U1_EDIT_IMG_CONDITION_ROLE,
        source="native_edit_img_condition_prompt",
        segment_type="edit_img_condition",
        session=session,
    )
    return prepared


def build_u1_native_edit_uncondition_prepared_input(
    *,
    tokenizer: Any,
    session: Any | None = None,
) -> UGSRTPreparedInput:
    prompt = build_u1_t2i_plain_query(prompt="", append_text=U1_IMG_START_TOKEN)
    input_ids = _u1_tokenize_to_ids(
        tokenizer,
        prompt,
        add_special_tokens=False,
    )
    if not input_ids:
        raise RuntimeError("U1 native edit uncondition prompt produced no input ids")
    img_start_id = tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN)
    if input_ids[-1] != img_start_id:
        raise RuntimeError("U1 native edit uncondition prompt must end with <img>")
    session_id = getattr(getattr(session, "handle", None), "session_id", None)
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=[UGInterleavedMessage(type="text", content="")],
        srt_sidecar_role=U1_EDIT_UNCONDITION_ROLE,
        srt_sidecar_session_id=(
            f"{session_id}:{U1_EDIT_UNCONDITION_ROLE}"
            if session_id is not None
            else None
        ),
        adapter_metadata={
            "u1": {
                "segment_type": "edit_uncondition",
                "source": "native_edit_uncondition_prompt",
                "prompt_ends_with_image_marker": True,
            }
        },
    )


def _build_u1_native_image_sidecar_prepared_input(
    *,
    tokenizer: Any,
    prompt: str,
    image: Any,
    pixel_values: Any,
    grid_hw: Any,
    role: str,
    source: str,
    segment_type: str,
    session: Any | None,
) -> UGSRTPreparedInput:
    input_ids = _u1_tokenize_to_ids(
        tokenizer,
        prompt,
        add_special_tokens=False,
    )
    if not input_ids:
        raise RuntimeError(f"U1 native {segment_type} prompt produced no input ids")
    context_id = tokenizer.convert_tokens_to_ids(U1_IMG_CONTEXT_TOKEN)
    selected = [
        index for index, token_id in enumerate(input_ids) if token_id == context_id
    ]
    if not selected:
        raise RuntimeError(
            f"U1 native {segment_type} prompt did not contain image context tokens"
        )

    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
    )
    from sglang.srt.models.neo_chat import build_u1_vlm_thw_indexes

    positions = build_u1_vlm_thw_indexes(
        input_ids,
        grid_hw=grid_hw,
        img_start_token_id=tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN),
        img_context_token_id=context_id,
    )
    g_position_start = int(positions[0].max().item()) + 1
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=pixel_values,
        model_specific_data={"image_grid_hws": grid_hw},
        offsets=[(selected[0], selected[-1])],
    )
    item.set_pad_value()
    mm_inputs = MultimodalInputs(mm_items=[item])
    session_id = getattr(getattr(session, "handle", None), "session_id", None)
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=[UGInterleavedMessage(type="image", content=image)],
        mm_inputs=mm_inputs,
        srt_sidecar_role=role,
        srt_sidecar_session_id=(
            f"{session_id}:{role}" if session_id is not None else None
        ),
        adapter_metadata={
            "u1": {
                "segment_type": segment_type,
                "source": source,
                "image_grid_hw": [list(map(int, row)) for row in grid_hw.tolist()],
                "image_offsets": [(selected[0], selected[-1])],
                "g_position_start": g_position_start,
            }
        },
    )


def build_u1_native_generated_image_commit_prepared_input(
    *,
    tokenizer: Any,
    image: Any,
    session: Any | None = None,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
) -> UGSRTPreparedInput:
    precomputed_embeddings = None
    if isinstance(image, dict) and image.get("precomputed_embeddings") is not None:
        import torch

        precomputed_embeddings = (
            torch.as_tensor(image["precomputed_embeddings"]).detach().cpu()
        )
        grid_hw_value = image.get("grid_hw", image.get("image_grid_hws"))
        pixel_values = None
        if grid_hw_value is None:
            if image.get("pixel_values") is None:
                raise ValueError(
                    "U1 generated image commit with precomputed embeddings "
                    "requires grid_hw/image_grid_hws or pixel_values"
                )
            pixel_values, loaded_grid_hw = load_u1_generated_image_for_commit(
                image,
                patch_size=patch_size,
            )
            grid_hw = loaded_grid_hw
        else:
            grid_hw = torch.as_tensor(grid_hw_value, dtype=torch.long)
    else:
        pixel_values, grid_hw = load_u1_generated_image_for_commit(
            image,
            patch_size=patch_size,
        )
    merge_size = _u1_merge_size_from_downsample_ratio(downsample_ratio)
    grid_h = int(grid_hw[0, 0])
    grid_w = int(grid_hw[0, 1])
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError(
            "U1 generated image patch grid must be divisible by merge size "
            f"{merge_size}, got {grid_h}x{grid_w}"
        )
    token_h = grid_h // merge_size
    token_w = grid_w // merge_size
    num_context_tokens = token_h * token_w
    if num_context_tokens <= 0:
        raise ValueError("U1 generated image commit requires image context tokens")
    if precomputed_embeddings is not None:
        flat_precomputed = precomputed_embeddings.reshape(
            -1, precomputed_embeddings.shape[-1]
        )
        if int(flat_precomputed.shape[0]) != num_context_tokens:
            raise ValueError(
                "U1 generated image commit precomputed embedding length must "
                f"match image context tokens, got {int(flat_precomputed.shape[0])} "
                f"vs {num_context_tokens}"
            )
        precomputed_embeddings = flat_precomputed

    img_start_id = tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN)
    context_id = tokenizer.convert_tokens_to_ids(U1_IMG_CONTEXT_TOKEN)
    img_end_id = tokenizer.convert_tokens_to_ids(U1_IMG_END_TOKEN)
    omit_start = _u1_session_has_open_image_marker(session, img_start_id)
    prefix_len = _u1_session_logical_position(session)
    if prefix_len is None:
        prefix_len = _u1_session_context_length(session)

    input_ids: list[int] = []
    position_ids: list[list[int]] = []
    if omit_start:
        context_t = prefix_len
    else:
        input_ids.append(int(img_start_id))
        position_ids.append([prefix_len, 0, 0])
        context_t = prefix_len + 1

    context_start = len(input_ids)
    input_ids.extend([int(context_id)] * num_context_tokens)
    for h_idx in range(token_h):
        for w_idx in range(token_w):
            position_ids.append([context_t, h_idx, w_idx])
    context_end = len(input_ids) - 1
    end_t = context_t + 1
    input_ids.append(int(img_end_id))
    position_ids.append([end_t, 0, 0])

    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
    )
    import torch

    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=pixel_values,
        precomputed_embeddings=precomputed_embeddings,
        model_specific_data={"image_grid_hws": grid_hw},
        offsets=[(context_start, context_end)],
    )
    item.set_pad_value()
    mm_inputs = MultimodalInputs(mm_items=[item])
    mm_inputs.mrope_positions = torch.tensor(position_ids, dtype=torch.long).t()
    mm_inputs.mrope_position_delta = (
        mm_inputs.mrope_positions[:, -1:]
        .max(
            dim=0,
            keepdim=True,
        )
        .values
    )

    message = UGInterleavedMessage(type="image", content=image)
    metadata = _u1_generated_image_commit_metadata(
        session=session,
        token_indices=list(range(context_start, context_end + 1)),
        grid_hw=grid_hw,
        omit_start=omit_start,
        g_position_start=end_t + 1,
    )
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text="<u1:generated_image_commit>",
        messages=[message],
        position_ids=position_ids,
        mm_inputs=mm_inputs,
        mot_image_token_indices=list(range(context_start, context_end + 1)),
        adapter_metadata=metadata,
    )


def build_u1_native_t2i_cfg_uncondition_prepared_input(
    *,
    tokenizer: Any,
    session: Any | None = None,
) -> UGSRTPreparedInput:
    prompt = build_u1_t2i_uncondition_prompt()
    input_ids = _u1_tokenize_to_ids(
        tokenizer,
        prompt,
        add_special_tokens=False,
    )
    if not input_ids:
        raise RuntimeError("U1 native T2I CFG prompt produced no input ids")
    img_start_id = tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN)
    if input_ids[-1] != img_start_id:
        raise RuntimeError("U1 native T2I CFG prompt must end with <img>")
    session_id = getattr(getattr(session, "handle", None), "session_id", None)
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=[UGInterleavedMessage(type="text", content="")],
        srt_sidecar_role=U1_T2I_CFG_UNCONDITION_ROLE,
        srt_sidecar_session_id=(
            f"{session_id}:{U1_T2I_CFG_UNCONDITION_ROLE}"
            if session_id is not None
            else None
        ),
        adapter_metadata={
            "u1": {
                "segment_type": "t2i_cfg_uncondition",
                "source": "native_t2i_cfg_uncondition_prompt",
                "prompt_ends_with_image_marker": True,
            }
        },
    )


def build_u1_t2i_prompt(*, prompt: str) -> str:
    return (
        f"<|im_start|>system\n{U1_SYSTEM_MESSAGE_FOR_GEN}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        f"{U1_IMG_START_TOKEN}"
    )


def build_u1_interleave_prompt(
    *,
    prompt: str,
    system_message: str = U1_INTERLEAVE_SYSTEM_MESSAGE,
    think_mode: bool = False,
) -> str:
    query = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    if not think_mode:
        query += "<think>\n\n</think>\n\n"
    return query


def build_u1_t2i_plain_query(*, prompt: str, append_text: str | None = None) -> str:
    query = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if append_text is not None:
        query += append_text
    return query


def _replace_u1_image_placeholders(prompt: str, grid_hw: Any) -> str:
    for i in range(int(grid_hw.shape[0])):
        num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * 0.5**2)
        image_tokens = (
            U1_IMG_START_TOKEN
            + U1_IMG_CONTEXT_TOKEN * num_patch_token
            + U1_IMG_END_TOKEN
        )
        prompt = prompt.replace(U1_IMAGE_PLACEHOLDER, image_tokens, 1)
    if U1_IMAGE_PLACEHOLDER in prompt:
        raise RuntimeError("U1 prompt still contains unresolved <image> placeholders")
    return prompt


def build_u1_t2i_uncondition_prompt() -> str:
    return build_u1_t2i_plain_query(prompt="", append_text=U1_IMG_START_TOKEN)


def _u1_image_contents(messages: list[UGInterleavedMessage]) -> list[Any]:
    return [message.content for message in messages if message.type == "image"]


def _u1_prompt_with_image_placeholders(prompt: str, *, image_count: int) -> str:
    if image_count <= 0:
        return prompt
    placeholder_count = prompt.count(U1_IMAGE_PLACEHOLDER)
    if placeholder_count > image_count:
        raise ValueError(
            "U1 prompt contains more <image> placeholders than image inputs: "
            f"{placeholder_count} > {image_count}"
        )
    if placeholder_count < image_count:
        prompt = (
            f"{U1_IMAGE_PLACEHOLDER}\n" * (image_count - placeholder_count) + prompt
        )
    return prompt


def _u1_image_context_offsets(
    input_ids: list[int],
    *,
    context_token_id: int,
) -> list[tuple[int, int]]:
    offsets = []
    index = 0
    while index < len(input_ids):
        if int(input_ids[index]) != int(context_token_id):
            index += 1
            continue
        start = index
        while index + 1 < len(input_ids) and int(input_ids[index + 1]) == int(
            context_token_id
        ):
            index += 1
        offsets.append((start, index))
        index += 1
    return offsets


def build_u1_vlm_input_ids_and_offsets(
    *,
    tokenizer: Any,
    grid_hw: Any,
    question: str,
) -> tuple[list[int], list[tuple[int, int]], str]:
    prompt = build_u1_vlm_prompt(question=question)
    for i in range(int(grid_hw.shape[0])):
        num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * 0.5**2)
        image_tokens = (
            U1_IMG_START_TOKEN
            + U1_IMG_CONTEXT_TOKEN * num_patch_token
            + U1_IMG_END_TOKEN
        )
        prompt = prompt.replace(U1_IMAGE_PLACEHOLDER, image_tokens, 1)

    input_ids = _u1_tokenize_to_ids(tokenizer, prompt)
    context_token_id = tokenizer.convert_tokens_to_ids(U1_IMG_CONTEXT_TOKEN)
    selected = [
        index
        for index, token_id in enumerate(input_ids)
        if token_id == context_token_id
    ]
    if not selected:
        raise RuntimeError("U1 native VLM prompt did not contain image context tokens")
    return input_ids, [(selected[0], selected[-1])], prompt


def build_u1_vlm_prompt(*, question: str) -> str:
    return (
        f"<|im_start|>user\n{U1_IMAGE_PLACEHOLDER}\n{question}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


class U1NativeSRTPixelFlowExecutor:
    """Run SenseNova U1 pixel-flow G steps through SRT's ModelRunner/KV path."""

    def __init__(
        self,
        srt_model: Any,
        *,
        forward_batch_provider: Any,
    ) -> None:
        self.srt_model = srt_model
        self.forward_batch_provider = forward_batch_provider
        self.debug_tensor_dump_dir = None
        self.debug_tensor_dump_max_g_calls = 32

    def generate(
        self,
        *,
        contexts: UGContextBundle,
        batch: Any,
        server_args: Any,
        srt_kv_token_binding: Any,
        cfg_img_condition_srt_kv_token_binding: Any | None = None,
        cfg_uncondition_srt_kv_token_binding: Any | None = None,
    ) -> UGGSegmentResult:
        import torch

        with torch.inference_mode():
            return self._generate_impl(
                contexts=contexts,
                batch=batch,
                server_args=server_args,
                srt_kv_token_binding=srt_kv_token_binding,
                cfg_img_condition_srt_kv_token_binding=(
                    cfg_img_condition_srt_kv_token_binding
                ),
                cfg_uncondition_srt_kv_token_binding=(
                    cfg_uncondition_srt_kv_token_binding
                ),
            )

    def _generate_impl(
        self,
        *,
        contexts: UGContextBundle,
        batch: Any,
        server_args: Any,
        srt_kv_token_binding: Any,
        cfg_img_condition_srt_kv_token_binding: Any | None = None,
        cfg_uncondition_srt_kv_token_binding: Any | None = None,
    ) -> UGGSegmentResult:
        del server_args
        import numpy as np
        import torch
        from PIL import Image

        if contexts.full.session is None:
            raise ValueError("U1 native pixel-flow requires contexts.full.session")
        sampling_params = batch.sampling_params
        cfg_text_scale = float(getattr(sampling_params, "cfg_text_scale", 1.0))
        cfg_img_scale = float(getattr(sampling_params, "cfg_img_scale", 1.0))
        needs_cfg = not (cfg_text_scale == 1.0 and cfg_img_scale == 1.0)
        needs_img_condition = needs_cfg and (
            cfg_img_scale == 1.0 or cfg_text_scale != cfg_img_scale
        )
        needs_uncondition = needs_cfg and cfg_img_scale != 1.0
        if (
            needs_img_condition
            and cfg_img_condition_srt_kv_token_binding is None
            and not needs_uncondition
            and cfg_uncondition_srt_kv_token_binding is not None
        ):
            cfg_img_condition_srt_kv_token_binding = (
                cfg_uncondition_srt_kv_token_binding
            )
            cfg_uncondition_srt_kv_token_binding = None
        if needs_img_condition and cfg_img_condition_srt_kv_token_binding is None:
            raise RuntimeError(
                "U1 native SRT pixel-flow CFG requires an image-condition "
                "SRT KV token binding"
            )
        if needs_uncondition and cfg_uncondition_srt_kv_token_binding is None:
            raise RuntimeError(
                "U1 native SRT pixel-flow CFG requires an uncondition SRT KV "
                "token binding"
            )

        image_size = _u1_batch_image_size(batch)
        width, height = image_size
        patch_size = int(self.srt_model.patch_size)
        merge_size = int(1 / float(self.srt_model.downsample_ratio))
        divisor = patch_size * merge_size
        if width % divisor or height % divisor:
            raise ValueError(
                "U1 native pixel-flow image size must be divisible by "
                f"{divisor}, got {width}x{height}"
            )

        token_h = height // divisor
        token_w = width // divisor
        grid_h = height // patch_size
        grid_w = width // patch_size
        steps = int(getattr(sampling_params, "num_inference_steps", None) or 0)
        if steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {steps}")

        device = _u1_model_device(self.srt_model)
        dtype = _u1_model_dtype(self.srt_model)
        seed = int(getattr(batch, "seed", None) or 0)
        generator = _u1_session_torch_generator(
            contexts.full.metadata,
            seed=seed,
            device=device,
        )
        noise_scale = float(
            self.srt_model.noise_scale_for_image(grid_h=grid_h, grid_w=grid_w)
        )
        image_prediction = noise_scale * torch.randn(
            (1, 3, height, width),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=device, dtype=torch.long)
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)
        timesteps = self.srt_model.apply_time_schedule(
            timesteps,
            image_seq_len=token_h * token_w,
            timestep_shift=float(getattr(sampling_params, "timestep_shift", 1.0)),
        )
        g_position_start = int(
            contexts.full.metadata.get(
                "u1_g_position_start",
                _u1_binding_position_count(srt_kv_token_binding),
            )
        )
        indexes_image = self.srt_model.build_t2i_image_indexes(
            token_h=token_h,
            token_w=token_w,
            text_len=g_position_start,
            device=device,
        )
        cfg_img_condition_position_count = None
        indexes_image_img_condition = None
        if cfg_img_condition_srt_kv_token_binding is not None:
            cfg_img_condition_position_count = int(
                _u1_binding_position_count(cfg_img_condition_srt_kv_token_binding)
            )
            indexes_image_img_condition = self.srt_model.build_t2i_image_indexes(
                token_h=token_h,
                token_w=token_w,
                text_len=cfg_img_condition_position_count,
                device=device,
            )
        cfg_uncondition_position_count = None
        indexes_image_uncondition = None
        if cfg_uncondition_srt_kv_token_binding is not None:
            cfg_uncondition_position_count = int(
                _u1_binding_position_count(cfg_uncondition_srt_kv_token_binding)
            )
            indexes_image_uncondition = self.srt_model.build_t2i_image_indexes(
                token_h=token_h,
                token_w=token_w,
                text_len=cfg_uncondition_position_count,
                device=device,
            )
        generation_input = {
            "packed_seqlens": torch.tensor(
                [token_h * token_w], dtype=torch.int32, device=device
            ),
            "packed_position_ids": indexes_image,
        }
        prepared = SimpleNamespace(
            generation_input=generation_input,
            srt_kv_token_binding=srt_kv_token_binding,
        )
        prepared_img_condition = None
        if cfg_img_condition_srt_kv_token_binding is not None:
            prepared_img_condition = SimpleNamespace(
                generation_input={
                    "packed_seqlens": generation_input["packed_seqlens"],
                    "packed_position_ids": indexes_image_img_condition,
                },
                srt_kv_token_binding=cfg_img_condition_srt_kv_token_binding,
            )
        prepared_uncondition = None
        if cfg_uncondition_srt_kv_token_binding is not None:
            prepared_uncondition = SimpleNamespace(
                generation_input={
                    "packed_seqlens": generation_input["packed_seqlens"],
                    "packed_position_ids": indexes_image_uncondition,
                },
                srt_kv_token_binding=cfg_uncondition_srt_kv_token_binding,
            )
        cfg_interval = list(getattr(sampling_params, "cfg_interval", [0.0, 1.0]))
        if len(cfg_interval) != 2:
            raise ValueError("U1 native pixel-flow cfg_interval must have two values")
        cfg_start, cfg_end = float(cfg_interval[0]), float(cfg_interval[1])
        cfg_renorm_type = str(getattr(sampling_params, "cfg_renorm_type", "none"))

        for step_i in range(steps):
            timestep = timesteps[step_i]
            next_timestep = timesteps[step_i + 1]
            use_cfg = (float(timestep) > cfg_start and float(timestep) < cfg_end) or (
                cfg_start == 0.0
            )
            z = self.srt_model.patchify(image_prediction, patch_size * merge_size)
            image_input = self.srt_model.patchify(
                image_prediction,
                patch_size,
                channel_first=True,
            )
            image_embeds = self.srt_model.extract_feature(
                image_input.view(grid_h * grid_w, -1),
                gen_model=True,
                grid_hw=gen_grid_hw,
            ).view(1, token_h * token_w, -1)
            timestep_values = timestep.expand(token_h * token_w)
            timestep_embeddings = self.srt_model.fm_modules["timestep_embedder"](
                timestep_values
            ).view(1, token_h * token_w, -1)
            if getattr(self.srt_model, "add_noise_scale_embedding", False):
                noise_values = torch.full_like(
                    timestep_values,
                    noise_scale / float(self.srt_model.noise_scale_max_value),
                )
                timestep_embeddings = timestep_embeddings + self.srt_model.fm_modules[
                    "noise_scale_embedder"
                ](noise_values).view(1, token_h * token_w, -1)
            image_embeds = image_embeds + timestep_embeddings

            v_condition = self._predict_v(
                prepared=prepared,
                image_embeds=image_embeds,
                indexes_image=indexes_image,
                timestep=timestep,
                z=z,
                image_size=image_size,
            )
            self._dump_debug_g_call(
                contexts=contexts,
                branch="condition",
                v=v_condition,
                image_embeds=image_embeds,
                indexes_image=indexes_image,
                timestep=timestep,
                z=z,
            )
            if not use_cfg or not needs_cfg:
                v_pred = v_condition
            elif cfg_img_scale == 1.0:
                v_img_condition = self._predict_v(
                    prepared=prepared_img_condition,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image_img_condition,
                    timestep=timestep,
                    z=z,
                    image_size=image_size,
                )
                self._dump_debug_g_call(
                    contexts=contexts,
                    branch="text_uncondition",
                    v=v_img_condition,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image_img_condition,
                    timestep=timestep,
                    z=z,
                )
                v_pred = v_img_condition + cfg_text_scale * (
                    v_condition - v_img_condition
                )
            elif cfg_text_scale == cfg_img_scale:
                v_uncondition = self._predict_v(
                    prepared=prepared_uncondition,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image_uncondition,
                    timestep=timestep,
                    z=z,
                    image_size=image_size,
                )
                self._dump_debug_g_call(
                    contexts=contexts,
                    branch="img_uncondition",
                    v=v_uncondition,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image_uncondition,
                    timestep=timestep,
                    z=z,
                )
                v_pred = v_uncondition + cfg_text_scale * (v_condition - v_uncondition)
            else:
                v_img_condition = self._predict_v(
                    prepared=prepared_img_condition,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image_img_condition,
                    timestep=timestep,
                    z=z,
                    image_size=image_size,
                )
                self._dump_debug_g_call(
                    contexts=contexts,
                    branch="text_uncondition",
                    v=v_img_condition,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image_img_condition,
                    timestep=timestep,
                    z=z,
                )
                v_uncondition = self._predict_v(
                    prepared=prepared_uncondition,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image_uncondition,
                    timestep=timestep,
                    z=z,
                    image_size=image_size,
                )
                self._dump_debug_g_call(
                    contexts=contexts,
                    branch="img_uncondition",
                    v=v_uncondition,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image_uncondition,
                    timestep=timestep,
                    z=z,
                )
                v_pred = (
                    v_uncondition
                    + cfg_text_scale * (v_condition - v_img_condition)
                    + cfg_img_scale * (v_img_condition - v_uncondition)
                )
            if needs_cfg and use_cfg:
                v_pred = self._apply_cfg_renorm(
                    v_condition=v_condition,
                    v_pred=v_pred,
                    cfg_renorm_type=cfg_renorm_type,
                )

            z = z + (next_timestep - timestep) * v_pred
            image_prediction = self.srt_model.unpatchify(
                z,
                patch_size * merge_size,
                height,
                width,
            )

        array = (
            (image_prediction[0].float() * 0.5 + 0.5)
            .clamp(0, 1)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        image = Image.fromarray((array * 255.0).round().astype(np.uint8), "RGB")
        commit_embeddings, commit_grid_hw = (
            _u1_precompute_generated_image_commit_embeddings(
                self.srt_model,
                image_prediction,
                patch_size=patch_size,
                grid_hw=gen_grid_hw[:1],
            )
        )
        commit_image = {
            "pixel_values": image_prediction.detach().to(torch.bfloat16).cpu(),
            "value_range": "minus_one_to_one",
            "grid_hw": commit_grid_hw,
            "precomputed_embeddings": commit_embeddings,
        }
        self._dump_debug_generated_image(
            contexts=contexts,
            image_prediction=image_prediction,
            commit_embeddings=commit_embeddings,
            commit_grid_hw=commit_grid_hw,
        )
        return UGGSegmentResult(
            type="image",
            image=image,
            metadata={
                "g_kind": "pixel_flow",
                "native_srt_pixel_flow": True,
                "temporary_g_kv": True,
                "timesteps": steps,
                "seed": seed,
                "width": width,
                "height": height,
                "grid": (token_h, token_w),
                "g_position_start": g_position_start,
                "condition_position_count": g_position_start,
                "cfg_img_condition_position_count": (cfg_img_condition_position_count),
                "cfg_uncondition_position_count": cfg_uncondition_position_count,
                "noise_scale": noise_scale,
                "cfg_text_scale": cfg_text_scale,
                "cfg_img_scale": cfg_img_scale,
                "cfg_renorm_type": (cfg_renorm_type if needs_cfg else "none"),
            },
            commit_image=commit_image,
        )

    def _predict_v(
        self,
        *,
        prepared: Any,
        image_embeds: Any,
        indexes_image: Any,
        timestep: Any,
        z: Any,
        image_size: tuple[int, int],
    ) -> Any:
        forward_batch_context = self.forward_batch_provider(
            prepared=prepared,
            latent_tokens=image_embeds,
            timestep=timestep,
        )
        forward_batch = getattr(
            forward_batch_context,
            "forward_batch",
            forward_batch_context,
        )
        try:
            v = self.srt_model.predict_u1_pixel_flow_from_srt(
                image_embeds=image_embeds,
                indexes_image=indexes_image,
                forward_batch=forward_batch,
                timestep=timestep,
                z=z,
                image_size=image_size,
            )
            self._last_predict_debug_payload = getattr(
                self.srt_model,
                "_last_u1_pixel_flow_debug",
                None,
            )
            return v
        finally:
            release = getattr(forward_batch_context, "release", None)
            if callable(release):
                release()

    def _dump_debug_g_call(
        self,
        *,
        contexts: UGContextBundle,
        branch: str,
        v: Any,
        image_embeds: Any,
        indexes_image: Any,
        timestep: Any,
        z: Any,
    ) -> None:
        dump_dir = getattr(self, "debug_tensor_dump_dir", None)
        if dump_dir is None:
            return
        metadata = contexts.full.metadata
        call_index = int(metadata.get("_u1_debug_g_call_index", 0))
        metadata["_u1_debug_g_call_index"] = call_index + 1
        if call_index >= int(getattr(self, "debug_tensor_dump_max_g_calls", 32)):
            return
        import torch

        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "call_index": call_index,
            "branch": branch,
            "input_embeds": image_embeds.detach().float().cpu(),
            "v": v.detach().float().cpu(),
            "indexes_image": indexes_image.detach().cpu(),
            "timestep": float(timestep.detach().float().cpu().item()),
            "z": z.detach().float().cpu(),
        }
        debug_payload = getattr(self, "_last_predict_debug_payload", None)
        if isinstance(debug_payload, dict):
            hidden_states = debug_payload.get("hidden_states")
            if hidden_states is not None:
                payload["hidden_states"] = hidden_states.detach().float().cpu()
            x_pred = debug_payload.get("x_pred")
            if x_pred is not None:
                payload["x_pred"] = x_pred.detach().float().cpu()
            layer_hidden_states = debug_payload.get("layer_hidden_states")
            if layer_hidden_states is not None:
                payload["layer_hidden_states"] = [
                    {
                        "layer": int(layer_index),
                        "hidden_states": layer_hidden.detach().float().cpu(),
                    }
                    for layer_index, layer_hidden in layer_hidden_states
                ]
            sublayer_states = debug_payload.get("sublayer_states")
            if sublayer_states is not None:
                payload["sublayer_states"] = [
                    {
                        "layer": int(record["layer"]),
                        "name": str(record["name"]),
                        "hidden_states": record["hidden_states"].detach().float().cpu(),
                    }
                    for record in sublayer_states
                ]
        torch.save(payload, dump_dir / f"candidate_g_call_{call_index:04d}.pt")

    def _dump_debug_generated_image(
        self,
        *,
        contexts: UGContextBundle,
        image_prediction: Any,
        commit_embeddings: Any,
        commit_grid_hw: Any,
    ) -> None:
        dump_dir = getattr(self, "debug_tensor_dump_dir", None)
        if dump_dir is None:
            return
        metadata = contexts.full.metadata
        image_index = int(metadata.get("_u1_debug_generated_image_index", 0))
        metadata["_u1_debug_generated_image_index"] = image_index + 1
        import torch

        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "image_index": image_index,
                "image_prediction": image_prediction.detach().float().cpu(),
                "commit_embeddings": commit_embeddings.detach().float().cpu(),
                "commit_grid_hw": commit_grid_hw.detach().cpu(),
            },
            dump_dir / f"candidate_generated_image_{image_index:04d}.pt",
        )

    @staticmethod
    def _apply_cfg_renorm(
        *,
        v_condition: Any,
        v_pred: Any,
        cfg_renorm_type: str,
    ) -> Any:
        if cfg_renorm_type == "none":
            return v_pred
        if cfg_renorm_type == "global":
            norm_v_condition = v_condition.norm(dim=(1, 2), keepdim=True)
            norm_v_cfg = v_pred.norm(dim=(1, 2), keepdim=True)
        elif cfg_renorm_type == "channel":
            norm_v_condition = v_condition.norm(dim=-1, keepdim=True)
            norm_v_cfg = v_pred.norm(dim=-1, keepdim=True)
        else:
            raise ValueError(
                f"Unsupported U1 native pixel-flow CFG renorm type: {cfg_renorm_type}"
            )
        scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
        return v_pred * scale


def _u1_tokenize_to_ids(
    tokenizer: Any,
    prompt: str,
    *,
    add_special_tokens: bool | None = None,
) -> list[int]:
    kwargs = {"return_tensors": "pt"}
    if add_special_tokens is not None:
        kwargs["add_special_tokens"] = add_special_tokens
    try:
        tokenized = tokenizer(prompt, **kwargs)
    except TypeError:
        tokenized = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    if hasattr(input_ids, "tolist"):
        return input_ids[0].tolist()
    return list(input_ids[0])


def _u1_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise RuntimeError(f"U1 tokenizer has no id for token {token!r}")
    return int(token_id)


def _u1_eos_token_ids(tokenizer: Any) -> set[int]:
    eos_ids = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        eos_ids.add(int(eos_token_id))
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        for token in ("<|im_end|>", "</s>"):
            try:
                token_id = convert(token)
            except Exception:
                continue
            if token_id is not None:
                try:
                    eos_ids.add(int(token_id))
                except (TypeError, ValueError):
                    pass
    return eos_ids


def _u1_decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return " ".join(str(token_id) for token_id in token_ids)
    try:
        return str(decode(token_ids, skip_special_tokens=True))
    except TypeError:
        return str(decode(token_ids))


def _u1_batch_image_size(batch: Any) -> tuple[int, int]:
    sampling_params = batch.sampling_params
    height = _u1_first_int(
        getattr(batch, "height", None),
        getattr(sampling_params, "height", None),
        default=1024,
    )
    width = _u1_first_int(
        getattr(batch, "width", None),
        getattr(sampling_params, "width", None),
        default=1024,
    )
    return width, height


def _u1_needs_text_cfg(sampling_params: Any | None) -> bool:
    if sampling_params is None:
        return False
    return float(getattr(sampling_params, "cfg_text_scale", 1.0)) > 1.0


def _u1_needs_any_cfg(sampling_params: Any | None) -> bool:
    if sampling_params is None:
        return False
    return not (
        float(getattr(sampling_params, "cfg_text_scale", 1.0)) == 1.0
        and float(getattr(sampling_params, "cfg_img_scale", 1.0)) == 1.0
    )


def _u1_binding_position_count(binding: Any) -> int:
    position_count = getattr(binding, "position_count", None)
    if position_count is not None:
        return int(position_count)
    return int(getattr(binding, "token_count"))


def _u1_session_torch_generator(
    metadata: dict[str, Any],
    *,
    seed: int,
    device: Any,
):
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


def _u1_first_int(*values, default: int) -> int:
    for value in values:
        if value is not None:
            return int(value)
    return int(default)


def _u1_model_device(srt_model: Any):
    import torch

    vision_model = getattr(srt_model, "vision_model", None)
    device = getattr(vision_model, "device", None)
    if device is not None:
        return device
    return next(srt_model.parameters()).device


def _u1_model_dtype(srt_model: Any):
    vision_model = getattr(srt_model, "vision_model", None)
    dtype = getattr(vision_model, "dtype", None)
    if dtype is not None:
        return dtype
    return next(srt_model.parameters()).dtype


def load_u1_native_image(
    image: Any,
    *,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
    upscale: bool = False,
):
    if isinstance(image, dict):
        pixel_values = image.get("pixel_values")
        grid_hw = image.get("grid_hw", image.get("image_grid_hws"))
        if pixel_values is not None and grid_hw is not None:
            import torch

            return (
                torch.as_tensor(pixel_values, dtype=torch.float32),
                torch.as_tensor(grid_hw, dtype=torch.long),
            )

    import numpy as np
    import torch
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    else:
        image = image.convert("RGB")

    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    resized = _u1_dynamic_preprocess_native_resolution(
        image,
        size_factor=int(patch_size // downsample_ratio),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    pixel_values = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(U1_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(U1_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    pixel_values = (pixel_values - mean) / std
    return _u1_preprocess_pixel_values(pixel_values, patch_size=patch_size)


def load_u1_generated_image_for_commit(
    image: Any,
    *,
    patch_size: int = 16,
):
    if isinstance(image, dict):
        pixel_values = image.get("pixel_values")
        if pixel_values is not None and image.get("value_range") == "minus_one_to_one":
            return _load_u1_generated_tensor_for_commit(
                pixel_values,
                patch_size=patch_size,
                minus_one_to_one=True,
            )
        grid_hw = image.get("grid_hw", image.get("image_grid_hws"))
        if pixel_values is not None and grid_hw is not None:
            import torch

            return (
                torch.as_tensor(pixel_values, dtype=torch.float32),
                torch.as_tensor(grid_hw, dtype=torch.long),
            )

    import numpy as np
    import torch
    from PIL import Image

    if torch.is_tensor(image):
        return _load_u1_generated_tensor_for_commit(
            image,
            patch_size=patch_size,
            minus_one_to_one=False,
        )
    else:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        pixel_values = torch.from_numpy(array).permute(2, 0, 1)

    height = int(pixel_values.shape[1])
    width = int(pixel_values.shape[2])
    if height % patch_size or width % patch_size:
        raise ValueError(
            "U1 generated image commit requires image size divisible by "
            f"patch_size={patch_size}, got {width}x{height}"
        )
    mean = torch.tensor(U1_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(U1_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    pixel_values = (pixel_values - mean) / std
    return _u1_preprocess_pixel_values(pixel_values, patch_size=patch_size)


def _load_u1_generated_tensor_for_commit(
    image: Any,
    *,
    patch_size: int,
    minus_one_to_one: bool,
):
    import torch

    pixel_values = torch.as_tensor(image).detach().cpu()
    pixel_values = (
        pixel_values.to(torch.bfloat16) if minus_one_to_one else pixel_values.float()
    )
    if pixel_values.ndim == 4:
        if int(pixel_values.shape[0]) != 1:
            raise ValueError(
                "U1 generated image commit expects a single image tensor, "
                f"got batch={int(pixel_values.shape[0])}"
            )
        pixel_values = pixel_values[0]
    if pixel_values.ndim != 3 or int(pixel_values.shape[0]) != 3:
        raise ValueError(
            "U1 generated image commit tensor must have shape [3,H,W] "
            f"or [1,3,H,W], got {tuple(pixel_values.shape)}"
        )
    if minus_one_to_one:
        pixel_values = pixel_values * 0.5 + 0.5
    elif float(pixel_values.min()) < 0.0:
        pixel_values = pixel_values * 0.5 + 0.5

    height = int(pixel_values.shape[1])
    width = int(pixel_values.shape[2])
    if height % patch_size or width % patch_size:
        raise ValueError(
            "U1 generated image commit requires image size divisible by "
            f"patch_size={patch_size}, got {width}x{height}"
        )
    mean = torch.tensor(U1_IMAGENET_MEAN, dtype=pixel_values.dtype).view(3, 1, 1)
    std = torch.tensor(U1_IMAGENET_STD, dtype=pixel_values.dtype).view(3, 1, 1)
    pixel_values = (pixel_values - mean) / std
    return _u1_preprocess_pixel_values(pixel_values, patch_size=patch_size)


def _u1_precompute_generated_image_commit_embeddings(
    srt_model: Any,
    image: Any,
    *,
    patch_size: int,
    grid_hw: Any | None = None,
):
    import torch

    pixel_values, loaded_grid_hw = _load_u1_generated_tensor_for_commit(
        image,
        patch_size=patch_size,
        minus_one_to_one=True,
    )
    if grid_hw is None:
        grid_hw = loaded_grid_hw
    else:
        grid_hw = torch.as_tensor(grid_hw, dtype=torch.long).detach().cpu()
    with torch.no_grad():
        embeddings = srt_model.extract_feature(
            pixel_values,
            grid_hw=grid_hw,
            gen_model=False,
        )
    return embeddings.reshape(-1, embeddings.shape[-1]).detach().cpu(), grid_hw


def _u1_session_context_length(session: Any | None) -> int:
    handle = getattr(session, "handle", None)
    context_length = getattr(handle, "context_length", None)
    if context_length is not None:
        return int(context_length)
    return int(getattr(session, "srt_last_origin_input_len", 0) or 0)


def _u1_session_logical_position(session: Any | None) -> int | None:
    metadata = getattr(session, "metadata", {}) or {}
    model_state = metadata.get("ug_model_state") or {}
    u1_state = model_state.get("u1") or {}
    g_position_start = u1_state.get("g_position_start")
    if g_position_start is None:
        return None
    return int(g_position_start)


def _u1_merge_size_from_downsample_ratio(downsample_ratio: float) -> int:
    if downsample_ratio <= 0:
        raise ValueError(f"downsample_ratio must be > 0, got {downsample_ratio}")
    merge_size = int(1 / downsample_ratio)
    if merge_size <= 0 or abs((1 / merge_size) - downsample_ratio) > 1e-6:
        raise ValueError(
            "U1 downsample_ratio must be the reciprocal of an integer, "
            f"got {downsample_ratio}"
        )
    return merge_size


def _u1_session_has_open_image_marker(
    session: Any | None,
    img_start_token_id: int,
) -> bool:
    metadata = getattr(session, "metadata", {}) or {}
    model_state = metadata.get("ug_model_state") or {}
    u1_state = model_state.get("u1") or {}
    if bool(u1_state.get("open_image_marker")):
        return True
    last_output_ids = metadata.get("srt_last_u_decode_output_ids") or ()
    return bool(last_output_ids) and int(last_output_ids[-1]) == int(img_start_token_id)


def _u1_generated_image_commit_metadata(
    *,
    session: Any | None,
    token_indices: list[int],
    grid_hw: Any,
    omit_start: bool,
    g_position_start: int | None = None,
) -> dict[str, Any]:
    metadata = getattr(session, "metadata", {}) or {}
    model_state = metadata.get("ug_model_state") or {}
    previous_state = model_state.get("u1") or {}
    previous_segments = [
        dict(segment) for segment in previous_state.get("segments", [])
    ]
    u1_segment = {
        "segment_type": "image",
        "source": "native_generated_image_commit",
        "token_indices": list(token_indices),
        "attention_rows": [
            {
                "kind": "image",
                "attention": "hybrid",
                "start": min(token_indices) if token_indices else 0,
                "end": (max(token_indices) + 1) if token_indices else 0,
            }
        ],
        "generated_image_commit": True,
        "native_generated_image_commit": True,
        "omit_image_start": bool(omit_start),
        "image_grid_hw": [list(map(int, row)) for row in grid_hw.tolist()],
    }
    u1_state = {
        "segments": previous_segments + [u1_segment],
        "last_segment_type": "image",
        "last_source": "native_generated_image_commit",
        "last_generated_image_commit": True,
        "native_generated_image_commit": True,
        "open_image_marker": False,
    }
    if g_position_start is not None:
        u1_state["g_position_start"] = int(g_position_start)
    return {
        "u1": u1_segment,
        "ug_model_state_updates": {"u1": u1_state},
    }


def _u1_dynamic_preprocess_native_resolution(
    image: Any,
    *,
    size_factor: int,
    min_pixels: int,
    max_pixels: int,
):
    width, height = image.size
    resized_height, resized_width = _u1_smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return image.resize((resized_width, resized_height))


def _u1_smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _u1_preprocess_pixel_values(pixel_values: Any, *, patch_size: int):
    import torch

    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size
    flatten_pixel_values = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, c * patch_size**2)
    )
    grid_hw = torch.tensor([[grid_h, grid_w]], dtype=torch.long)
    return flatten_pixel_values.to(torch.float32), grid_hw


def _not_wired() -> NotImplementedError:
    return NotImplementedError(
        "SenseNova U1 UG backend is not wired yet. This shell only declares "
        "the pixel_flow capability; U path, G pixel-flow mechanics, and true "
        "weights are covered by later roadmap items."
    )


def _first_u1_image_path(messages: list[UGInterleavedMessage]) -> Path:
    content = _first_u1_image_content(messages)
    if isinstance(content, (str, Path)):
        return Path(content)
    save = getattr(content, "save", None)
    if callable(save):
        path = Path(tempfile.mkdtemp(prefix="u1-vlm-image-")) / "image.png"
        save(path)
        return path
    raise TypeError(
        "U1 VLM image message must be a path or PIL image, " f"got {type(content)}"
    )


def _first_u1_image_content(messages: list[UGInterleavedMessage]) -> Any:
    for message in messages:
        if message.type != "image":
            continue
        return message.content
    raise ValueError("U1 VLM text generation requires an image message")


def _u1_question_text(messages: list[UGInterleavedMessage]) -> str:
    parts = [str(message.content) for message in messages if message.type == "text"]
    question = "\n".join(part for part in parts if part)
    if not question:
        raise ValueError("U1 VLM text generation requires a text question")
    return question


def _tail(text: str | bytes | None, *, limit: int = 2000) -> str | None:
    if text is None:
        return None
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[-limit:]
