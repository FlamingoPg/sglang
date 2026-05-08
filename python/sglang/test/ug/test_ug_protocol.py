# SPDX-License-Identifier: Apache-2.0

import dataclasses
import importlib.util
import unittest
from pathlib import Path
from typing import Any

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[3] / "sglang" / "srt" / "ug" / "protocol.py"
)
_SPEC = importlib.util.spec_from_file_location("ug_protocol_for_test", PROTOCOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROTOCOL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROTOCOL)

GeneratedSegment = _PROTOCOL.GeneratedSegment
UGContextRef = _PROTOCOL.UGContextRef
UGCoordinatorRequest = _PROTOCOL.UGCoordinatorRequest
UGCoordinatorResponse = _PROTOCOL.UGCoordinatorResponse


FORBIDDEN_KV_WORDS = ("allocator", "page", "slot")


class TestUGProtocol(unittest.TestCase):
    def test_coordinator_shapes_are_dataclass_serializable(self):
        context = UGContextRef(
            context_id="ctx-1",
            session_id="session-1",
            version=3,
            metadata={"context_length": 128},
        )
        segment = GeneratedSegment(
            type="image",
            image="<image>",
            commit_payload={"kind": "generated_image"},
            metadata={"width": 512, "height": 512},
        )
        response = UGCoordinatorResponse(
            segments=(segment,),
            context=context,
            stats={"prefill_count": 1},
        )

        serialized = dataclasses.asdict(response)

        self.assertEqual("ctx-1", serialized["context"]["context_id"])
        self.assertEqual("image", serialized["segments"][0]["type"])
        self.assertEqual(1, serialized["stats"]["prefill_count"])

    def test_request_and_handles_do_not_expose_raw_kv_fields(self):
        request = UGCoordinatorRequest(
            messages=({"type": "text", "text": "draw a cup"},),
            metadata={"mode": "interleave"},
        )
        context = UGContextRef(
            context_id="ctx-safe",
            session_id="session-safe",
            metadata={"context_length": 64},
        )
        segment = GeneratedSegment(
            type="text",
            text="done",
            metadata={"token_ids": [1, 2, 3]},
        )

        for obj in (request, context, segment):
            self.assertEqual([], _find_forbidden_kv_words(dataclasses.asdict(obj)))
            self.assertEqual([], _find_forbidden_kv_words(_dataclass_field_names(obj)))


def _dataclass_field_names(obj: Any) -> dict[str, None]:
    return {field.name: None for field in dataclasses.fields(obj)}


def _find_forbidden_kv_words(value: Any) -> list[str]:
    found: list[str] = []
    _collect_forbidden_kv_words(value, found=found, path="$")
    return found


def _collect_forbidden_kv_words(value: Any, *, found: list[str], path: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_str = str(key)
            lowered = key_str.lower()
            for word in FORBIDDEN_KV_WORDS:
                if word in lowered:
                    found.append(f"{path}.{key_str}")
            _collect_forbidden_kv_words(nested, found=found, path=f"{path}.{key_str}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _collect_forbidden_kv_words(nested, found=found, path=f"{path}[{index}]")


if __name__ == "__main__":
    unittest.main()
