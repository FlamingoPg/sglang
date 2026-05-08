# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_coordinator_module():
    path = REPO_ROOT / "python" / "sglang" / "srt" / "ug" / "coordinator.py"
    spec = importlib.util.spec_from_file_location("ug_coordinator_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestUGInterleaveCoordinator(unittest.TestCase):
    def test_interleave_commits_generated_image_before_continuing_u(self):
        module = _load_coordinator_module()
        contexts = SimpleNamespace(full=SimpleNamespace(session_id="s0"))
        bridge = _FakeBridge(
            continuation=[
                SimpleNamespace(type="text", text="after", token_ids=(7,)),
                SimpleNamespace(type="done", text=None, token_ids=()),
            ]
        )
        calls = []

        def g_segment_executor(*, bridge, contexts, batch, server_args):
            calls.append(("g", contexts, batch, server_args))
            return SimpleNamespace(type="image", image="image0", metadata={"step": 0})

        coordinator = module.UGInterleaveCoordinator(
            bridge=bridge,
            g_segment_executor=g_segment_executor,
        )
        batch = SimpleNamespace(
            extra={"ug_pre_image_segments": [{"type": "text", "text": "before"}]}
        )
        server_args = SimpleNamespace(name="server")

        segments = coordinator.run_generation(
            batch=batch,
            contexts=contexts,
            server_args=server_args,
            metadata={
                "mode": "interleave",
                "max_interleave_images": 1,
                "max_interleave_text_segments": 2,
            },
        )

        self.assertEqual(
            [
                {"type": "text", "text": "before"},
                {"type": "image", "image": "image0", "metadata": {"step": 0}},
                {"type": "text", "text": "after", "metadata": {"token_ids": [7]}},
            ],
            segments,
        )
        self.assertEqual(
            [
                ("run_g", contexts),
                ("commit", contexts),
                ("decode", contexts),
                ("decode", contexts),
            ],
            bridge.calls,
        )
        self.assertEqual([("g", contexts, batch, server_args)], calls)


class _FakeBridge:
    def __init__(self, continuation):
        self.continuation = list(continuation)
        self.calls = []

    def run_g_segment(self, *, contexts, executor):
        self.calls.append(("run_g", contexts))
        return executor(contexts)

    def commit_generated_segment(self, *, contexts, segment):
        self.calls.append(("commit", contexts))
        self.committed_segment = segment

    def continue_u_decode(self, *, contexts):
        self.calls.append(("decode", contexts))
        return self.continuation.pop(0)


if __name__ == "__main__":
    unittest.main()
