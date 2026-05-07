# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class UGSRTSchedulerHandle:
    """Owns the SRT scheduler used by the standalone UG smoke path."""

    scheduler: Any

    def close(self) -> None:
        for name in (
            "recv_from_tokenizer",
            "recv_from_rpc",
            "send_metrics_from_scheduler",
        ):
            socket = getattr(self.scheduler, name, None)
            close = getattr(socket, "close", None)
            if callable(close):
                close(linger=0)


class _NoopSender:
    def send_output(self, *args, **kwargs):
        del args, kwargs


def is_real_u1_ug_model(model_path: str | None, model_id: str | None = None) -> bool:
    identifier = " ".join(str(value or "") for value in (model_path, model_id)).lower()
    return "sensenova-u1" in identifier or "sensenova_u1" in identifier


def create_u1_srt_scheduler(
    *,
    checkpoint_dir: str,
    gpu_id: int = 0,
    dtype: str = "bfloat16",
    mem_fraction_static: float = 0.35,
    chunked_prefill_size: int = -1,
    attention_backend: str | None = None,
    log_level: str = "error",
) -> UGSRTSchedulerHandle:
    """Create the SRT Scheduler used as the SenseNova U1 UG session owner."""

    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import (
        PortArgs,
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    server_args = ServerArgs(
        model_path=str(Path(checkpoint_dir).expanduser()),
        tokenizer_path=str(Path(checkpoint_dir).expanduser()),
        trust_remote_code=False,
        dtype=dtype,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        disable_overlap_schedule=True,
        skip_server_warmup=True,
        # U1 official interleave uses SDPA-style attention. Triton is faster,
        # but the small logits drift can flip near-tie U decode tokens and send
        # the closed-loop image trajectory down a different branch.
        attention_backend=attention_backend or "torch_native",
        mem_fraction_static=float(mem_fraction_static),
        # U1 edit/VLM prefixes use block-causal image context where tokens with
        # the same t index attend bidirectionally. Splitting that image block
        # across chunked-prefill boundaries commits earlier KV before later
        # same-t tokens exist, so keep U1 UG prefill unchunked for parity.
        chunked_prefill_size=int(chunked_prefill_size),
        log_level=log_level,
    )
    server_args.check_server_args()
    set_global_server_args_for_scheduler(server_args)

    scheduler = Scheduler(
        server_args,
        PortArgs.init_new(server_args),
        gpu_id=int(gpu_id),
        tp_rank=0,
        moe_ep_rank=0,
        pp_rank=0,
        attn_cp_rank=0,
        moe_dp_rank=0,
        dp_rank=None,
    )
    _replace_sender_with_noop(scheduler, "send_to_tokenizer")
    _replace_sender_with_noop(scheduler, "send_to_detokenizer")
    return UGSRTSchedulerHandle(scheduler=scheduler)


def _replace_sender_with_noop(scheduler: Any, name: str) -> None:
    sender = getattr(scheduler, name, None)
    socket = getattr(sender, "socket", None)
    if socket is not None:
        socket.close(linger=0)
    setattr(scheduler, name, _NoopSender())
