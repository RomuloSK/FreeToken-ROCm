"""Minimal two-process RCCL acceptance smoke for a ROCm runner.

This is intentionally executable with ``torchrun`` and does not import the
full model stack, so a compatible-driver MI50 host can validate collective
initialisation before spending time on a checkpoint/API test.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    if not torch.version.hip:
        raise SystemExit("RCCL smoke requires a HIP/ROCm Torch build")
    if torch.cuda.device_count() < 2:
        raise SystemExit(f"RCCL smoke requires two visible GPUs, found {torch.cuda.device_count()}")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 2:
        raise SystemExit(f"RCCL smoke expects WORLD_SIZE=2, got {world}")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    for _ in range(3):
        value = torch.tensor([rank + 1], dtype=torch.float32, device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if not torch.equal(value, torch.tensor([3.0], device="cuda")):
            raise SystemExit(f"rank {rank}: RCCL all_reduce returned {value.item()}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
