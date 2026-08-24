#!/usr/bin/env python3
"""Small correctness and timing probe for torchrun/Slurm/PBS GPU jobs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class RankReceipt:
    rank: int
    local_rank: int
    host: str
    device: str
    capability: tuple[int, int]
    total_memory_gib: float
    all_reduce_seconds: float
    all_reduce_algorithm_gbps: float
    all_reduce_bus_gbps: float
    collective_correct: bool
    ddp_loss: float
    ddp_parameter_checksum: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-mb", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--ddp-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return rank, local_rank, world_size


def timed_all_reduce(
    payload_mb: int,
    warmup: int,
    iterations: int,
    world_size: int,
    device: torch.device,
) -> tuple[float, float, float, bool]:
    element_count = max(1, payload_mb * 1024 * 1024 // 4)
    value = torch.ones(element_count, dtype=torch.float32, device=device)
    if world_size == 1:
        return 0.0, 0.0, 0.0, bool(torch.all(value == 1).item())

    for _ in range(warmup):
        value.fill_(1.0)
        dist.all_reduce(value)
    torch.cuda.synchronize(device)
    dist.barrier()

    started = time.perf_counter()
    for _ in range(iterations):
        value.fill_(1.0)
        dist.all_reduce(value)
    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = (time.perf_counter() - started) / iterations
    expected = float(world_size)
    correct = bool(torch.allclose(value, torch.full_like(value, expected)))
    payload_bytes = element_count * value.element_size()
    algorithm_gbps = payload_bytes / elapsed / 1e9
    bus_gbps = algorithm_gbps * 2.0 * (world_size - 1) / world_size
    return elapsed, algorithm_gbps, bus_gbps, correct


def ddp_probe(
    rank: int,
    world_size: int,
    device: torch.device,
    steps: int,
    seed: int,
) -> tuple[float, float, float]:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 64),
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    loss_value = 0.0
    for _ in range(steps):
        inputs = torch.randn(32, 256, generator=generator, device=device)
        targets = torch.randn(32, 64, generator=generator, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())

    checksum = float(
        sum(parameter.detach().float().sum() for parameter in model.parameters()).cpu()
    )
    checksum_tensor = torch.tensor([checksum], dtype=torch.float64, device=device)
    if world_size > 1:
        gathered = [torch.zeros_like(checksum_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, checksum_tensor)
        values = torch.cat(gathered)
        spread = float((values.max() - values.min()).cpu())
    else:
        spread = 0.0
    return loss_value, checksum, spread


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; run this probe inside a GPU allocation")
    rank, local_rank, world_size = distributed_context()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)

    seconds, algorithm_gbps, bus_gbps, collective_correct = timed_all_reduce(
        args.payload_mb,
        args.warmup,
        args.iterations,
        world_size,
        device,
    )
    loss, checksum, spread = ddp_probe(
        rank,
        world_size,
        device,
        args.ddp_steps,
        args.seed,
    )
    receipt = RankReceipt(
        rank=rank,
        local_rank=local_rank,
        host=socket.gethostname(),
        device=properties.name,
        capability=torch.cuda.get_device_capability(device),
        total_memory_gib=properties.total_memory / 1024**3,
        all_reduce_seconds=seconds,
        all_reduce_algorithm_gbps=algorithm_gbps,
        all_reduce_bus_gbps=bus_gbps,
        collective_correct=collective_correct,
        ddp_loss=loss,
        ddp_parameter_checksum=checksum,
    )

    receipts: list[dict[str, object]] = [asdict(receipt)]
    if world_size > 1:
        gathered: list[dict[str, object] | None] | None = (
            [None for _ in range(world_size)] if rank == 0 else None
        )
        dist.gather_object(asdict(receipt), gathered, dst=0)
        if rank == 0:
            receipts = [item for item in gathered or [] if item is not None]

    if rank == 0:
        output = {
            "ok": collective_correct and spread <= 1e-8,
            "world_size": world_size,
            "backend": "nccl" if world_size > 1 else "single-process",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "payload_mb": args.payload_mb,
            "iterations": args.iterations,
            "ddp_parameter_spread": spread,
            "ranks": receipts,
        }
        print(json.dumps(output, indent=2, sort_keys=True), flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
