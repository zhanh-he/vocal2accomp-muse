# Cluster Deployment and Smoke Tests

This directory qualifies compute placement before any Muse reward-training run.
The smoke test checks device visibility, NCCL collectives, DDP gradient updates,
and per-rank provenance. Passing it proves that the distributed substrate works;
it does not prove that Muse, MuCodec, source separation, or a reward service fits.

## Current placement hypothesis

| host | topology | intended work |
| --- | --- | --- |
| lab5090 | one RTX 5090, 32 GB | environment setup, E1/E2 scoring, Muse generation and single-GPU LoRA feasibility |
| Kaya | two V100 per node | large offline scoring; 2-GPU same-node fallback; 4-GPU only after a 2-node NCCL test |
| Gadi gpuvolta | four V100-32GB per node | same-node 4-GPU compatibility and budget-efficient pilot |
| Gadi gpuhopper | four H200-141GB per node | preferred 4-GPU Muse GRPO once the stack and queue are qualified |

The 4-GPU Kaya launcher deliberately requests two nodes. Do not interpret it as
a same-node test. Gadi is the cleaner four-card training target because both GPU
queues expose four devices within one node.

## Smoke program

`distributed_smoke.py` runs four checks:

1. one process is bound to each visible GPU;
2. NCCL all-reduce returns the expected value;
3. a small DDP model completes synchronized optimizer steps;
4. rank-zero emits one JSON receipt with hosts, devices, timings, and versions.

Important outputs:

- `all_reduce_algorithm_gbps`: payload bytes divided by collective latency;
- `all_reduce_bus_gbps`: ring-normalized communication bandwidth;
- `ddp_parameter_spread`: maximum cross-rank parameter disagreement after DDP;
- `ok`: all collective and DDP correctness checks passed.

## Local or lab5090

```bash
PYTHON_BIN=/path/to/python \
  bash experiments/open_reward_safety/cluster/launchers/lab5090_single_gpu.sh
```

## Kaya

Submit from the repository root. The existing CUDA PyTorch environment can be
selected without changing the public script:

```bash
PYTHON_BIN=/group/<project>/<user>/envs/<env>/bin/python \
  sbatch experiments/open_reward_safety/cluster/launchers/kaya_2gpu.sbatch

PYTHON_BIN=/group/<project>/<user>/envs/<env>/bin/python \
  sbatch experiments/open_reward_safety/cluster/launchers/kaya_4gpu_2node.sbatch
```

The two-node launcher uses one Slurm task per node. Each task starts a local
`torchrun` worker group, with `SLURM_NODEID` mapped to `node_rank`.

## Gadi

The launchers use the system PyTorch module and default to project `wa66`.
Override `PROJECT` at submission if another allocation should be charged.

```bash
qsub -P wa66 experiments/open_reward_safety/cluster/launchers/gadi_gpuvolta_4gpu.pbs
qsub -P wa66 experiments/open_reward_safety/cluster/launchers/gadi_gpuhopper_4gpu.pbs
qsub -P wa66 experiments/open_reward_safety/cluster/launchers/gadi_muse_token_h200.pbs
```

Gadi compute nodes have no external network access. Repositories, wheels, model
weights, and datasets must be staged under the declared `/g/data` or `/scratch`
storage before the GPU job starts.

The H200 token smoke uses the isolated CUDA 12.8 environment, loads the staged
Muse checkpoint, and writes a machine-readable receipt. It is intentionally
separate from the four-GPU NCCL/DDP smoke.

## Promotion gates

| gate | requirement |
| --- | --- |
| single GPU | CUDA allocation and DDP step complete |
| same-node multi-GPU | `ok=true`, zero parameter spread, no NCCL warning/error |
| cross-node Kaya | both hosts represented, world size four, zero spread, stable repeated collectives |
| Muse dry-run | one prompt, K=2, shortened completion, decode succeeds |
| reward dry-run | decoded audio produces component scores and an auditable validity decision |

The full Muse pilot starts only after these gates and a measured memory receipt.
