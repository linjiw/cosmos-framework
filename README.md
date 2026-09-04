<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">
    <a href="https://github.com/NVIDIA/Cosmos">NVIDIA Cosmos</a> |
    🤗 <a href="https://huggingface.co/collections/nvidia/cosmos3">Cosmos 3 </a>
</p>

<p align="center">
    Part of the <a href="https://github.com/NVIDIA/Cosmos">NVIDIA Cosmos</a> project family — the training and serving framework repository.
</p>

# Cosmos-Framework

**Cosmos-Framework** is an end-to-end framework for training and serving world models, including the **Cosmos3** model family. Everything lives in a single top-level [`cosmos_framework/`](./cosmos_framework) Python package:

- **Training** — distributed FSDP / TP / CP / PP trainer, native DCP checkpoints with HuggingFace `safetensors` import/export, JSONL / WebDataset / LeRobot dataset adapters. Entry point: `cosmos_framework.scripts.train`. See [`docs/training.md`](./docs/training.md).
- **Inference** — Diffusers / Transformers / vLLM backends with offline batch generation and online serving (Ray + Gradio). Entry point: `cosmos_framework.scripts.inference`. Ecosystem-facing shim libraries (lightweight standalone wrappers for downstream projects) live under [`packages/`](./packages).

## Cosmos 3

**Cosmos 3** is our newest model family [[Report]](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) [[Website]](https://research.nvidia.com/labs/cosmos-lab/cosmos3/). It is a suite of omnimodal world models designed to jointly process and generate language, images, video, audio, and action sequences within a unified Mixture-of-Transformers architecture. By supporting highly flexible input-output configurations, it seamlessly unifies critical modalities for Physical AI — effectively subsuming vision-language models, video generators, world simulators, and world-action models into a single framework. For a guided experience to test out Cosmos3, please visit [[Cosmos]](https://github.com/nvidia/cosmos).

## Framework Documentation

- [Quickstart](#setup)
- [Setup](./docs/setup.md)
- [Training (Supervised Fine-Tuning)](./docs/training.md)
  - [JSONL Dataset](./docs/dataset_jsonl.md)
- [Inference](./docs/inference.md)
- [Local INT8 Research Showcase](./showcase/README.md)
- [Policy Server](./docs/action_policy_droid_server.md)
- [Agent Skills](#agent-skills)
- Reference
  - [Code Structure](./docs/code_structure.md)
  - [Environment Variables](./docs/environment_variables.md)
  - [FAQ](./docs/faq.md)
  - [AGENTS.md — repo map for coding agents](./AGENTS.md)

## Setup

For more details and alternative installation methods, see [Setup](./docs/setup.md#installation). Before installing, make sure your machine meets the [System Requirements](./docs/setup.md#system-requirements). If you want a curated PyTorch + CUDA environment, start from the [recommended NVIDIA NGC base image](./docs/setup.md#recommended-base-image).

Install system dependencies:

```shell
sudo apt-get install -y --no-install-recommends curl ffmpeg git-lfs libx11-dev tree wget
```

Install the package with `uv` (pick the dependency group that matches your CUDA toolkit — see [CUDA Variants](./docs/setup.md#cuda-variants)):

```shell
# CUDA 13.0 (recommended)
uv sync --all-extras --group=cu130-train
# Or, for CUDA 12.8:
# uv sync --all-extras --group=cu128-train
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

If you are starting from the recommended NGC image (`nvcr.io/nvidia/pytorch:25.09-py3`), see the [one-shot quickstart](./docs/setup.md#quickstart-from-the-recommended-base-image).

## Training

For the full guide (data preparation, base-checkpoint conversion, parallelism strategies, mixed precision, resuming), see [Training](./docs/training.md). The number of GPUs required depends on the recipe; the shipped recipes under [`examples/`](./examples/README.md) are 8-GPU configurations (tested on 8× H100 80 GB) launched via their paired launch shells, e.g.:

```shell
bash examples/launch_sft_vision_nano.sh
```

Users may adjust the GPU count to match their model and underlying hardware architecture — tune `NPROC_PER_NODE` and the parallelism degrees (DP/CP/FSDP shard) in the recipe accordingly.

## Inference

See [Inference](./docs/inference.md) for the full guide — launch commands, supported modes, parallelism presets, and troubleshooting.

Quick single-GPU launch:

```shell
python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/omni/t2v.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

### Low-memory Cosmos3-Edge inference

This optimization adds streamed INT8 checkpoint loading and optional vision-VAE CPU
offload so Cosmos3-Edge can run on memory-constrained GPUs without first
materializing the complete BF16 model on CUDA. `int8wo` keeps BF16 activations;
`int8dq` also dynamically quantizes activations for W8A8 tensor-core execution.
See the [inference guide](./docs/inference.md#cosmos3-edge) for the command.

Measured on one GeForce RTX 5080 (16,303 MiB), using the repository `t2i`
example at 480p, seed 0, 35 UniPC steps, VAE CPU offload, and one compile warmup:

| Weight path | Peak GPU memory | Batch latency | Versus compiled BF16 |
| ----------- | --------------: | ------------: | -------------------- |
| BF16 + compile | 10,048 MiB | 5.60 s | baseline |
| INT8WO + compile | 7,374 MiB | 6.09 s | 26.6% less memory; 8.7% slower |
| INT8DQ + compile | 7,358 MiB | 4.18 s | 26.8% less memory; 25.4% faster |

INT8WO remains the conservative recommendation while INT8DQ undergoes a
multi-seed T2I/I2V quality evaluation. Peak memory is process VRAM sampled at
100 ms; latency excludes model loading and compilation warmup. The reproducible
four-seed runner and paired metric gate live in
[`tools/run_edge_int8_quality_eval.sh`](./tools/run_edge_int8_quality_eval.sh)
and [`tools/edge_int8_quality_eval.py`](./tools/edge_int8_quality_eval.py).

#### Optimized example outputs

The outputs below use streamed INT8WO loading, VAE CPU offload, torch compile,
and seed 0 with the repository's existing [`t2i`](./inputs/omni/t2i.json) and
[`i2v`](./inputs/omni/i2v.json) prompts.

| Text-to-image (480p, 35 steps) | Image-to-video (256p, 25 frames, 35 steps) |
| ------------------------------ | ------------------------------------------ |
| ![INT8WO text-to-image result: a robotic arm in a bright laboratory](./docs/assets/cosmos3-edge-int8/t2i-int8wo-seed0.jpg) | [![INT8WO image-to-video frame strip showing a robotic arm reaching toward a red ball](./docs/assets/cosmos3-edge-int8/i2v-int8wo-seed0-strip.jpg)](./docs/assets/cosmos3-edge-int8/i2v-int8wo-seed0.mp4) |

The I2V run peaked at 6,456 MiB and completed in 18.62 seconds; select its frame
strip to open the generated MP4.

For an interactive local presentation of the architecture, benchmarks, paired
T2I outputs, and playable I2V result, see the
[INT8 research showcase](./showcase/README.md).

## Policy Server

See [Policy Server](./docs/action_policy_droid_server.md) for the full guide.

## Agent Skills

Coding agents (Claude Code, Codex CLI, Cursor, and other AGENTS.md-aware
tools) can load task-specific instructions for this repo from the bundled
AgentSkills. Each skill is a self-contained `SKILL.md` that the agent
invokes automatically when the user's request matches its description.

Skills live in [`.agents/skills/`](./.agents/skills) (canonical) and are
mirrored under [`.claude/skills/`](./.claude/skills) for Claude Code:

| Skill                                                                            | When it activates                                                                    |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| [`cosmos3-setup`](./.agents/skills/cosmos3-setup/SKILL.md)                       | Installation, environment setup, checkpoint downloads, Docker.                       |
| [`cosmos3-codebase-nav`](./.agents/skills/cosmos3-codebase-nav/SKILL.md)         | "Where is X" / "where do I change parameter Y" questions across `cosmos_framework/`. |
| [`cosmos3-inference`](./.agents/skills/cosmos3-inference/SKILL.md)               | Running offline or online inference, parallelism, sampling parameters.               |
| [`cosmos3-post-training`](./.agents/skills/cosmos3-post-training/SKILL.md)       | SFT post-training end-to-end: data prep, DCP conversion, launch, export.             |
| [`cosmos3-env-troubleshoot`](./.agents/skills/cosmos3-env-troubleshoot/SKILL.md) | Diagnosing install/runtime errors (ImportError, CUDA, Docker, checkpoint failures).  |

See [`AGENTS.md`](./AGENTS.md) for the canonical repo map that agents load
first; the skills above are referenced from there.

## Reference

| Topic                                                        | What it covers                                                                                                                            |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| [Setup](./docs/setup.md)                                     | Hardware/software prerequisites, `uv` install paths, CUDA variants, Docker base image, and base-checkpoint downloading.                   |
| [Code Structure](./docs/code_structure.md)                   | Repository layout and a per-subpackage tour of `cosmos_framework/` — where each concern lives and where to add new code.                  |
| [Training](./docs/training.md)                               | Launching multi-GPU and multi-node runs; parallelism strategies; mixed precision; resuming.                                               |
| [Inference (from a trained checkpoint)](./docs/inference.md) | Loading a trained checkpoint into one of the inference backends.                                                                          |
| [Policy Server](./docs/action_policy_droid_server.md)        | Running the server-client pipeline for Cosmos3-Policy-DROID.                                                                              |
| [FAQ](./docs/faq.md)                                         | Troubleshooting (OOM, NCCL hangs, slow training), environment variables, and common pitfalls.                                             |
| [AGENTS.md](./AGENTS.md) + [Agent Skills](#agent-skills)     | Repo map and task-specific `SKILL.md` files loaded automatically by AGENTS.md-aware coding agents (Claude Code, Codex CLI, Cursor, etc.). |
