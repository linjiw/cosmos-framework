#!/usr/bin/env bash
# Single-GPU 24GB-optimized launcher for vision_sft_edge (see OPTIMIZATION_PLAN.md).
# cap 12288 + grad_accum 61 reproduces the stock effective batch (282 vs 283 samples/step)
# with zero sample drops/truncation (max Bridge sample = 3374 tokens).
# MEMTEST=1 switches to the grad_accum=2 variant (memory-identical, much faster smoke).
if [[ -n "${MEMTEST:-}" ]]; then
    TOML_FILE="examples/toml/sft_config/vision_sft_edge_24gb_memtest.toml"
elif [[ -n "${EMAVAL:-}" ]]; then
    TOML_FILE="examples/toml/sft_config/vision_sft_edge_24gb_emaval.toml"
else
    TOML_FILE="examples/toml/sft_config/vision_sft_edge_24gb.toml"
fi
: "${DATASET_PATH:=examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
: "${NPROC_PER_NODE:=1}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
