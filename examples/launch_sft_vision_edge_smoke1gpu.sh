#!/usr/bin/env bash
# Single-GPU, reduced-memory SMOKE launcher for vision_sft_edge on a 23 GB A10G
# shared with other projects. Mirrors launch_sft_vision_edge.sh but points at the
# reduced-token-cap TOML (max_*=6144 instead of 45056) so activations fit in the
# VRAM left over by co-tenant jobs. Pair with COSMOS_SMOKE=1 NPROC_PER_NODE=1.
TOML_FILE="examples/toml/sft_config/vision_sft_edge_smoke1gpu.toml"
: "${DATASET_PATH:=examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
