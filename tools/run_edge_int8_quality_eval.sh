#!/usr/bin/env bash
set -euo pipefail

eval_root="${1:-outputs/edge_int8_quality}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"

mkdir -p "${eval_root}"
if [[ ! -f "${eval_root}/conditioning.jpg" ]]; then
    curl -4 -fsS -o "${eval_root}/conditioning.jpg" \
        "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/robot_153.jpg"
fi

run_case() {
    local modality="$1"
    local mode="$2"
    local input_file="$3"
    shift 3

    quantization_args=()
    if [[ "${mode}" != "bf16" ]]; then
        quantization_args=("--quantization-method=${mode}")
    fi

    COSMOS_VAE_CPU_OFFLOAD=1 LD_LIBRARY_PATH= "${python_bin}" \
        -m cosmos_framework.scripts.inference \
        --parallelism-preset=latency \
        -i "${input_file}" \
        -o "${eval_root}/${modality}_${mode}" \
        --checkpoint-path=Cosmos3-Edge \
        --use-torch-compile \
        --no-guardrails \
        --benchmark \
        --seed=0 \
        --num-outputs=4 \
        "${quantization_args[@]}" \
        "$@"
}

for mode in bf16 int8wo int8dq; do
    run_case t2i "${mode}" inputs/omni/t2i.json --resolution=480
done

for mode in bf16 int8wo int8dq; do
    run_case i2v "${mode}" inputs/omni/i2v.json --resolution=256 --num-frames=25
done

LD_LIBRARY_PATH= "${python_bin}" tools/edge_int8_quality_eval.py \
    --root "${eval_root}" \
    --conditioning-image "${eval_root}/conditioning.jpg" \
    --output "${eval_root}/metrics.json"
