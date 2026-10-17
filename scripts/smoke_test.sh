#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-nvidia/Nemotron-Labs-Diffusion-14B}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/smoke}"

if [[ ! -s "${REPO_ROOT}/eval_datasets/aime25.jsonl" ]]; then
    echo "Missing or empty dataset: ${REPO_ROOT}/eval_datasets/aime25.jsonl" >&2
    exit 1
fi

for row in "AR autoregressive 0" \
           "SD linear_self_speculation 1" \
           "NSD rejection_sampling_then_target_top_p 1"; do
    read -r method rule use_lora <<< "${row}"
    env \
        INFERENCE_BACKEND=transformers \
        PYTHON_BIN="${PYTHON_BIN}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        MASTER_ADDR=127.0.0.1 \
        MASTER_PORT="$((39000 + RANDOM % 1000))" \
        TARGET_MODEL="${MODEL}" \
        DRAFT_MODEL="${MODEL}" \
        SOURCE_DATASET_ROOT="${REPO_ROOT}/eval_datasets" \
        TASKS=aime25 \
        SAMPLE_INDICES=0 \
        SEED=22 \
        DATASET_SEED=20260721 \
        MAX_NEW_TOKENS=16 \
        TEMPERATURE=1.0 \
        ACCEPTANCE_RULE="${rule}" \
        ACCEPTANCE_TOP_P=0.95 \
        TARGET_SAMPLING_TOP_P=1.0 \
        NEMOTRON_LINEAR_SPEC_BLOCK_LENGTH=32 \
        NEMOTRON_LINEAR_SPEC_THRESHOLD=0.0 \
        NEMOTRON_LINEAR_SPEC_USE_LORA="${use_lora}" \
        NEMOTRON_LINEAR_SPEC_LORA_SUBFOLDER=linear_spec_lora \
        RECORD_TOKEN_DIAGNOSTICS=0 \
        RESULT_ROOT="${RESULT_ROOT}/${method}" \
        RUN_NAME=aime25 \
        bash scripts/eval/run_four_benchmarks_with_throughput.sh
done
