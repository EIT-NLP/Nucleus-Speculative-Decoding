#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

unset MAX_SAMPLES_PER_TASK
unset SAMPLE_INDICES

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-nvidia/Nemotron-Labs-Diffusion-14B}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/nemotron14b_seed22}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

SEED=22
DATASET_SEED=20260721
MAX_NEW_TOKENS=32768
DATASETS=(gsm8k math500 aime24 aime25 aime26 humaneval mbpp)
METHODS=(AR SD NSD)

if [[ -d "${MODEL}" && ! -d "${MODEL}/linear_spec_lora" ]]; then
    echo "The local model is missing ${MODEL}/linear_spec_lora" >&2
    exit 1
fi

for dataset in "${DATASETS[@]}"; do
    dataset_path="${REPO_ROOT}/eval_datasets/${dataset}.jsonl"
    if [[ ! -s "${dataset_path}" ]]; then
        echo "Missing or empty dataset: ${dataset_path}" >&2
        exit 1
    fi
done

free_port() {
    "${PYTHON_BIN}" - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

for method in "${METHODS[@]}"; do
    case "${method}" in
        AR)
            acceptance_rule=autoregressive
            use_lora=0
            ;;
        SD)
            acceptance_rule=linear_self_speculation
            use_lora=1
            ;;
        NSD)
            acceptance_rule=rejection_sampling_then_target_top_p
            use_lora=1
            ;;
    esac

    for dataset in "${DATASETS[@]}"; do
        output_dir="${RESULT_ROOT}/${method}/${dataset}/seed_${SEED}"
        if [[ "${SKIP_COMPLETED}" == 1 \
              && -s "${output_dir}/quality_metrics.json" \
              && -s "${output_dir}/throughput_metrics.json" ]]; then
            echo "Skipping completed cell: ${output_dir}"
            continue
        fi

        echo "============================================================"
        echo "Model=${MODEL}"
        echo "Method=${method} (${acceptance_rule})"
        echo "Dataset=${dataset}; seed=${SEED}"
        echo "Temperature=1.0; target Top-P=1.0; max new tokens=32768"
        echo "Output=${output_dir}"
        echo "============================================================"

        env \
            INFERENCE_BACKEND=transformers \
            PYTHON_BIN="${PYTHON_BIN}" \
            CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
            MASTER_ADDR=127.0.0.1 \
            MASTER_PORT="$(free_port)" \
            TARGET_MODEL="${MODEL}" \
            DRAFT_MODEL="${MODEL}" \
            SOURCE_DATASET_ROOT="${REPO_ROOT}/eval_datasets" \
            TASKS="${dataset}" \
            SEED="${SEED}" \
            DATASET_SEED="${DATASET_SEED}" \
            MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
            TEMPERATURE=1.0 \
            ACCEPTANCE_RULE="${acceptance_rule}" \
            ACCEPTANCE_TOP_P=0.95 \
            ACCEPTANCE_TOP_K=2 \
            TARGET_SAMPLING_TOP_P=1.0 \
            NEMOTRON_LINEAR_SPEC_BLOCK_LENGTH=32 \
            NEMOTRON_LINEAR_SPEC_THRESHOLD=0.0 \
            NEMOTRON_LINEAR_SPEC_USE_LORA="${use_lora}" \
            NEMOTRON_LINEAR_SPEC_LORA_SUBFOLDER=linear_spec_lora \
            RECORD_TOKEN_DIAGNOSTICS=0 \
            RECORD_EXACT_NSD_RHO=0 \
            RESULT_ROOT="${RESULT_ROOT}/${method}" \
            RUN_NAME="${dataset}" \
            bash scripts/eval/run_four_benchmarks_with_throughput.sh
    done
done

"${PYTHON_BIN}" scripts/summarize_results.py \
    --result-root "${RESULT_ROOT}" \
    --output "${RESULT_ROOT}/summary.json"

echo "Finished. Summary: ${RESULT_ROOT}/summary.json"
