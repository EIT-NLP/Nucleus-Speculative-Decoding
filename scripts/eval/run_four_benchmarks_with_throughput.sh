#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-transformers}"
TARGET_MODEL="${TARGET_MODEL:-nvidia/Nemotron-Labs-Diffusion-14B}"
DRAFT_MODEL="${DRAFT_MODEL:-${TARGET_MODEL}}"
SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-${REPO_ROOT}/eval_datasets}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/four_benchmark_eval}"

TASKS="${TASKS:-gsm8k,math500,aime24,aime25,aime26,mbpp,humaneval}"
SEED="${SEED:-3}"
DATASET_SEED="${DATASET_SEED:-20260721}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-1.0}"
ACCEPTANCE_RULE="${ACCEPTANCE_RULE:-rejection_sampling_then_target_top_p}"
ACCEPTANCE_TOP_P="${ACCEPTANCE_TOP_P:-0.95}"
DYNAMIC_TOP_P_BLEND="${DYNAMIC_TOP_P_BLEND:-0.0}"
DYNAMIC_TOP_P_SUPPORT="${DYNAMIC_TOP_P_SUPPORT:-0.95}"
DYNAMIC_TOP_P_OUTER_ALPHA="${DYNAMIC_TOP_P_OUTER_ALPHA:-0.0}"
DYNAMIC_TOP_P_SIGNAL="${DYNAMIC_TOP_P_SIGNAL:-entropy_disagreement}"
RISK_BUDGET_MAX="${RISK_BUDGET_MAX:-1.0}"
RISK_BUDGET_SAFE_TOP_P="${RISK_BUDGET_SAFE_TOP_P:-0.90}"
ACCEPTANCE_TOP_K="${ACCEPTANCE_TOP_K:-2}"
CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
MARS_RELAXATION_THRESHOLD="${MARS_RELAXATION_THRESHOLD:-0.90}"
TARGET_SAMPLING_TOP_P="${TARGET_SAMPLING_TOP_P:-1.0}"
EAGLE3_PROPOSAL_TOKENS="${EAGLE3_PROPOSAL_TOKENS:-7}"
MTP_PROPOSAL_TOKENS="${MTP_PROPOSAL_TOKENS:-2}"
QWEN35_VLLM_MAX_MODEL_LEN="${QWEN35_VLLM_MAX_MODEL_LEN:-16384}"
QWEN35_VLLM_GPU_MEMORY_UTILIZATION="${QWEN35_VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
QWEN35_VLLM_ENFORCE_EAGER="${QWEN35_VLLM_ENFORCE_EAGER:-1}"
NEMOTRON_LINEAR_SPEC_BLOCK_LENGTH="${NEMOTRON_LINEAR_SPEC_BLOCK_LENGTH:-32}"
NEMOTRON_LINEAR_SPEC_THRESHOLD="${NEMOTRON_LINEAR_SPEC_THRESHOLD:-0.0}"
NEMOTRON_LINEAR_SPEC_USE_LORA="${NEMOTRON_LINEAR_SPEC_USE_LORA:-0}"
NEMOTRON_LINEAR_SPEC_LORA_SUBFOLDER="${NEMOTRON_LINEAR_SPEC_LORA_SUBFOLDER:-linear_spec_lora}"
RECORD_VERIFICATION_TIMING="${RECORD_VERIFICATION_TIMING:-0}"
VERIFICATION_TIMING_WARMUP_SAMPLES="${VERIFICATION_TIMING_WARMUP_SAMPLES:-1}"
BLIND_ACCEPT_EVERY_NTH_DRAFT_BLOCK="${BLIND_ACCEPT_EVERY_NTH_DRAFT_BLOCK:-0}"
if [[ -n "${RUN_NAME:-}" ]]; then
    RUN_NAME="${RUN_NAME}"
elif [[ "${ACCEPTANCE_RULE}" == "cactus" ]]; then
    RUN_NAME="cactus_delta${CACTUS_DELTA}"
elif [[ "${ACCEPTANCE_RULE}" == "mars" ]]; then
    RUN_NAME="mars_threshold${MARS_RELAXATION_THRESHOLD}"
elif [[ "${ACCEPTANCE_RULE}" == "rejection_sampling_then_dynamic_top_p" ]]; then
    RUN_NAME="${ACCEPTANCE_RULE}_${DYNAMIC_TOP_P_SIGNAL}_blend${DYNAMIC_TOP_P_BLEND}_support${DYNAMIC_TOP_P_SUPPORT}_alpha${DYNAMIC_TOP_P_OUTER_ALPHA}"
elif [[ "${ACCEPTANCE_RULE}" == "rejection_sampling_then_risk_budget_top_p" ]]; then
    RUN_NAME="${ACCEPTANCE_RULE}_budget${RISK_BUDGET_MAX}_safe${RISK_BUDGET_SAFE_TOP_P}_p${ACCEPTANCE_TOP_P}"
elif [[ "${ACCEPTANCE_RULE}" == "rejection_sampling_then_entropy_risk_budget_top_p" ]]; then
    RUN_NAME="${ACCEPTANCE_RULE}_alpha${DYNAMIC_TOP_P_OUTER_ALPHA}_budget${RISK_BUDGET_MAX}_safe${RISK_BUDGET_SAFE_TOP_P}_support${DYNAMIC_TOP_P_SUPPORT}"
elif [[ "${BLIND_ACCEPT_EVERY_NTH_DRAFT_BLOCK}" -gt 0 ]]; then
    RUN_NAME="${ACCEPTANCE_RULE}_blind_every${BLIND_ACCEPT_EVERY_NTH_DRAFT_BLOCK}"
else
    RUN_NAME="${ACCEPTANCE_RULE}_p${ACCEPTANCE_TOP_P}"
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29764}"
CODE_WORKERS="${CODE_WORKERS:-4}"
HUMANEVAL_TIMEOUT="${HUMANEVAL_TIMEOUT:-${CODE_TIMEOUT:-3}}"
MBPP_TIMEOUT="${MBPP_TIMEOUT:-${CODE_TIMEOUT:-10}}"

OUTPUT_DIR="${RESULT_ROOT}/${RUN_NAME}/seed_${SEED}"
COMBINED_DATASET_ROOT="${OUTPUT_DIR}/datasets"
CODE_CONTRACT_DATASET_ROOT="${COMBINED_DATASET_ROOT}/dspark_code_contract"
LOG_PATH="${OUTPUT_DIR}/run.log"
mkdir -p "${COMBINED_DATASET_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

export CUDA_VISIBLE_DEVICES
export MASTER_ADDR
export MASTER_PORT
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/opencompass${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -r -a task_array <<< "${TASKS}"
code_task_requested=0
for task in "${task_array[@]}"; do
    case "${task}" in
        gsm8k|gsm_hard|math500|aime24|aime25|aime26|gpqa_diamond|arc_challenge|mmlu|mmlu_pro|commonsense_qa|openbookqa|piqa) ;;
        mbpp|humaneval)
            code_task_requested=1
            ;;
        *)
            echo "Unsupported task for this wrapper: ${task}" >&2
            exit 1
            ;;
    esac
done

if [[ "${code_task_requested}" -eq 1 ]]; then
    echo "Tagging the original DSpark MBPP/HumanEval prompts for functional scoring"
    "${PYTHON_BIN}" scripts/eval/build_dspark_code_datasets.py \
        --source-root "${SOURCE_DATASET_ROOT}" \
        --output-root "${CODE_CONTRACT_DATASET_ROOT}"
fi

for task in "${task_array[@]}"; do
    case "${task}" in
        gsm8k|gsm_hard|math500|aime24|aime25|aime26|gpqa_diamond|arc_challenge|mmlu|mmlu_pro|commonsense_qa|openbookqa|piqa)
            source_path="${SOURCE_DATASET_ROOT}/${task}.jsonl"
            ;;
        mbpp)
            source_path="${CODE_CONTRACT_DATASET_ROOT}/mbpp.jsonl"
            ;;
        humaneval)
            source_path="${CODE_CONTRACT_DATASET_ROOT}/humaneval.jsonl"
            ;;
    esac
    if [[ ! -s "${source_path}" ]]; then
        echo "Missing or empty dataset: ${source_path}" >&2
        exit 1
    fi
    cp -f -- "${source_path}" "${COMBINED_DATASET_ROOT}/${task}.jsonl"
done

eval_extra_args=()
if [[ "${ACCEPTANCE_RULE}" == asd ]]; then
    [[ "${INFERENCE_BACKEND}" == transformers ]] || {
        echo 'ASD is supported only by Transformers, not vLLM.' >&2; exit 1;
    }
    eval_extra_args+=(--asd-budget "${ASD_BUDGET:-8.0}" --asd-gate "${ASD_GATE:-0.25}"
                     --asd-max-exceptions "${ASD_MAX_EXCEPTIONS:-2}"
                     --asd-value-scheme "${ASD_VALUE_SCHEME:-suffix_length}")
fi
if [[ -n "${AUTOREGRESSIVE_TOP_K:-}" ]]; then
    if [[ "${INFERENCE_BACKEND}" != "transformers" ]]; then
        echo "AUTOREGRESSIVE_TOP_K is supported only by the Transformers AR evaluator" >&2
        exit 1
    fi
    eval_extra_args+=(--autoregressive-top-k "${AUTOREGRESSIVE_TOP_K}")
fi
if [[ -n "${MAX_SAMPLES_PER_TASK:-}" ]]; then
    eval_extra_args+=(--max-samples-per-task "${MAX_SAMPLES_PER_TASK}")
fi
if [[ -n "${SAMPLE_INDICES:-}" ]]; then
    eval_extra_args+=(--sample-indices "${SAMPLE_INDICES}")
fi
if [[ "${RECORD_TOKEN_DIAGNOSTICS:-0}" == "1" ]]; then
    eval_extra_args+=(
        --record-token-diagnostics
        --token-diagnostics-mode "${TOKEN_DIAGNOSTICS_MODE:-all}"
        --token-diagnostics-tasks "${TOKEN_DIAGNOSTICS_TASKS:-${TASKS}}"
    )
fi
if [[ "${RECORD_EXACT_NSD_RHO:-0}" == "1" ]]; then
    if [[ "${RECORD_TOKEN_DIAGNOSTICS:-0}" != "1" ]]; then
        echo "RECORD_EXACT_NSD_RHO=1 requires RECORD_TOKEN_DIAGNOSTICS=1" >&2
        exit 1
    fi
    eval_extra_args+=(--record-exact-nsd-rho)
fi
if [[ "${RECORD_VERIFICATION_TIMING}" == "1" ]]; then
    [[ "${INFERENCE_BACKEND}" == transformers ]] || {
        echo 'Verification phase timing is supported only by Transformers.' >&2; exit 1;
    }
    eval_extra_args+=(--record-verification-timing
                     --verification-timing-warmup-samples
                     "${VERIFICATION_TIMING_WARMUP_SAMPLES}")
fi

echo "Running ${TASKS}"
echo "Output: ${OUTPUT_DIR}"
echo "Inference backend: ${INFERENCE_BACKEND}"
echo "Throughput: generated output tokens / dataset inference wall time"
echo "Code evaluation profile: original DSpark prompts, DSpark-aware extraction, functional tests"
echo "Blind-accept period: ${BLIND_ACCEPT_EVERY_NTH_DRAFT_BLOCK} (0=disabled)"
echo "Cactus delta: ${CACTUS_DELTA}"
echo "MARS relaxation threshold: ${MARS_RELAXATION_THRESHOLD}"
echo "Dynamic Top-P blend: ${DYNAMIC_TOP_P_BLEND}"
echo "Dynamic Top-P support: ${DYNAMIC_TOP_P_SUPPORT}"
echo "Dynamic Top-P outer alpha: ${DYNAMIC_TOP_P_OUTER_ALPHA}"
echo "Dynamic Top-P signal: ${DYNAMIC_TOP_P_SIGNAL}"
echo "Risk budget max: ${RISK_BUDGET_MAX}"
echo "Risk budget safe Top-P: ${RISK_BUDGET_SAFE_TOP_P}"
echo "EAGLE3 proposal tokens: ${EAGLE3_PROPOSAL_TOKENS}"
echo "Qwen3.5 MTP proposal tokens: ${MTP_PROPOSAL_TOKENS}"
echo "Nemotron Linear Self-Speculation block length: ${NEMOTRON_LINEAR_SPEC_BLOCK_LENGTH}"
echo "Nemotron Linear Self-Speculation threshold: ${NEMOTRON_LINEAR_SPEC_THRESHOLD}"
echo "Nemotron Linear Self-Speculation LoRA: ${NEMOTRON_LINEAR_SPEC_USE_LORA}"
echo "Nemotron Linear Self-Speculation LoRA subfolder: ${NEMOTRON_LINEAR_SPEC_LORA_SUBFOLDER}"

if [[ "${INFERENCE_BACKEND}" == "vllm" ]]; then
    vllm_extra_args=()
    if [[ -n "${MAX_SAMPLES_PER_TASK:-}" ]]; then
        vllm_extra_args+=(--max-samples-per-task "${MAX_SAMPLES_PER_TASK}")
    fi
    if [[ -n "${SAMPLE_INDICES:-}" ]]; then
        vllm_extra_args+=(--sample-indices "${SAMPLE_INDICES}")
    fi
    "${PYTHON_BIN}" scripts/eval/eval_qwen35_vllm_mtp.py \
        --model "${TARGET_MODEL}" \
        --dataset-root "${COMBINED_DATASET_ROOT}" \
        --output-dir "${OUTPUT_DIR}" \
        --tasks "${TASKS}" \
        --seed "${SEED}" \
        --dataset-seed "${DATASET_SEED}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --target-sampling-top-p "${TARGET_SAMPLING_TOP_P}" \
        --acceptance-rule "${ACCEPTANCE_RULE}" \
        --acceptance-top-p "${ACCEPTANCE_TOP_P}" \
        --mtp-proposal-tokens "${MTP_PROPOSAL_TOKENS}" \
        --vllm-max-model-len "${QWEN35_VLLM_MAX_MODEL_LEN}" \
        --vllm-gpu-memory-utilization \
        "${QWEN35_VLLM_GPU_MEMORY_UTILIZATION}" \
        --vllm-enforce-eager "${QWEN35_VLLM_ENFORCE_EAGER}" \
        "${vllm_extra_args[@]}"
elif [[ "${INFERENCE_BACKEND}" == "transformers" ]]; then
    "${PYTHON_BIN}" eval.py \
    --target_name_or_path "${TARGET_MODEL}" \
    --draft_name_or_path "${DRAFT_MODEL}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --tensorboard-dir "${OUTPUT_DIR}" \
    --step "${SEED}" \
    --seed "${SEED}" \
    --dataset-seed "${DATASET_SEED}" \
    --dataset-root "${COMBINED_DATASET_ROOT}" \
    --tasks "${TASKS}" \
    --acceptance-rule "${ACCEPTANCE_RULE}" \
    --acceptance-top-k "${ACCEPTANCE_TOP_K}" \
    --acceptance-top-p "${ACCEPTANCE_TOP_P}" \
    --dynamic-top-p-blend "${DYNAMIC_TOP_P_BLEND}" \
    --dynamic-top-p-support "${DYNAMIC_TOP_P_SUPPORT}" \
    --dynamic-top-p-outer-alpha "${DYNAMIC_TOP_P_OUTER_ALPHA}" \
    --dynamic-top-p-signal "${DYNAMIC_TOP_P_SIGNAL}" \
    --risk-budget-max "${RISK_BUDGET_MAX}" \
    --risk-budget-safe-top-p "${RISK_BUDGET_SAFE_TOP_P}" \
    --cactus-delta "${CACTUS_DELTA}" \
    --mars-relaxation-threshold "${MARS_RELAXATION_THRESHOLD}" \
    --target-sampling-top-p "${TARGET_SAMPLING_TOP_P}" \
    --eagle3-proposal-tokens "${EAGLE3_PROPOSAL_TOKENS}" \
    --mtp-proposal-tokens "${MTP_PROPOSAL_TOKENS}" \
    --nemotron-linear-spec-block-length \
    "${NEMOTRON_LINEAR_SPEC_BLOCK_LENGTH}" \
    --nemotron-linear-spec-threshold \
    "${NEMOTRON_LINEAR_SPEC_THRESHOLD}" \
    --nemotron-linear-spec-use-lora \
    "${NEMOTRON_LINEAR_SPEC_USE_LORA}" \
    --nemotron-linear-spec-lora-subfolder \
    "${NEMOTRON_LINEAR_SPEC_LORA_SUBFOLDER}" \
    --blind-accept-every-nth-draft-block \
    "${BLIND_ACCEPT_EVERY_NTH_DRAFT_BLOCK}" \
    --dump-predictions \
    --record-throughput \
    "${eval_extra_args[@]}"
else
    echo "Unsupported INFERENCE_BACKEND=${INFERENCE_BACKEND}; expected transformers or vllm" >&2
    exit 1
fi

echo "Scoring ${TASKS}"
"${PYTHON_BIN}" -m scripts.eval.score_predictions \
    "${OUTPUT_DIR}" \
    --tasks "${TASKS}" \
    --allow-code-execution \
    --code-workers "${CODE_WORKERS}" \
    --humaneval-timeout "${HUMANEVAL_TIMEOUT}" \
    --mbpp-timeout "${MBPP_TIMEOUT}" \
    --bwrap /usr/bin/bwrap \
    --python-bin "${PYTHON_BIN}"

echo "Benchmark evaluation complete"
echo "Accuracy: ${OUTPUT_DIR}/quality_metrics.json"
echo "Throughput and acceptance: ${OUTPUT_DIR}/throughput_metrics.json"
echo "Log: ${LOG_PATH}"
