from __future__ import annotations
import argparse
import json
import math
import torch
from transformers import AutoConfig
from deepspec.eval.nemotron_diffusion import NemotronLabsDiffusionEvaluator
from deepspec.eval.autoregressive import AutoregressiveEvaluator
from deepspec.eval.token_diagnostics import TOKEN_DIAGNOSTICS_MODES
from deepspec.utils import CustomJSONEncoder

EVALUATORS = {
    "NemotronLabsDiffusionModel": NemotronLabsDiffusionEvaluator,
}

TASKS = [
    ("gsm8k", 1319),
    ("math500", 500),
    ("aime24", 30),
    ("aime25", 30),
    ("aime26", 30),
    ("humaneval", 164),
    ("mbpp", 257),
]


def _parse_task_list(task_names: str | None) -> list[tuple[str, int]]:
    task_rows = list(TASKS)
    if task_names is None:
        return task_rows
    requested = [name.strip() for name in task_names.split(",") if name.strip()]
    known = {name: max_samples for name, max_samples in TASKS}
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError(
            f"Unknown task name(s): {unknown}. Expected one of {sorted(known)}."
        )
    return [(name, known[name]) for name in requested]


def _parse_nonnegative_int_list(values: str | None) -> list[int] | None:
    if values is None:
        return None
    parsed = [int(value.strip()) for value in values.split(",") if value.strip()]
    if not parsed:
        raise ValueError("Expected at least one sample index.")
    if any(value < 0 for value in parsed):
        raise ValueError("Sample indices must be non-negative.")
    if len(parsed) != len(set(parsed)):
        raise ValueError("Sample indices must not contain duplicates.")
    return parsed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path",type=str,required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--mtp-proposal-tokens",
        type=int,
        default=5,
        help=(
            "Qwen3.5-MTP only: number of speculative draft tokens proposed "
            "before one target verification."
        ),
    )
    parser.add_argument(
        "--eagle3-proposal-tokens",
        type=int,
        default=7,
        help=(
            "Official EAGLE3 linear adapter only: number of autoregressive "
            "draft tokens proposed before target verification."
        ),
    )
    parser.add_argument(
        "--draft-proposal-blocks",
        type=int,
        default=1,
        help=(
            "DSpark-only: number of draft block-widths to propose before one "
            "target verification. With a block-7 draft, 2 proposes 14 draft "
            "tokens at once. Default 1 preserves existing behavior."
        ),
    )
    parser.add_argument(
        "--blind-accept-every-nth-draft-block",
        type=int,
        default=0,
        help=(
            "DSpark-only approximate mode. With N>=2, every Nth logical "
            "draft block is blindly accepted and fused with the following "
            "normally verified block in one target forward pass. Zero "
            "disables the mode. Requires rejection_sampling, "
            "draft_proposal_blocks=1, and confidence_threshold=0."
        ),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--autoregressive-top-k",
        type=int,
        default=None,
        help=(
            "AR baseline only: override the model generation-config Top-K. "
            "Use 0 for unrestricted sampling; omission preserves existing defaults. "
            "Nemotron's custom AR loop supports only 0."
        ),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument(
        "--dump-predictions",
        action="store_true",
        help=(
            "Persist rank-local generated answers and source/reference metadata "
            "as JSONL while each dataset is evaluated."
        ),
    )
    parser.add_argument(
        "--record-throughput",
        action="store_true",
        help=(
            "Measure per-dataset inference wall time and persist generated-token "
            "and sample throughput to <tensorboard-dir>/throughput_metrics.json. "
            "Model loading and post-hoc scoring are excluded."
        ),
    )
    parser.add_argument(
        "--record-verification-timing",
        action="store_true",
        help=(
            "Record accelerator-synchronized wall time for target verification "
            "forwards and acceptance-rule work. This adds synchronization overhead "
            "and is intended only for small phase-timing experiments."
        ),
    )
    parser.add_argument(
        "--verification-timing-warmup-samples",
        type=int,
        default=1,
        help="Exclude this many leading samples per dataset from phase timing.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help=(
            "Optional prediction artifact root. Defaults to "
            "<tensorboard-dir>/artifacts when prediction dumping is enabled."
        ),
    )
    parser.add_argument(
        "--acceptance-rule",
        choices=(
            "rejection_sampling",
            "rejection_sampling_then_target_top_p",
            "linear_self_speculation",
            "autoregressive",
        ),
        default="rejection_sampling",
        help=(
            "Nemotron reproduction modes: target-only autoregressive decoding, "
            "official Linear Self-Speculation, standard rejection sampling, or "
            "rejection sampling followed by target Top-P rescue (NucleusSD)."
        ),
    )
    parser.add_argument(
        "--nemotron-linear-spec-block-length",
        type=int,
        default=32,
        help=(
            "Nemotron-Labs-Diffusion only: total Linear Self-Speculation "
            "block width, including the current seed token."
        ),
    )
    parser.add_argument(
        "--nemotron-linear-spec-threshold",
        type=float,
        default=0.0,
        help=(
            "Nemotron-Labs-Diffusion only: iterative diffusion-draft "
            "confidence threshold. Zero performs one draft forward per block."
        ),
    )
    parser.add_argument(
        "--nemotron-linear-spec-use-lora",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Nemotron-Labs-Diffusion only: load the model's Linear "
            "Self-Speculation LoRA adapter for diffusion drafting."
        ),
    )
    parser.add_argument(
        "--nemotron-linear-spec-lora-subfolder",
        type=str,
        default="linear_spec_lora",
        help=(
            "Nemotron-Labs-Diffusion only: adapter subfolder passed to "
            "PeftModel.from_pretrained."
        ),
    )
    parser.add_argument(
        "--acceptance-top-k",
        type=int,
        default=2,
        help=(
            "Top-K threshold used by target_top_k_top_p and "
            "rejection_sampling_then_target_top_k_top_p."
        ),
    )
    parser.add_argument("--asd-budget", type=float, default=8.0,
                        help="ASD persistent request-level regret budget (nats).")
    parser.add_argument("--asd-gate", type=float, default=0.25,
                        help="ASD maximum regret divided by suffix value.")
    parser.add_argument("--asd-max-exceptions", type=int, default=2,
                        help="ASD maximum accepted mismatches in one proposal block.")
    parser.add_argument("--asd-value-scheme", default="suffix_length",
                        choices=("suffix_length", "normalized_suffix"),
                        help="Paper Algorithm 1 or supplied DSpark reproduction variant.")
    parser.add_argument(
        "--cactus-delta",
        type=float,
        default=1.0,
        help=(
            "Finite, nonnegative Cactus KL-divergence budget. Delta 0 "
            "recovers standard rejection sampling."
        ),
    )
    parser.add_argument(
        "--mars-relaxation-threshold",
        type=float,
        default=0.90,
        help=(
            "MARS only: switch to target Top-2 verification when the ratio "
            "of the second-largest target logit to the largest target logit "
            "exceeds this threshold. The reference default is 0.90."
        ),
    )
    parser.add_argument(
        "--acceptance-top-p",
        type=float,
        default=0.95,
        help=(
            "Top-p threshold used by the fixed target Top-p acceptance rules. "
            "It is also the maximum rescue nucleus for Risk-Budgeted Top-P. "
            "Dynamic Top-P Rescue uses its separately configured support gate "
            "and computed rescue threshold."
        ),
    )
    parser.add_argument(
        "--dynamic-top-p-blend",
        type=float,
        default=0.0,
        help=(
            "Dynamic Top-P Rescue only: interpolate its computed threshold "
            "toward 0.95. Zero preserves the original formula; one is "
            "equivalent to the fixed 0.95 rescue threshold."
        ),
    )
    parser.add_argument(
        "--risk-budget-max",
        type=float,
        default=1.0,
        help=(
            "Risk-Budgeted Top-P only: maximum cumulative rescue risk per "
            "draft block. Standard rejection-sampling accepts consume zero."
        ),
    )
    parser.add_argument(
        "--risk-budget-safe-top-p",
        type=float,
        default=0.90,
        help=(
            "Risk-Budgeted Top-P only: prefix-mass threshold below which a "
            "rescued token consumes zero risk. Must be smaller than "
            "--acceptance-top-p."
        ),
    )
    parser.add_argument(
        "--dynamic-top-p-support",
        type=float,
        default=0.95,
        help=(
            "Dynamic Top-P Rescue only: target nucleus support ceiling. "
            "The original method uses 0.95; values up to 1.0 expose an outer "
            "candidate shell."
        ),
    )
    parser.add_argument(
        "--dynamic-top-p-outer-alpha",
        type=float,
        default=0.0,
        help=(
            "Dynamic Top-P Rescue only: expand the blended inner threshold "
            "toward --dynamic-top-p-support by alpha * entropy * (1 - TV). "
            "Zero disables the outer shell."
        ),
    )
    parser.add_argument(
        "--dynamic-top-p-signal",
        choices=("entropy_disagreement", "entropy_only"),
        default="entropy_disagreement",
        help=(
            "Dynamic Top-P Rescue only: entropy_disagreement preserves the "
            "TV penalty in the inner and outer thresholds; entropy_only logs "
            "TV but does not use it in either acceptance threshold."
        ),
    )
    parser.add_argument(
        "--target-sampling-top-p",
        type=float,
        default=1.0,
        help=(
            "Nucleus sampling threshold for target-model distributions used "
            "for target sampling, acceptance ratios and residual sampling. "
            "Default 1.0 preserves full-distribution sampling."
        ),
    )
    parser.add_argument(
        "--record-rejected-token-diagnostics",
        action="store_true",
        help=(
            "For DSpark confidence-head models, write JSONL diagnostics for "
            "the first rejected draft token in each failed proposal."
        ),
    )
    parser.add_argument(
        "--record-token-diagnostics",
        action="store_true",
        help=(
            "Write rank-local JSONL diagnostics for proposed draft tokens, "
            "including target/draft probability and entropy."
        ),
    )
    parser.add_argument(
        "--token-diagnostics-mode",
        choices=tuple(sorted(TOKEN_DIAGNOSTICS_MODES)),
        default="all",
        help=(
            "Which proposed draft tokens to log when "
            "--record-token-diagnostics is enabled."
        ),
    )
    parser.add_argument(
        "--record-exact-nsd-rho",
        action="store_true",
        help=(
            "When token diagnostics are enabled for "
            "rejection_sampling_then_target_top_p, compute the exact "
            "full-vocabulary one-step NSD/target-nucleus TV and rho."
        ),
    )
    parser.add_argument(
        "--token-diagnostics-tasks",
        type=str,
        default=None,
        help=(
            "Optional comma-separated task filter for token diagnostics. "
            "Generation and prediction dumping still run for every selected task."
        ),
    )
    parser.add_argument(
        "--replay-samples-jsonl",
        type=str,
        default=None,
        help=(
            "Optional samples.rankXXXXX.jsonl file containing generated_token_ids "
            "used to warm-start generation from a saved output prefix."
        ),
    )
    parser.add_argument(
        "--replay-prefix-tokens",
        type=int,
        default=None,
        help=(
            "Number of generated tokens to load from the replay sample. "
            "Defaults to the full saved output when replay is enabled."
        ),
    )
    parser.add_argument(
        "--replay-sample-index",
        type=int,
        default=None,
        help="Restrict prefix replay to one sample_index from the samples JSONL.",
    )
    parser.add_argument(
        "--force-reject-sample-index",
        type=int,
        default=None,
        help=(
            "Restrict a forced draft-token rejection intervention to one "
            "sample_index."
        ),
    )
    parser.add_argument(
        "--force-reject-proposal-index",
        type=int,
        default=None,
        help="Proposal index whose draft token should be forced to reject.",
    )
    parser.add_argument(
        "--force-reject-proposal-position",
        type=int,
        default=None,
        help=(
            "0-based draft-token position inside force-reject-proposal-index "
            "to force reject."
        ),
    )
    parser.add_argument(
        "--force-reject-generated-position",
        type=int,
        default=None,
        help=(
            "Absolute generated_position from token diagnostics whose draft "
            "token should be forced to reject."
        ),
    )
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        help=(
            "Seed used only for max-sample dataset subsetting. Defaults to "
            "--seed for backward compatibility."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help=(
            "Comma-separated task names to run, e.g. "
            "math500,aime25,mbpp,humaneval,opencompass_mbpp,"
            "opencompass_humaneval,mt-bench,alpaca,arena-hard-v2."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./eval_datasets",
        help="Directory containing <task>.jsonl benchmark files.",
    )
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=None,
        help="Cap each selected task to at most this many samples.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated dataset indices to evaluate after deterministic "
            "dataset shuffling/truncation. The original per-sample RNG seed remains "
            "seed + sample_index, enabling targeted counterfactual reruns."
        ),
    )
    args = parser.parse_args()
    if args.record_rejected_token_diagnostics and args.tensorboard_dir is None:
        parser.error(
            "--tensorboard-dir is required when "
            "--record-rejected-token-diagnostics is set."
        )
    if args.record_token_diagnostics and args.tensorboard_dir is None:
        parser.error(
            "--tensorboard-dir is required when "
            "--record-token-diagnostics is set."
        )
    if args.record_exact_nsd_rho and not args.record_token_diagnostics:
        parser.error(
            "--record-exact-nsd-rho requires --record-token-diagnostics."
        )
    if (
        args.dump_predictions
        and args.predictions_dir is None
        and args.tensorboard_dir is None
    ):
        parser.error(
            "--dump-predictions requires --predictions-dir or --tensorboard-dir."
        )
    if args.record_throughput and args.tensorboard_dir is None:
        parser.error("--record-throughput requires --tensorboard-dir.")
    if args.record_verification_timing:
        if not args.dump_predictions:
            parser.error("--record-verification-timing requires --dump-predictions")
        if args.tensorboard_dir is None and args.predictions_dir is None:
            parser.error("--record-verification-timing requires a prediction output directory")
        if args.acceptance_rule in {"nemotron_diffusion", "linear_self_speculation"}:
            parser.error(
                "--record-verification-timing requires AR target-forward timing "
                "or a supported speculative verifier"
            )
    if int(args.verification_timing_warmup_samples) < 0:
        parser.error("--verification-timing-warmup-samples must be >= 0")
    if not 0.0 <= float(args.acceptance_top_p) <= 1.0:
        parser.error("--acceptance-top-p must be in [0, 1].")
    if (
        not math.isfinite(float(args.risk_budget_max))
        or float(args.risk_budget_max) < 0.0
    ):
        parser.error("--risk-budget-max must be finite and >= 0.")
    if (
        not math.isfinite(float(args.risk_budget_safe_top_p))
        or not 0.0 <= float(args.risk_budget_safe_top_p) < 1.0
    ):
        parser.error("--risk-budget-safe-top-p must be finite and in [0, 1).")
    if args.acceptance_rule in {
        "rejection_sampling_then_risk_budget_top_p",
        "rejection_sampling_then_entropy_risk_budget_top_p",
    }:
        if float(args.acceptance_top_p) <= 0.0:
            parser.error(
                "Risk-Budgeted Top-P requires --acceptance-top-p in (0, 1]."
            )
        if float(args.risk_budget_safe_top_p) >= float(args.acceptance_top_p):
            parser.error(
                "--risk-budget-safe-top-p must be smaller than "
                "--acceptance-top-p for Risk-Budgeted Top-P."
            )
    if (
        not math.isfinite(float(args.dynamic_top_p_blend))
        or not 0.0 <= float(args.dynamic_top_p_blend) <= 1.0
    ):
        parser.error("--dynamic-top-p-blend must be finite and in [0, 1].")
    if (
        not math.isfinite(float(args.dynamic_top_p_support))
        or not 0.95 <= float(args.dynamic_top_p_support) <= 1.0
    ):
        parser.error("--dynamic-top-p-support must be finite and in [0.95, 1].")
    if (
        not math.isfinite(float(args.dynamic_top_p_outer_alpha))
        or not 0.0 <= float(args.dynamic_top_p_outer_alpha) <= 1.0
    ):
        parser.error(
            "--dynamic-top-p-outer-alpha must be finite and in [0, 1]."
        )
    if not 0.0 <= float(args.target_sampling_top_p) <= 1.0:
        parser.error("--target-sampling-top-p must be in [0, 1].")
    if int(args.acceptance_top_k) < 1:
        parser.error("--acceptance-top-k must be >= 1.")
    from deepspec.eval.approximate_sd import validate_asd
    try:
        validate_asd(args.asd_budget, args.asd_gate, args.asd_max_exceptions, args.asd_value_scheme)
    except ValueError as exc:
        parser.error(str(exc))
    if args.acceptance_rule == "asd":
        if args.blind_accept_every_nth_draft_block or any(value is not None for value in
                (args.force_reject_proposal_index, args.force_reject_generated_position)):
            parser.error("ASD cannot be combined with blind acceptance or forced rejection")
    if not math.isfinite(float(args.cactus_delta)) or float(args.cactus_delta) < 0.0:
        parser.error("--cactus-delta must be finite and >= 0.")
    if (
        not math.isfinite(float(args.mars_relaxation_threshold))
        or not 0.0 <= float(args.mars_relaxation_threshold) <= 1.0
    ):
        parser.error(
            "--mars-relaxation-threshold must be finite and in [0, 1]."
        )
    if int(args.draft_proposal_blocks) < 1:
        parser.error("--draft-proposal-blocks must be >= 1.")
    if int(args.eagle3_proposal_tokens) < 1:
        parser.error("--eagle3-proposal-tokens must be >= 1.")
    if int(args.blind_accept_every_nth_draft_block) == 1:
        parser.error("--blind-accept-every-nth-draft-block must be 0 or >= 2.")
    if int(args.blind_accept_every_nth_draft_block) < 0:
        parser.error("--blind-accept-every-nth-draft-block must be 0 or >= 2.")
    if int(args.mtp_proposal_tokens) < 1:
        parser.error("--mtp-proposal-tokens must be >= 1.")
    if int(args.nemotron_linear_spec_block_length) < 2:
        parser.error("--nemotron-linear-spec-block-length must be >= 2.")
    if (
        not math.isfinite(float(args.nemotron_linear_spec_threshold))
        or not 0.0 <= float(args.nemotron_linear_spec_threshold) <= 1.0
    ):
        parser.error(
            "--nemotron-linear-spec-threshold must be finite and in [0, 1]."
        )
    if (
        int(args.nemotron_linear_spec_use_lora) == 1
        and not str(args.nemotron_linear_spec_lora_subfolder).strip()
    ):
        parser.error(
            "--nemotron-linear-spec-lora-subfolder must not be empty when "
            "LoRA is enabled."
        )
    if (
        args.force_reject_proposal_index is not None
        and args.force_reject_generated_position is not None
    ):
        parser.error(
            "Use either --force-reject-proposal-index/position or "
            "--force-reject-generated-position, not both."
        )
    if (
        args.force_reject_proposal_index is None
        and args.force_reject_proposal_position is not None
    ):
        parser.error(
            "--force-reject-proposal-position requires "
            "--force-reject-proposal-index."
        )
    if (
        args.force_reject_proposal_index is not None
        and args.force_reject_proposal_position is None
    ):
        parser.error(
            "--force-reject-proposal-index requires "
            "--force-reject-proposal-position."
        )
    if args.replay_prefix_tokens is not None and args.replay_prefix_tokens < 0:
        parser.error("--replay-prefix-tokens must be >= 0.")
    if args.max_samples_per_task is not None and args.max_samples_per_task < 1:
        parser.error("--max-samples-per-task must be >= 1.")
    try:
        tasks = _parse_task_list(args.tasks)
        args.sample_indices = _parse_nonnegative_int_list(args.sample_indices)
    except ValueError as exc:
        parser.error(str(exc))
    if args.token_diagnostics_tasks is not None:
        requested_diagnostic_tasks = {
            name.strip()
            for name in args.token_diagnostics_tasks.split(",")
            if name.strip()
        }
        selected_task_names = {name for name, _ in tasks}
        unknown_diagnostic_tasks = sorted(
            requested_diagnostic_tasks - selected_task_names
        )
        if unknown_diagnostic_tasks:
            parser.error(
                "--token-diagnostics-tasks contains tasks not selected by --tasks: "
                f"{unknown_diagnostic_tasks}"
            )
        args.token_diagnostics_tasks = sorted(requested_diagnostic_tasks)
    if args.max_samples_per_task is not None:
        tasks = [
            (task_name, min(int(max_samples), int(args.max_samples_per_task)))
            for task_name, max_samples in tasks
        ]
    args.task_names = args.tasks
    args.tasks = tasks
    if args.dataset_seed is None:
        args.dataset_seed = args.seed
    return args


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    if args.acceptance_rule == "autoregressive":
        evaluator_cls = AutoregressiveEvaluator
    else:
        draft_config = AutoConfig.from_pretrained(
            args.draft_name_or_path,
            trust_remote_code=True,
        )
        evaluator_cls = EVALUATORS[draft_config.architectures[0]]
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    nprocs = torch.cuda.device_count()
    if nprocs > 0:
        torch.multiprocessing.spawn(
            main,
            args=(args,),
            nprocs=nprocs,
        )
    else:
        main(0, args)
