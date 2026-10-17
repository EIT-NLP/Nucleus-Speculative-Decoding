from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, DynamicCache

from deepspec.data.parser import encode_chat_messages, render_chat_messages
from deepspec.eval.approximate_sd import ASDState, compute_asd_prefix
from deepspec.eval.cactus import (
    cactus_adjusted_distribution,
    cactus_adjusted_selected_probs as compute_cactus_adjusted_selected_probs,
    validate_cactus_delta,
)
from deepspec.eval.dynamic_top_p import (
    DYNAMIC_TOP_P_SIGNAL_ENTROPY_DISAGREEMENT,
    DYNAMIC_TOP_P_SIGNAL_ENTROPY_ONLY,
    compute_dynamic_top_p,
    validate_dynamic_top_p_blend,
    validate_dynamic_top_p_outer_alpha,
    validate_dynamic_top_p_signal,
    validate_dynamic_top_p_support,
)
from deepspec.eval.mars import (
    DEFAULT_MARS_RELAXATION_THRESHOLD,
    compute_mars_acceptance,
    validate_mars_relaxation_threshold,
)
from deepspec.eval.clean_nsd import target_top_p_accept_mask as clean_top_p_accept_mask
from deepspec.eval.risk_budget_top_p import (
    DEFAULT_RISK_BUDGET_MAX,
    DEFAULT_RISK_BUDGET_SAFE_TOP_P,
    apply_block_risk_budget,
    compute_risk_budget_top_p,
    compute_target_prefix_mass,
    validate_risk_budget_max,
    validate_risk_budget_top_p_thresholds,
)
from deepspec.eval.token_diagnostics import TokenDiagnosticsRecorder
from deepspec.eval.verification_timing import VerificationPhaseTimer
from deepspec.utils.sampling import (
    gather_token_probs,
    logits_to_probs,
    sample_from_probs,
    sample_residual,
)
from deepspec.utils import (
    init_dist,
    seed_all,
)


DEFAULT_DATASET_ROOT = "./eval_datasets"
ACCEPTANCE_REJECTION_SAMPLING = "rejection_sampling"
ACCEPTANCE_CACTUS = "cactus"
ACCEPTANCE_MARS = "mars"
ACCEPTANCE_ASD = "asd"
ACCEPTANCE_CLEAN_NSD = "cleanNSD"
ACCEPTANCE_TARGET_TOP_P = "target_top_p"
ACCEPTANCE_ONLY_TOP_P = "onlytopp"
ACCEPTANCE_REJECTION_SAMPLING_THEN_TARGET_TOP_P = (
    "rejection_sampling_then_target_top_p"
)
ACCEPTANCE_REJECTION_SAMPLING_THEN_DYNAMIC_TOP_P = (
    "rejection_sampling_then_dynamic_top_p"
)
ACCEPTANCE_REJECTION_SAMPLING_THEN_RISK_BUDGET_TOP_P = (
    "rejection_sampling_then_risk_budget_top_p"
)
ACCEPTANCE_REJECTION_SAMPLING_THEN_ENTROPY_RISK_BUDGET_TOP_P = (
    "rejection_sampling_then_entropy_risk_budget_top_p"
)
ACCEPTANCE_REJECTION_SAMPLING_THEN_TARGET_TOP_K_TOP_P = (
    "rejection_sampling_then_target_top_k_top_p"
)
ACCEPTANCE_TARGET_TOP_K_TOP_P = "target_top_k_top_p"
ACCEPTANCE_RULES = {
    ACCEPTANCE_ASD,
    ACCEPTANCE_CLEAN_NSD,
    ACCEPTANCE_REJECTION_SAMPLING,
    ACCEPTANCE_CACTUS,
    ACCEPTANCE_MARS,
    ACCEPTANCE_TARGET_TOP_P,
    ACCEPTANCE_ONLY_TOP_P,
    ACCEPTANCE_REJECTION_SAMPLING_THEN_TARGET_TOP_P,
    ACCEPTANCE_REJECTION_SAMPLING_THEN_DYNAMIC_TOP_P,
    ACCEPTANCE_REJECTION_SAMPLING_THEN_RISK_BUDGET_TOP_P,
    ACCEPTANCE_REJECTION_SAMPLING_THEN_ENTROPY_RISK_BUDGET_TOP_P,
    ACCEPTANCE_REJECTION_SAMPLING_THEN_TARGET_TOP_K_TOP_P,
    ACCEPTANCE_TARGET_TOP_K_TOP_P,
}


def load_and_process_dataset(
    data_name: str,
    dataset_root: str = DEFAULT_DATASET_ROOT,
):
    dataset_path = Path(dataset_root) / f"{data_name}.jsonl"
    assert dataset_path.exists()

    rows = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            turns = row.get("turns")
            assert (
                isinstance(turns, list)
                and len(turns) > 0
                and all(isinstance(turn, str) for turn in turns)
            ), (
                f"{dataset_path}:{line_number} must contain a non-empty string list field `turns`."
            )
            row["all_turns"] = list(turns)
            row["turns"] = turns[:1]
            rows.append(row)
    return rows


def encode_evaluation_instance(
    tokenizer,
    instance: dict[str, Any],
) -> torch.LongTensor:
    """Encode one evaluation prompt while preserving the legacy path.

    Existing datasets contain only ``turns`` and continue through the exact
    single-user-message encoder used before OpenCompass-style code evaluation
    was added.  Separate OpenCompass datasets may provide a role-preserving
    ``messages`` list and an optional prefix that begins the assistant answer.
    """
    messages = instance.get("messages")
    assistant_prefix = instance.get("assistant_prefix", "")
    if messages is None and not assistant_prefix:
        return encode_chat_messages(
            tokenizer,
            [{"role": "user", "content": instance["turns"][0]}],
            add_generation_prompt=True,
            enable_thinking=False,
        )

    if messages is None:
        messages = [{"role": "user", "content": instance["turns"][0]}]
    if not isinstance(messages, list) or not messages:
        raise TypeError("Optional evaluation field `messages` must be a non-empty list.")
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(
                f"Evaluation message {message_index} must be a dictionary."
            )
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(
                f"Evaluation message {message_index} has unsupported role "
                f"{message.get('role')!r}."
            )
        if not isinstance(message.get("content"), str):
            raise TypeError(
                f"Evaluation message {message_index} must contain string content."
            )
    if not isinstance(assistant_prefix, str):
        raise TypeError("Optional evaluation field `assistant_prefix` must be a string.")

    conversation_text = render_chat_messages(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    conversation_text += assistant_prefix
    return tokenizer(
        conversation_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids


def trim_output_ids(
    output_ids: torch.Tensor,
    num_input_tokens: int,
    stop_token_ids: list[int] | None,
) -> torch.Tensor:
    if stop_token_ids is None:
        return output_ids
    stop_token_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
    stop_token_indices = torch.isin(
        output_ids[0][num_input_tokens:],
        stop_token_tensor,
    ).nonzero(as_tuple=True)[0]
    if stop_token_indices.numel() == 0:
        return output_ids
    return output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]


def has_stop_token(token_ids: torch.Tensor, stop_token_ids: list[int] | None) -> bool:
    if stop_token_ids is None:
        return False
    stop_token_tensor = torch.tensor(stop_token_ids, device=token_ids.device)
    return bool(torch.isin(token_ids, stop_token_tensor).any().item())


def resolve_stop_token_ids(target_model, tokenizer) -> list[int] | None:
    generation_config = getattr(target_model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]

    stop_token_ids = []
    for token_id in eos_token_id:
        token_id = int(token_id)
        if token_id not in stop_token_ids:
            stop_token_ids.append(token_id)
    return stop_token_ids


def get_text_config(config):
    return getattr(config, "text_config", config)


def assert_target_draft_compatible(target_model, draft_model) -> None:
    target_config = target_model.config
    target_config = get_text_config(target_config)
    draft_config = get_text_config(draft_model.config)

    last_layer_id = int(target_config.num_hidden_layers) - 1
    target_layer_ids = [int(layer_id) for layer_id in draft_model.target_layer_ids]
    out_of_range_layer_ids = [
        layer_id
        for layer_id in target_layer_ids
        if layer_id != -1 and not 0 <= layer_id <= last_layer_id
    ]
    if out_of_range_layer_ids:
        raise ValueError(
            "Draft target_layer_ids are incompatible with the target model: "
            f"{out_of_range_layer_ids} are outside the valid range "
            f"{{-1}} U [0, {last_layer_id}] for "
            f"num_hidden_layers={target_config.num_hidden_layers}. "
            "Use a draft checkpoint trained against this target architecture."
        )
    assert last_layer_id not in target_layer_ids, (
        "target_layer_ids must not include the final target decoder layer "
        f"{last_layer_id}. Transformers output_hidden_states stores the final "
        "normalized hidden state at that position, while target cache generation "
        "stores raw decoder-layer outputs. Use an earlier layer and regenerate "
        "the target cache and draft checkpoint."
    )
    if int(getattr(target_config, "hidden_size")) != int(
        getattr(draft_config, "hidden_size")
    ):
        raise ValueError(
            "Target and draft hidden sizes differ: "
            f"target hidden_size={target_config.hidden_size}, "
            f"draft hidden_size={draft_config.hidden_size}. "
            "Use a draft checkpoint trained against this target architecture."
        )
    if int(getattr(target_config, "vocab_size")) != int(
        getattr(draft_config, "vocab_size")
    ):
        raise ValueError(
            "Target and draft vocab sizes differ: "
            f"target vocab_size={target_config.vocab_size}, "
            f"draft vocab_size={draft_config.vocab_size}. "
            "Speculative decoding requires the draft and target to share token ids."
        )


def build_dynamic_cache(config=None) -> DynamicCache:
    if config is not None:
        config = get_text_config(config)
        try:
            return DynamicCache(config=config)
        except TypeError:
            pass
    return DynamicCache()


def build_results_table(
    *,
    rows: list[dict[str, object]],
    model_name_or_path: str,
    draft_name_or_path: str,
    header: bool = True,
) -> str:
    from prettytable import PrettyTable

    table = PrettyTable()
    max_positions = max(
        (len(metrics["accept_rates_by_position"]) for metrics in rows),
        default=0,
    )
    field_names = [
        "dataset",
        "target_model",
        "draft_model",
        "#propose",
        "accept_len",
        "verify_rate",
    ]
    show_throughput = any(
        metrics.get("throughput_tokens_per_second") is not None for metrics in rows
    )
    if show_throughput:
        field_names.extend(["output_tokens", "elapsed_s", "output_tok/s", "samples/s"])
    show_blind_accept = any(
        int(metrics.get("blind_accept_event_count", 0)) > 0 for metrics in rows
    )
    if show_blind_accept:
        field_names.extend(["logical_blocks", "blind_blocks", "blind_tokens"])
    field_names.extend(f"accept_rate@{pos_idx}" for pos_idx in range(max_positions))
    table.field_names = field_names
    table.header = header

    normalized_target_model_name = model_name_or_path.rstrip("/")
    target_model_name = (
        os.path.basename(normalized_target_model_name) or normalized_target_model_name
    )
    normalized_draft_model_name = draft_name_or_path.rstrip("/")
    draft_model_name = (
        os.path.basename(normalized_draft_model_name) or normalized_draft_model_name
    )
    for metrics in rows:
        row = [
            metrics["dataset"],
            target_model_name,
            draft_model_name,
            f"{metrics['draft_tokens_per_proposal']:.2f}+1",
            f"{metrics['acceptance_length']:.2f}",
            f"{metrics['verify_rate']:.4f}",
        ]
        if show_throughput:
            throughput = metrics.get("throughput_tokens_per_second")
            sample_throughput = metrics.get("throughput_samples_per_second")
            elapsed_seconds = metrics.get("elapsed_seconds")
            if (
                throughput is None
                or sample_throughput is None
                or elapsed_seconds is None
            ):
                row.extend(["-", "-", "-", "-"])
            else:
                row.extend(
                    [
                        str(int(metrics["output_token_count"])),
                        f"{float(elapsed_seconds):.3f}",
                        f"{float(throughput):.3f}",
                        f"{float(sample_throughput):.5f}",
                    ]
                )
        if show_blind_accept:
            row.extend(
                [
                    str(int(metrics.get("logical_draft_block_count", 0))),
                    str(int(metrics.get("blind_accept_event_count", 0))),
                    str(int(metrics.get("blind_accepted_token_count", 0))),
                ]
            )
        row.extend(
            f"{accept_rate:.4f}" if accept_rate is not None else "-"
            for accept_rate in metrics["accept_rates_by_position"]
        )
        row.extend(["-"] * (max_positions - len(metrics["accept_rates_by_position"])))
        table.add_row(row)
    return table.get_string()


@dataclass
class DraftProposal:
    draft_token_count: int
    verify_input_ids: torch.Tensor
    draft_probs: torch.Tensor | None
    blind_accept_prefix_length: int = 0
    logical_draft_block_count: int = 1
    draft_block_size: int = 0


@dataclass
class VerificationResult:
    target_output: Any
    target_probs: torch.Tensor
    accept_prefix_mask: torch.Tensor | None
    accepted_draft_tokens: int
    next_token: torch.Tensor
    effective_proposal_length: int
    terminated_by_stop_token: bool = False
    committed_tokens: torch.Tensor | None = None
    accept_mask: torch.Tensor | None = None
    standard_accept_mask: torch.Tensor | None = None
    standard_uniform_random: torch.Tensor | None = None
    target_top_p_accept_mask: torch.Tensor | None = None
    target_top_k_top_p_accept_mask: torch.Tensor | None = None
    dynamic_top_p_initial_accept_mask: torch.Tensor | None = None
    dynamic_top_p_initial_nucleus_size: torch.Tensor | None = None
    dynamic_top_p_entropy: torch.Tensor | None = None
    dynamic_top_p_tv_distance: torch.Tensor | None = None
    dynamic_top_p_threshold: torch.Tensor | None = None
    dynamic_top_p_blend: float = 0.0
    dynamic_top_p_support: float = 0.95
    dynamic_top_p_outer_alpha: float = 0.0
    dynamic_top_p_signal: str = DYNAMIC_TOP_P_SIGNAL_ENTROPY_DISAGREEMENT
    dynamic_top_p_accept_mask: torch.Tensor | None = None
    dynamic_top_p_rescue_mask: torch.Tensor | None = None
    dynamic_top_p_nucleus_size: torch.Tensor | None = None
    risk_budget_target_prefix_mass: torch.Tensor | None = None
    risk_budget_inside_target_top_p: torch.Tensor | None = None
    risk_budget_rescue_eligibility_mask: torch.Tensor | None = None
    risk_budget_token_risk: torch.Tensor | None = None
    risk_budget_block_risk_before: torch.Tensor | None = None
    risk_budget_candidate_block_risk: torch.Tensor | None = None
    risk_budget_block_risk_after: torch.Tensor | None = None
    risk_budget_rescue_mask: torch.Tensor | None = None
    risk_budget_reject_outside_top_p_mask: torch.Tensor | None = None
    risk_budget_reject_budget_mask: torch.Tensor | None = None
    risk_budget_max: float = DEFAULT_RISK_BUDGET_MAX
    risk_budget_safe_top_p: float = DEFAULT_RISK_BUDGET_SAFE_TOP_P
    selected_target_probs: torch.Tensor | None = None
    selected_draft_probs: torch.Tensor | None = None
    accept_prob: torch.Tensor | None = None
    cactus_adjusted_selected_probs: torch.Tensor | None = None
    cactus_accept_prob: torch.Tensor | None = None
    cactus_accept_mask: torch.Tensor | None = None
    cactus_delta: float = 1.0
    mars_top2_logit_ratio: torch.Tensor | None = None
    mars_relaxation_mask: torch.Tensor | None = None
    mars_inside_target_top2: torch.Tensor | None = None
    mars_relaxed_accept_mask: torch.Tensor | None = None
    mars_relaxed_reject_mask: torch.Tensor | None = None
    mars_relaxation_threshold: float = DEFAULT_MARS_RELAXATION_THRESHOLD
    forced_reject_position: int | None = None
    blind_accept_prefix_length: int = 0
    logical_draft_block_count: int = 1
    asd_decision: Any = None
    asd_statistics: dict | None = None


def _target_top_p_accept_mask(
    *,
    target_probs: torch.Tensor,
    target_logits: torch.Tensor,
    proposed_tokens: torch.Tensor,
    acceptance_top_p: float,
) -> torch.Tensor:
    token_logits = target_logits.gather(
        dim=-1,
        index=proposed_tokens.unsqueeze(-1),
    ).squeeze(-1)
    higher_prob_mask = target_logits > token_logits.unsqueeze(-1)
    cumulative_before = target_probs.masked_fill(~higher_prob_mask, 0.0).sum(dim=-1)
    return cumulative_before <= float(acceptance_top_p)


def _target_top_k_top_p_accept_mask(
    *,
    target_probs: torch.Tensor,
    target_logits: torch.Tensor,
    proposed_tokens: torch.Tensor,
    acceptance_top_k: int,
    acceptance_top_p: float,
) -> torch.Tensor:
    token_logits = target_logits.gather(
        dim=-1,
        index=proposed_tokens.unsqueeze(-1),
    ).squeeze(-1)
    higher_prob_mask = target_logits > token_logits.unsqueeze(-1)
    higher_prob_count = higher_prob_mask.sum(dim=-1)
    cumulative_before = target_probs.masked_fill(~higher_prob_mask, 0.0).sum(dim=-1)
    return (higher_prob_count < int(acceptance_top_k)) & (
        cumulative_before <= float(acceptance_top_p)
    )


def verify_draft_tokens(
    *,
    target_model,
    proposal: DraftProposal,
    position_ids: torch.Tensor,
    start: int,
    past_key_values_target: DynamicCache,
    temperature: float,
    max_proposal_tokens: int,
    current_token_ids: torch.Tensor | None = None,
    stop_token_ids: list[int] | None = None,
    acceptance_rule: str = ACCEPTANCE_REJECTION_SAMPLING,
    acceptance_top_k: int = 2,
    acceptance_top_p: float = 0.95,
    dynamic_top_p_blend: float = 0.0,
    dynamic_top_p_support: float = 0.95,
    dynamic_top_p_outer_alpha: float = 0.0,
    dynamic_top_p_signal: str = DYNAMIC_TOP_P_SIGNAL_ENTROPY_DISAGREEMENT,
    risk_budget_max: float = DEFAULT_RISK_BUDGET_MAX,
    risk_budget_safe_top_p: float = DEFAULT_RISK_BUDGET_SAFE_TOP_P,
    cactus_delta: float = 1.0,
    mars_relaxation_threshold: float = DEFAULT_MARS_RELAXATION_THRESHOLD,
    target_sampling_top_p: float = 1.0,
    force_reject_position: int | None = None,
    asd_state: ASDState | None = None,
    asd_gate: float = 0.25,
    asd_max_exceptions: int = 2,
    asd_value_scheme: str = "suffix_length",
    asd_max_emitted_tokens: int | None = None,
    verification_phase_timer: VerificationPhaseTimer | None = None,
) -> VerificationResult:
    """Verify draft tokens with the target model."""
    if acceptance_rule not in ACCEPTANCE_RULES:
        raise ValueError(
            f"Unsupported acceptance_rule={acceptance_rule!r}; "
            f"expected one of {sorted(ACCEPTANCE_RULES)}."
        )
    if not 0.0 <= float(acceptance_top_p) <= 1.0:
        raise ValueError(
            "acceptance_top_p must be in [0, 1], "
            f"got {acceptance_top_p}."
        )
    if int(acceptance_top_k) < 1:
        raise ValueError(
            "acceptance_top_k must be >= 1, "
            f"got {acceptance_top_k}."
        )
    dynamic_top_p_blend = validate_dynamic_top_p_blend(dynamic_top_p_blend)
    dynamic_top_p_support = validate_dynamic_top_p_support(dynamic_top_p_support)
    dynamic_top_p_outer_alpha = validate_dynamic_top_p_outer_alpha(
        dynamic_top_p_outer_alpha
    )
    dynamic_top_p_signal = validate_dynamic_top_p_signal(dynamic_top_p_signal)
    if (
        acceptance_rule
        == ACCEPTANCE_REJECTION_SAMPLING_THEN_ENTROPY_RISK_BUDGET_TOP_P
    ):
        dynamic_top_p_signal = DYNAMIC_TOP_P_SIGNAL_ENTROPY_ONLY
    risk_budget_max = validate_risk_budget_max(risk_budget_max)
    risk_budget_safe_top_p = float(risk_budget_safe_top_p)
    if acceptance_rule in {
        ACCEPTANCE_REJECTION_SAMPLING_THEN_RISK_BUDGET_TOP_P,
        ACCEPTANCE_REJECTION_SAMPLING_THEN_ENTROPY_RISK_BUDGET_TOP_P,
    }:
        risk_budget_safe_top_p, acceptance_top_p = (
            validate_risk_budget_top_p_thresholds(
                safe_top_p=risk_budget_safe_top_p,
                target_top_p=acceptance_top_p,
            )
        )
    cactus_delta = validate_cactus_delta(cactus_delta)
    mars_relaxation_threshold = validate_mars_relaxation_threshold(
        mars_relaxation_threshold
    )
    if not 0.0 <= float(target_sampling_top_p) <= 1.0:
        raise ValueError(
            "target_sampling_top_p must be in [0, 1], "
            f"got {target_sampling_top_p}."
        )
    if proposal.draft_token_count > max_proposal_tokens:
        raise ValueError(
            "DraftProposal.draft_token_count must not exceed "
            f"max_proposal_tokens={max_proposal_tokens}, "
            f"got {proposal.draft_token_count}."
        )
    if current_token_ids is not None and not torch.equal(
        proposal.verify_input_ids[:, :1],
        current_token_ids,
    ):
        raise ValueError(
            "DraftProposal.verify_input_ids must start with the current "
            "accepted token."
        )

    draft_token_count = int(proposal.draft_token_count)
    blind_accept_prefix_length = int(
        getattr(proposal, "blind_accept_prefix_length", 0)
    )
    if not 0 <= blind_accept_prefix_length <= draft_token_count:
        raise ValueError(
            "DraftProposal.blind_accept_prefix_length must be between zero "
            f"and draft_token_count={draft_token_count}, got "
            f"{blind_accept_prefix_length}."
        )
    verify_length = draft_token_count + 1
    verify_position_ids = position_ids[:, start : start + verify_length]
    if verification_phase_timer is not None:
        verification_phase_timer.start_target_model()
    target_output = target_model(
        input_ids=proposal.verify_input_ids,
        position_ids=verify_position_ids,
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
    )
    if verification_phase_timer is not None:
        verification_phase_timer.stop_target_model()
        verification_phase_timer.start_acceptance_rule()
    if target_output.logits.ndim != 3:
        raise ValueError(
            "target model must return rank-3 logits [B, S, V], "
            f"got ndim={target_output.logits.ndim}."
        )
    if acceptance_rule == ACCEPTANCE_ASD:
        if asd_state is None:
            raise ValueError("ASD requires a persistent request state")
        if blind_accept_prefix_length or force_reject_position is not None:
            raise ValueError("ASD cannot be combined with blind acceptance or forced rejection")
        tokens = proposal.verify_input_ids[:, 1:]
        target_probs = logits_to_probs(
            target_output.logits,
            float(temperature),
            top_p=float(target_sampling_top_p),
        )
        decision = compute_asd_prefix(
            target_logits=target_output.logits[:, :draft_token_count, :],
            proposed_tokens=tokens, state=asd_state, gate=asd_gate,
            cap=asd_max_exceptions, value_scheme=asd_value_scheme,
            stop_token_ids=stop_token_ids, max_emitted_tokens=asd_max_emitted_tokens,
        )
        accepted = decision.accepted_draft_tokens
        next_token = sample_from_probs(
            target_probs[:, accepted : accepted + 1, :]
        ).squeeze(1)
        committed = tokens[:, :accepted]
        if not decision.terminated_by_stop_token:
            committed = torch.cat([committed, next_token.unsqueeze(1)], dim=1)
        effective = accepted if decision.terminated_by_stop_token else draft_token_count
        logical = int(getattr(proposal, "logical_draft_block_count", 1))
        block_size = int(getattr(proposal, "draft_block_size", 0))
        if block_size > 0 and effective > 0:
            logical = min(logical, math.ceil(effective / block_size))
        result = VerificationResult(
            target_output=target_output,
            target_probs=target_probs,
            accept_prefix_mask=decision.accept_prefix_mask,
            accepted_draft_tokens=accepted, next_token=next_token,
            effective_proposal_length=effective,
            terminated_by_stop_token=decision.terminated_by_stop_token,
            committed_tokens=committed, accept_mask=decision.accept_mask,
            logical_draft_block_count=logical, asd_decision=decision,
            asd_statistics=asd_state.statistics(),
        )
        if verification_phase_timer is not None:
            verification_phase_timer.stop_acceptance_rule()
        return result
    target_probs = logits_to_probs(
        target_output.logits,
        float(temperature),
        top_p=float(target_sampling_top_p),
    )
    if (
        draft_token_count > 0
        and proposal.draft_probs is not None
        and proposal.draft_probs.size(-1) != target_probs.size(-1)
    ):
        raise ValueError(
            "DraftProposal.draft_probs vocab size must match target logits, "
            f"got {proposal.draft_probs.size(-1)} and {target_probs.size(-1)}."
        )

    accept_prefix_mask = None
    accept_mask = None
    standard_accept_mask = None
    standard_uniform_random = None
    target_top_p_accept_mask = None
    target_top_k_top_p_accept_mask = None
    dynamic_top_p_initial_accept_mask = None
    dynamic_top_p_initial_nucleus_size = None
    dynamic_top_p_entropy = None
    dynamic_top_p_tv_distance = None
    dynamic_top_p_threshold = None
    dynamic_top_p_accept_mask = None
    dynamic_top_p_rescue_mask = None
    dynamic_top_p_nucleus_size = None
    risk_budget_target_prefix_mass = None
    risk_budget_inside_target_top_p = None
    risk_budget_rescue_eligibility_mask = None
    risk_budget_token_risk = None
    risk_budget_block_risk_before = None
    risk_budget_candidate_block_risk = None
    risk_budget_block_risk_after = None
    risk_budget_rescue_mask = None
    risk_budget_reject_outside_top_p_mask = None
    risk_budget_reject_budget_mask = None
    selected_target_probs = None
    selected_draft_probs = None
    accept_prob = None
    cactus_adjusted_selected_probs = None
    cactus_accept_prob = None
    cactus_accept_mask = None
    mars_top2_logit_ratio = None
    mars_relaxation_mask = None
    mars_inside_target_top2 = None
    mars_relaxed_accept_mask = None
    mars_relaxed_reject_mask = None
    requested_force_reject_position = (
        int(force_reject_position) if force_reject_position is not None else None
    )
    forced_reject_position = None
    if draft_token_count > 0:
        assert proposal.draft_probs is not None
        proposed_tokens = proposal.verify_input_ids[:, 1:]
        selected_target_probs = gather_token_probs(
            target_probs[:, :-1, :],
            proposed_tokens,
        )
        selected_draft_probs = gather_token_probs(
            proposal.draft_probs,
            proposed_tokens,
        ).clamp_min(1e-8)
        accept_prob = torch.clamp(
            selected_target_probs / selected_draft_probs,
            max=1.0,
        )
        nucleus_membership = (clean_top_p_accept_mask if acceptance_rule == ACCEPTANCE_CLEAN_NSD
                              else _target_top_p_accept_mask)
        target_top_p_accept_mask = nucleus_membership(
            target_probs=target_probs[:, :-1, :],
            target_logits=(target_output.logits[:, :-1, :] if acceptance_rule == ACCEPTANCE_CLEAN_NSD
                           else target_output.logits[:, :-1, :].float()),
            proposed_tokens=proposed_tokens,
            acceptance_top_p=float(acceptance_top_p),
        )
        if acceptance_rule != ACCEPTANCE_CLEAN_NSD:
            target_top_k_top_p_accept_mask = _target_top_k_top_p_accept_mask(
                target_probs=target_probs[:, :-1, :],
                target_logits=target_output.logits[:, :-1, :].float(),
                proposed_tokens=proposed_tokens,
                acceptance_top_k=int(acceptance_top_k),
                acceptance_top_p=float(acceptance_top_p),
            )
        if acceptance_rule == ACCEPTANCE_CACTUS:
            cactus_adjusted_selected_probs = compute_cactus_adjusted_selected_probs(
                target_probs[:, :-1, :],
                proposed_tokens,
                cactus_delta,
            )
            cactus_accept_prob = torch.clamp(
                cactus_adjusted_selected_probs / selected_draft_probs,
                max=1.0,
            )
            cactus_accept_mask = (
                torch.rand_like(cactus_accept_prob) < cactus_accept_prob
            )
            accept_mask = cactus_accept_mask.to(torch.int64)
        elif acceptance_rule == ACCEPTANCE_MARS:
            mars_result = compute_mars_acceptance(
                target_probs=target_probs[:, :-1, :],
                target_logits=target_output.logits[:, :-1, :].float(),
                draft_probs=proposal.draft_probs,
                proposed_tokens=proposed_tokens,
                relaxation_threshold=mars_relaxation_threshold,
            )
            standard_uniform_random = mars_result.uniform_random
            standard_accept_mask = mars_result.standard_accept_mask
            mars_top2_logit_ratio = mars_result.top2_logit_ratio
            mars_relaxation_mask = mars_result.relaxation_mask
            mars_inside_target_top2 = mars_result.inside_target_top2
            mars_relaxed_accept_mask = mars_result.relaxed_accept_mask
            mars_relaxed_reject_mask = mars_result.relaxed_reject_mask
            accept_mask = mars_result.accept_mask.to(torch.int64)
        elif acceptance_rule in {
            ACCEPTANCE_CLEAN_NSD,
            ACCEPTANCE_REJECTION_SAMPLING,
            ACCEPTANCE_REJECTION_SAMPLING_THEN_TARGET_TOP_P,
            ACCEPTANCE_REJECTION_SAMPLING_THEN_DYNAMIC_TOP_P,
            ACCEPTANCE_REJECTION_SAMPLING_THEN_RISK_BUDGET_TOP_P,
            ACCEPTANCE_REJECTION_SAMPLING_THEN_ENTROPY_RISK_BUDGET_TOP_P,
            ACCEPTANCE_REJECTION_SAMPLING_THEN_TARGET_TOP_K_TOP_P,
        }:
            standard_uniform_random = torch.rand_like(accept_prob)
            standard_accept_mask = standard_uniform_random < accept_prob
            if acceptance_rule == ACCEPTANCE_REJECTION_SAMPLING:
                accept_mask = standard_accept_mask.to(torch.int64)
            elif acceptance_rule in {ACCEPTANCE_CLEAN_NSD, ACCEPTANCE_REJECTION_SAMPLING_THEN_TARGET_TOP_P}:
                accept_mask = (standard_accept_mask | target_top_p_accept_mask).to(
                    torch.int64
                )
            elif (
                acceptance_rule
                == ACCEPTANCE_REJECTION_SAMPLING_THEN_DYNAMIC_TOP_P
            ):
                dynamic_top_p_result = compute_dynamic_top_p(
                    target_probs=target_probs[:, :-1, :],
                    draft_probs=proposal.draft_probs,
                    proposed_tokens=proposed_tokens,
                    dynamic_top_p_blend=dynamic_top_p_blend,
                    dynamic_top_p_support=dynamic_top_p_support,
                    dynamic_top_p_outer_alpha=dynamic_top_p_outer_alpha,
                    dynamic_top_p_signal=dynamic_top_p_signal,
                )
                dynamic_top_p_initial_accept_mask = (
                    dynamic_top_p_result.inside_initial_nucleus
                )
                dynamic_top_p_initial_nucleus_size = (
                    dynamic_top_p_result.initial_nucleus_size
                )
                dynamic_top_p_entropy = (
                    dynamic_top_p_result.normalized_nucleus_entropy
                )
                dynamic_top_p_tv_distance = (
                    dynamic_top_p_result.draft_target_tv_distance
                )
                dynamic_top_p_threshold = dynamic_top_p_result.dynamic_top_p
                dynamic_top_p_accept_mask = (
                    dynamic_top_p_result.inside_dynamic_nucleus
                )
                dynamic_top_p_nucleus_size = (
                    dynamic_top_p_result.dynamic_nucleus_size
                )
                dynamic_top_p_rescue_mask = (
                    ~standard_accept_mask
                    & dynamic_top_p_initial_accept_mask
                    & dynamic_top_p_accept_mask
                )
                accept_mask = (
                    standard_accept_mask | dynamic_top_p_rescue_mask
                ).to(torch.int64)
            elif (
                acceptance_rule
                == ACCEPTANCE_REJECTION_SAMPLING_THEN_RISK_BUDGET_TOP_P
            ):
                risk_budget_result = compute_risk_budget_top_p(
                    target_probs=target_probs[:, :-1, :],
                    target_logits=target_output.logits[:, :-1, :].float(),
                    proposed_tokens=proposed_tokens,
                    standard_accept_mask=standard_accept_mask,
                    risk_budget_max=risk_budget_max,
                    risk_budget_safe_top_p=risk_budget_safe_top_p,
                    target_top_p=acceptance_top_p,
                )
                risk_budget_target_prefix_mass = (
                    risk_budget_result.target_prefix_mass
                )
                risk_budget_inside_target_top_p = (
                    risk_budget_result.inside_target_top_p
                )
                risk_budget_rescue_eligibility_mask = (
                    risk_budget_result.rescue_eligibility_mask
                )
                risk_budget_token_risk = risk_budget_result.token_risk
                risk_budget_block_risk_before = (
                    risk_budget_result.block_risk_before
                )
                risk_budget_candidate_block_risk = (
                    risk_budget_result.candidate_block_risk
                )
                risk_budget_block_risk_after = (
                    risk_budget_result.block_risk_after
                )
                risk_budget_rescue_mask = risk_budget_result.rescue_mask
                risk_budget_reject_outside_top_p_mask = (
                    risk_budget_result.reject_outside_top_p_mask
                )
                risk_budget_reject_budget_mask = (
                    risk_budget_result.reject_risk_budget_mask
                )
                accept_mask = risk_budget_result.accept_mask.to(torch.int64)
            elif (
                acceptance_rule
                == ACCEPTANCE_REJECTION_SAMPLING_THEN_ENTROPY_RISK_BUDGET_TOP_P
            ):
                dynamic_top_p_result = compute_dynamic_top_p(
                    target_probs=target_probs[:, :-1, :],
                    draft_probs=proposal.draft_probs,
                    proposed_tokens=proposed_tokens,
                    dynamic_top_p_blend=dynamic_top_p_blend,
                    dynamic_top_p_support=dynamic_top_p_support,
                    dynamic_top_p_outer_alpha=dynamic_top_p_outer_alpha,
                    dynamic_top_p_signal=DYNAMIC_TOP_P_SIGNAL_ENTROPY_ONLY,
                )
                dynamic_top_p_initial_accept_mask = (
                    dynamic_top_p_result.inside_initial_nucleus
                )
                dynamic_top_p_initial_nucleus_size = (
                    dynamic_top_p_result.initial_nucleus_size
                )
                dynamic_top_p_entropy = (
                    dynamic_top_p_result.normalized_nucleus_entropy
                )
                dynamic_top_p_tv_distance = (
                    dynamic_top_p_result.draft_target_tv_distance
                )
                dynamic_top_p_threshold = dynamic_top_p_result.dynamic_top_p
                dynamic_top_p_accept_mask = (
                    dynamic_top_p_result.inside_dynamic_nucleus
                )
                dynamic_top_p_nucleus_size = (
                    dynamic_top_p_result.dynamic_nucleus_size
                )
                dynamic_top_p_rescue_mask = (
                    ~standard_accept_mask
                    & dynamic_top_p_initial_accept_mask
                    & dynamic_top_p_accept_mask
                )
                target_prefix_mass = compute_target_prefix_mass(
                    target_probs=target_probs[:, :-1, :].float(),
                    target_logits=target_output.logits[:, :-1, :].float(),
                    proposed_tokens=proposed_tokens,
                )
                risk_budget_result = apply_block_risk_budget(
                    target_prefix_mass=target_prefix_mass,
                    inside_target_top_p=(
                        target_prefix_mass < float(acceptance_top_p)
                    ),
                    rescue_eligibility_mask=(
                        dynamic_top_p_initial_accept_mask
                        & dynamic_top_p_accept_mask
                    ),
                    standard_accept_mask=standard_accept_mask,
                    risk_budget_max=risk_budget_max,
                    risk_budget_safe_top_p=risk_budget_safe_top_p,
                    risk_reference_top_p=acceptance_top_p,
                )
                risk_budget_target_prefix_mass = (
                    risk_budget_result.target_prefix_mass
                )
                risk_budget_inside_target_top_p = (
                    risk_budget_result.inside_target_top_p
                )
                risk_budget_rescue_eligibility_mask = (
                    risk_budget_result.rescue_eligibility_mask
                )
                risk_budget_token_risk = risk_budget_result.token_risk
                risk_budget_block_risk_before = (
                    risk_budget_result.block_risk_before
                )
                risk_budget_candidate_block_risk = (
                    risk_budget_result.candidate_block_risk
                )
                risk_budget_block_risk_after = (
                    risk_budget_result.block_risk_after
                )
                risk_budget_rescue_mask = risk_budget_result.rescue_mask
                risk_budget_reject_outside_top_p_mask = (
                    risk_budget_result.reject_outside_top_p_mask
                )
                risk_budget_reject_budget_mask = (
                    risk_budget_result.reject_risk_budget_mask
                )
                accept_mask = risk_budget_result.accept_mask.to(torch.int64)
            else:
                accept_mask = (
                    standard_accept_mask | target_top_k_top_p_accept_mask
                ).to(torch.int64)
        elif acceptance_rule == ACCEPTANCE_TARGET_TOP_K_TOP_P:
            accept_mask = target_top_k_top_p_accept_mask.to(torch.int64)
        elif acceptance_rule in {
            ACCEPTANCE_TARGET_TOP_P,
            ACCEPTANCE_ONLY_TOP_P,
        }:
            accept_mask = target_top_p_accept_mask.to(torch.int64)
        else:
            raise AssertionError(
                f"Unhandled acceptance_rule={acceptance_rule!r}."
            )
        if blind_accept_prefix_length > 0:



            accept_mask = accept_mask.clone()
            accept_mask[:, :blind_accept_prefix_length] = 1
        if (
            requested_force_reject_position is not None
            and 0 <= requested_force_reject_position < draft_token_count
        ):
            forced_reject_position = requested_force_reject_position
            accept_mask = accept_mask.clone()
            accept_mask[:, forced_reject_position] = 0
        accept_prefix_mask = accept_mask.cumprod(dim=1)
        accepted_draft_tokens = int(accept_prefix_mask.sum(dim=1)[0].item())
    else:
        accepted_draft_tokens = 0

    effective_proposal_length = draft_token_count
    terminated_by_stop_token = False
    if stop_token_ids and accepted_draft_tokens > 0:
        accepted_slice = proposal.verify_input_ids[0, 1 : accepted_draft_tokens + 1]
        stop_tensor = torch.tensor(
            stop_token_ids,
            device=accepted_slice.device,
            dtype=accepted_slice.dtype,
        )
        eos_hits = torch.isin(accepted_slice, stop_tensor).nonzero(as_tuple=True)[0]
        if eos_hits.numel() > 0:
            eos_pos = int(eos_hits[0].item())
            accepted_draft_tokens = eos_pos + 1
            effective_proposal_length = eos_pos + 1
            terminated_by_stop_token = True

    draft_block_size = int(getattr(proposal, "draft_block_size", 0))
    logical_draft_block_count = int(
        getattr(proposal, "logical_draft_block_count", 1)
    )
    if draft_block_size > 0 and effective_proposal_length > 0:
        logical_draft_block_count = min(
            logical_draft_block_count,
            math.ceil(effective_proposal_length / draft_block_size),
        )

    if 0 < draft_token_count and accepted_draft_tokens < draft_token_count:
        assert proposal.draft_probs is not None
        recovery_target_probs = target_probs[:, accepted_draft_tokens, :]
        if acceptance_rule == ACCEPTANCE_CACTUS:
            recovery_target_probs = cactus_adjusted_distribution(
                recovery_target_probs,
                proposed_tokens[:, accepted_draft_tokens],
                cactus_delta,
            )
        mars_relaxation_rejection = (
            acceptance_rule == ACCEPTANCE_MARS
            and mars_relaxation_mask is not None
            and bool(
                mars_relaxation_mask[0, accepted_draft_tokens].item()
            )
        )
        if acceptance_rule == ACCEPTANCE_ONLY_TOP_P or mars_relaxation_rejection:



            next_token = sample_from_probs(
                recovery_target_probs.unsqueeze(1)
            ).squeeze(1)
        else:
            next_token = sample_residual(
                recovery_target_probs,
                proposal.draft_probs[:, accepted_draft_tokens, :],
            )
    else:
        next_token = sample_from_probs(target_probs[:, -1:, :]).squeeze(1)

    committed_tokens = torch.cat(
        [
            proposal.verify_input_ids[:, 1 : accepted_draft_tokens + 1],
            next_token.unsqueeze(1),
        ],
        dim=1,
    )

    result = VerificationResult(
        target_output=target_output,
        target_probs=target_probs,
        accept_prefix_mask=accept_prefix_mask,
        accepted_draft_tokens=accepted_draft_tokens,
        next_token=next_token,
        effective_proposal_length=effective_proposal_length,
        terminated_by_stop_token=terminated_by_stop_token,
        committed_tokens=committed_tokens,
        accept_mask=accept_mask,
        standard_accept_mask=standard_accept_mask,
        standard_uniform_random=standard_uniform_random,
        target_top_p_accept_mask=target_top_p_accept_mask,
        target_top_k_top_p_accept_mask=target_top_k_top_p_accept_mask,
        dynamic_top_p_initial_accept_mask=dynamic_top_p_initial_accept_mask,
        dynamic_top_p_initial_nucleus_size=dynamic_top_p_initial_nucleus_size,
        dynamic_top_p_entropy=dynamic_top_p_entropy,
        dynamic_top_p_tv_distance=dynamic_top_p_tv_distance,
        dynamic_top_p_threshold=dynamic_top_p_threshold,
        dynamic_top_p_blend=dynamic_top_p_blend,
        dynamic_top_p_support=dynamic_top_p_support,
        dynamic_top_p_outer_alpha=dynamic_top_p_outer_alpha,
        dynamic_top_p_signal=dynamic_top_p_signal,
        dynamic_top_p_accept_mask=dynamic_top_p_accept_mask,
        dynamic_top_p_rescue_mask=dynamic_top_p_rescue_mask,
        dynamic_top_p_nucleus_size=dynamic_top_p_nucleus_size,
        risk_budget_target_prefix_mass=risk_budget_target_prefix_mass,
        risk_budget_inside_target_top_p=risk_budget_inside_target_top_p,
        risk_budget_rescue_eligibility_mask=(
            risk_budget_rescue_eligibility_mask
        ),
        risk_budget_token_risk=risk_budget_token_risk,
        risk_budget_block_risk_before=risk_budget_block_risk_before,
        risk_budget_candidate_block_risk=risk_budget_candidate_block_risk,
        risk_budget_block_risk_after=risk_budget_block_risk_after,
        risk_budget_rescue_mask=risk_budget_rescue_mask,
        risk_budget_reject_outside_top_p_mask=(
            risk_budget_reject_outside_top_p_mask
        ),
        risk_budget_reject_budget_mask=risk_budget_reject_budget_mask,
        risk_budget_max=risk_budget_max,
        risk_budget_safe_top_p=risk_budget_safe_top_p,
        selected_target_probs=selected_target_probs,
        selected_draft_probs=selected_draft_probs,
        accept_prob=accept_prob,
        cactus_adjusted_selected_probs=cactus_adjusted_selected_probs,
        cactus_accept_prob=cactus_accept_prob,
        cactus_accept_mask=cactus_accept_mask,
        cactus_delta=cactus_delta,
        mars_top2_logit_ratio=mars_top2_logit_ratio,
        mars_relaxation_mask=mars_relaxation_mask,
        mars_inside_target_top2=mars_inside_target_top2,
        mars_relaxed_accept_mask=mars_relaxed_accept_mask,
        mars_relaxed_reject_mask=mars_relaxed_reject_mask,
        mars_relaxation_threshold=mars_relaxation_threshold,
        forced_reject_position=forced_reject_position,
        blind_accept_prefix_length=blind_accept_prefix_length,
        logical_draft_block_count=logical_draft_block_count,
    )
    if verification_phase_timer is not None:
        verification_phase_timer.stop_acceptance_rule()
    return result


@torch.inference_mode()
def generate_decoding_sample(
    *,
    target_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    max_proposal_tokens: int,
    temperature: float,
    stop_token_ids: list[int] | None,
    init_context: Callable[..., Any],
    propose: Callable[..., DraftProposal],
    update: Callable[[Any, VerificationResult], None],
    post_verify: Callable[..., None] | None = None,
    dataset_name: str | None = None,
    sample_index: int | None = None,
    acceptance_rule: str = ACCEPTANCE_REJECTION_SAMPLING,
    acceptance_top_k: int = 2,
    acceptance_top_p: float = 0.95,
    dynamic_top_p_blend: float = 0.0,
    dynamic_top_p_support: float = 0.95,
    dynamic_top_p_outer_alpha: float = 0.0,
    dynamic_top_p_signal: str = DYNAMIC_TOP_P_SIGNAL_ENTROPY_DISAGREEMENT,
    risk_budget_max: float = DEFAULT_RISK_BUDGET_MAX,
    risk_budget_safe_top_p: float = DEFAULT_RISK_BUDGET_SAFE_TOP_P,
    cactus_delta: float = 1.0,
    mars_relaxation_threshold: float = DEFAULT_MARS_RELAXATION_THRESHOLD,
    target_sampling_top_p: float = 1.0,
    replay_prefix_token_ids: torch.Tensor | list[int] | None = None,
    force_reject_proposal_index: int | None = None,
    force_reject_proposal_position: int | None = None,
    force_reject_generated_position: int | None = None,
    asd_budget: float = 8.0,
    asd_gate: float = 0.25,
    asd_max_exceptions: int = 2,
    asd_value_scheme: str = "suffix_length",
    record_verification_timing: bool = False,
) -> SimpleNamespace:
    """Speculative-decoding loop.

    `init_context(initial_output, output_ids, position_ids, num_input_tokens)`
    builds the algorithm-specific state once after prefill. For prefix replay,
    `num_input_tokens` is the length of the rebuilt prefill context
    (prompt + prefix[:-1]), while the response still reports the original
    prompt length. `propose(context,
    output_ids, position_ids, start, stop_token_ids)` returns the next
    DraftProposal. `update(context, verification)` advances the state when the
    loop continues. `post_verify` is an optional diagnostic hook called after
    every verification with proposal metadata (used for confidence calibration
    and rejected-token diagnostics).
    """
    assert max_proposal_tokens >= 1
    assert input_ids.size(0) == 1, "only bsz=1 is supported"

    device = input_ids.device
    verification_phase_timer = (
        VerificationPhaseTimer(device) if record_verification_timing else None
    )
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + int(max_new_tokens)

    output_ids = torch.empty(
        (1, max_length + max_proposal_tokens + 1),
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    past_key_values_target = build_dynamic_cache(getattr(target_model, "config", None))

    asd_state = ASDState(asd_budget) if acceptance_rule == ACCEPTANCE_ASD else None
    output_ids[:, :num_input_tokens] = input_ids
    acceptance_lengths: list[int] = []
    proposal_lengths: list[int] = []
    accepted_draft_lengths: list[int] = []
    blind_accept_prefix_lengths: list[int] = []
    logical_draft_block_counts: list[int] = []
    forced_reject_events: list[dict[str, int]] = []
    risk_budget_rescue_count = 0
    risk_budget_reject_outside_top_p_count = 0
    risk_budget_reject_budget_count = 0
    replay_prefix_tokens = 0

    if replay_prefix_token_ids is not None:
        if not torch.is_tensor(replay_prefix_token_ids):
            replay_prefix_token_ids = torch.tensor(
                replay_prefix_token_ids,
                dtype=torch.long,
                device=device,
            )
        replay_prefix_token_ids = replay_prefix_token_ids.to(
            device=device,
            dtype=torch.long,
        ).view(1, -1)
        replay_prefix_tokens = int(replay_prefix_token_ids.shape[1])
        if replay_prefix_tokens > int(max_new_tokens):
            raise ValueError(
                "replay prefix is longer than max_new_tokens: "
                f"{replay_prefix_tokens} > {int(max_new_tokens)}."
            )

    if replay_prefix_tokens > 0:
        output_ids[
            :,
            num_input_tokens : num_input_tokens + replay_prefix_tokens,
        ] = replay_prefix_token_ids
        if replay_prefix_tokens == 1:
            prefill_input_ids = input_ids
        else:
            prefill_input_ids = torch.cat(
                [input_ids, replay_prefix_token_ids[:, :-1]],
                dim=1,
            )
        prefill_tokens = int(prefill_input_ids.shape[1])
        output = target_model(
            input_ids=prefill_input_ids,
            position_ids=position_ids[:, :prefill_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        start = num_input_tokens + replay_prefix_tokens - 1
    else:
        output = target_model(
            input_ids=input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )

        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample_from_probs(
            logits_to_probs(
                output.logits,
                float(temperature),
                top_p=float(target_sampling_top_p),
            )
        )
        start = input_ids.shape[1]
        prefill_tokens = num_input_tokens

    initial_token = output_ids[:, start : start + 1]
    if has_stop_token(initial_token, stop_token_ids):
        output_ids = output_ids[:, : start + 1]
        output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)
        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=output_ids.shape[1] - num_input_tokens,
            acceptance_lengths=acceptance_lengths,
            proposal_lengths=proposal_lengths,
            accepted_draft_lengths=accepted_draft_lengths,
            blind_accept_prefix_lengths=blind_accept_prefix_lengths,
            logical_draft_block_counts=logical_draft_block_counts,
            blind_accept_event_count=0,
            blind_accepted_token_count=0,
            verify_count=0,
            asd_statistics=asd_state.statistics() if asd_state is not None else None,
            replay_prefix_tokens=replay_prefix_tokens,
            forced_reject_events=forced_reject_events,
            risk_budget_rescue_count=0,
            risk_budget_reject_outside_top_p_count=0,
            risk_budget_reject_budget_count=0,
            verification_timing=(
                verification_phase_timer.summary()
                if verification_phase_timer is not None else None
            ),
        )

    context = init_context(
        initial_output=output,
        output_ids=output_ids,
        position_ids=position_ids,
        num_input_tokens=prefill_tokens,
    )
    if verification_phase_timer is not None:
        verification_phase_timer.start_generation()

    while start < max_length:
        proposal_index = len(proposal_lengths)
        proposal = propose(
            context=context,
            output_ids=output_ids,
            position_ids=position_ids,
            start=start,
            stop_token_ids=stop_token_ids,
        )
        force_reject_position = None
        if force_reject_generated_position is not None:
            maybe_position = int(force_reject_generated_position) - int(start) - 1
            if 0 <= maybe_position < int(proposal.draft_token_count):
                force_reject_position = maybe_position
        if (
            force_reject_proposal_index is not None
            and int(force_reject_proposal_index) == proposal_index
        ):
            force_reject_position = int(force_reject_proposal_position)
        verification = verify_draft_tokens(
            target_model=target_model,
            proposal=proposal,
            position_ids=position_ids,
            start=start,
            past_key_values_target=past_key_values_target,
            temperature=temperature,
            max_proposal_tokens=max_proposal_tokens,
            current_token_ids=output_ids[:, start : start + 1],
            stop_token_ids=stop_token_ids,
            acceptance_rule=acceptance_rule,
            acceptance_top_k=acceptance_top_k,
            acceptance_top_p=acceptance_top_p,
            dynamic_top_p_blend=dynamic_top_p_blend,
            dynamic_top_p_support=dynamic_top_p_support,
            dynamic_top_p_outer_alpha=dynamic_top_p_outer_alpha,
            dynamic_top_p_signal=dynamic_top_p_signal,
            risk_budget_max=risk_budget_max,
            risk_budget_safe_top_p=risk_budget_safe_top_p,
            cactus_delta=cactus_delta,
            mars_relaxation_threshold=mars_relaxation_threshold,
            target_sampling_top_p=target_sampling_top_p,
            force_reject_position=force_reject_position,
            asd_state=asd_state, asd_gate=asd_gate,
            asd_max_exceptions=asd_max_exceptions, asd_value_scheme=asd_value_scheme,
            asd_max_emitted_tokens=max(0, max_length - start - 1),
            verification_phase_timer=verification_phase_timer,
        )
        if verification.forced_reject_position is not None:
            forced_reject_events.append(
                {
                    "proposal_index": int(proposal_index),
                    "proposal_position": int(verification.forced_reject_position),
                    "generated_position": int(start)
                    + int(verification.forced_reject_position)
                    + 1,
                }
            )
        risk_count_length = int(verification.effective_proposal_length)
        if verification.risk_budget_rescue_mask is not None:
            risk_budget_rescue_count += int(
                verification.risk_budget_rescue_mask[
                    :, :risk_count_length
                ].sum().item()
            )
        if verification.risk_budget_reject_outside_top_p_mask is not None:
            risk_budget_reject_outside_top_p_count += int(
                verification.risk_budget_reject_outside_top_p_mask[
                    :, :risk_count_length
                ].sum().item()
            )
        if verification.risk_budget_reject_budget_mask is not None:
            risk_budget_reject_budget_count += int(
                verification.risk_budget_reject_budget_mask[
                    :, :risk_count_length
                ].sum().item()
            )
        if post_verify is not None:
            post_verify(
                proposal,
                verification,
                start=start,
                proposal_index=proposal_index,
                dataset_name=dataset_name,
                sample_index=sample_index,
            )

        proposal_lengths.append(int(verification.effective_proposal_length))
        accepted_draft_tokens = int(verification.accepted_draft_tokens)
        accepted_draft_lengths.append(accepted_draft_tokens)
        blind_accept_prefix_lengths.append(
            int(verification.blind_accept_prefix_length)
        )
        logical_draft_block_counts.append(
            int(verification.logical_draft_block_count)
        )

        output_ids[:, start : start + accepted_draft_tokens + 1] = (
            proposal.verify_input_ids[:, : accepted_draft_tokens + 1]
        )

        if verification.terminated_by_stop_token:
            acceptance_lengths.append(accepted_draft_tokens)
            start += accepted_draft_tokens
            past_key_values_target.crop(start)
            break

        output_ids[:, start + accepted_draft_tokens + 1] = verification.next_token
        new_token_ids = output_ids[:, start + 1 : start + accepted_draft_tokens + 2]
        acceptance_lengths.append(accepted_draft_tokens + 1)
        start += accepted_draft_tokens + 1
        past_key_values_target.crop(start)
        update(context, verification)

        if has_stop_token(new_token_ids, stop_token_ids):
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)
    if verification_phase_timer is not None:
        verification_phase_timer.stop_generation()
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=output_ids.shape[1] - num_input_tokens,
        acceptance_lengths=acceptance_lengths,
        proposal_lengths=proposal_lengths,
        accepted_draft_lengths=accepted_draft_lengths,
        blind_accept_prefix_lengths=blind_accept_prefix_lengths,
        logical_draft_block_counts=logical_draft_block_counts,
        blind_accept_event_count=sum(
            int(prefix_length > 0)
            for prefix_length in blind_accept_prefix_lengths
        ),
        blind_accepted_token_count=sum(
            min(prefix_length, accepted_length)
            for prefix_length, accepted_length in zip(
                blind_accept_prefix_lengths,
                accepted_draft_lengths,
            )
        ),
        verify_count=len(proposal_lengths),
        replay_prefix_tokens=replay_prefix_tokens,
        forced_reject_events=forced_reject_events,
        asd_statistics=asd_state.statistics() if asd_state is not None else None,
        risk_budget_rescue_count=risk_budget_rescue_count,
        risk_budget_reject_outside_top_p_count=(
            risk_budget_reject_outside_top_p_count
        ),
        risk_budget_reject_budget_count=risk_budget_reject_budget_count,
        verification_timing=(
            verification_phase_timer.summary()
            if verification_phase_timer is not None else None
        ),
    )


class BaseEvaluator:

    supports_periodic_blind_accept = False

    def should_record_verification_timing(self) -> bool:
        if not bool(getattr(self.args, "record_verification_timing", False)):
            return False
        warmup = int(getattr(self.args, "verification_timing_warmup_samples", 1))
        return int(getattr(self, "_current_dataset_ordinal", 0)) >= warmup

    def __init__(self, local_rank: int, args):
        self.args = args
        self._validate_generation_controls()
        self.device, self.global_rank, self.world_size = init_dist(local_rank)
        self.tasks = args.tasks

        self.target_model, self.draft_model, self.tokenizer = self.build_models()
        self.replay_records_by_sample_index = self._load_replay_records()
        self.token_diagnostics_recorder = self._build_token_diagnostics_recorder()
        diagnostic_tasks = getattr(self.args, "token_diagnostics_tasks", None)
        self.token_diagnostics_tasks = (
            None if diagnostic_tasks is None else set(diagnostic_tasks)
        )
        self.metrics_rows: list[dict[str, object]] = []

    def _prediction_dataset_dir(self, dataset_name: str) -> Path | None:
        if not bool(getattr(self.args, "dump_predictions", False)):
            return None

        predictions_dir = getattr(self.args, "predictions_dir", None)
        if predictions_dir:
            root = Path(predictions_dir)
        elif getattr(self.args, "tensorboard_dir", None):
            root = Path(self.args.tensorboard_dir) / "artifacts"
        else:
            raise ValueError(
                "Prediction dumping requires --predictions-dir or "
                "--tensorboard-dir."
            )

        step = getattr(self.args, "step", None)
        step_name = "step_unknown" if step is None else f"step_{int(step)}"
        return root / step_name / str(dataset_name)

    def _build_prediction_record(
        self,
        *,
        dataset_name: str,
        sample_index: int,
        instance: dict[str, Any],
        response: SimpleNamespace,
    ) -> dict[str, Any]:
        output_token_ids = (
            response.output_ids[0, response.num_input_tokens :]
            .detach()
            .cpu()
            .tolist()
        )
        output_text = self.tokenizer.decode(
            output_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_token_count = int(response.num_output_tokens)
        record = {
            "schema_version": 1,
            "dataset": str(dataset_name),
            "sample_index": int(sample_index),
            "source_index": instance.get("source_index", sample_index),
            "source_id": instance.get("source_id"),
            "source_dataset": instance.get("source_dataset"),
            "source_config": instance.get("source_config"),
            "source_split": instance.get("source_split"),
            "rank": int(self.global_rank),
            "world_size": int(self.world_size),
            "prompt": instance["turns"][0],
            "turns": instance.get("all_turns", instance["turns"]),
            "evaluated_turn_count": 1,
            "reference": instance.get("reference"),
            "source": instance.get("source"),
            "output": output_text,
            "generated_token_ids": [int(token_id) for token_id in output_token_ids],
            "input_token_count": int(response.num_input_tokens),
            "output_token_count": output_token_count,
            "max_new_tokens": int(self.args.max_new_tokens),
            "asd_statistics": getattr(response, "asd_statistics", None),
            "verification_timing": getattr(response, "verification_timing", None),
            "reached_max_new_tokens": (
                output_token_count >= int(self.args.max_new_tokens)
            ),
            "verify_count": int(getattr(response, "verify_count", 0)),
            "acceptance_lengths": [
                int(value)
                for value in getattr(response, "acceptance_lengths", [])
            ],
            "proposal_lengths": [
                int(value) for value in getattr(response, "proposal_lengths", [])
            ],
            "accepted_draft_lengths": [
                int(value)
                for value in getattr(response, "accepted_draft_lengths", [])
            ],
            "blind_accept_prefix_lengths": [
                int(value)
                for value in getattr(response, "blind_accept_prefix_lengths", [])
            ],
            "logical_draft_block_counts": [
                int(value)
                for value in getattr(response, "logical_draft_block_counts", [])
            ],
            "blind_accept_event_count": int(
                getattr(response, "blind_accept_event_count", 0)
            ),
            "blind_accepted_token_count": int(
                getattr(response, "blind_accepted_token_count", 0)
            ),
            "risk_budget_rescue_count": int(
                getattr(response, "risk_budget_rescue_count", 0)
            ),
            "risk_budget_reject_outside_top_p_count": int(
                getattr(response, "risk_budget_reject_outside_top_p_count", 0)
            ),
            "risk_budget_reject_budget_count": int(
                getattr(response, "risk_budget_reject_budget_count", 0)
            ),
            "replay_prefix_tokens": int(
                getattr(response, "replay_prefix_tokens", 0)
            ),
            "forced_reject_events": getattr(
                response,
                "forced_reject_events",
                [],
            ),
        }
        if "messages" in instance:
            record["messages"] = instance["messages"]
        if "assistant_prefix" in instance:
            record["assistant_prefix"] = instance["assistant_prefix"]
        if "evaluation_profile" in instance:
            record["evaluation_profile"] = instance["evaluation_profile"]
        return record

    @staticmethod
    def _prediction_json_default(value: Any):
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    @property
    def max_proposal_tokens(self) -> int:
        raise NotImplementedError

    def build_models(self):
        raise NotImplementedError

    def _optional_int_arg(self, name: str) -> int | None:
        value = getattr(self.args, name, None)
        if value is None or value == "":
            return None
        return int(value)

    def _validate_generation_controls(self) -> None:
        proposal_index = getattr(self.args, "force_reject_proposal_index", None)
        proposal_position = getattr(self.args, "force_reject_proposal_position", None)
        generated_position = getattr(self.args, "force_reject_generated_position", None)
        if proposal_index not in (None, "") and generated_position not in (None, ""):
            raise ValueError(
                "Use either force_reject_proposal_index/position or "
                "force_reject_generated_position, not both."
            )
        if proposal_index in (None, "") and proposal_position not in (None, ""):
            raise ValueError(
                "force_reject_proposal_position requires "
                "force_reject_proposal_index."
            )
        if proposal_index not in (None, "") and proposal_position in (None, ""):
            raise ValueError(
                "force_reject_proposal_index requires "
                "force_reject_proposal_position."
            )
        replay_prefix_tokens = getattr(self.args, "replay_prefix_tokens", None)
        if replay_prefix_tokens not in (None, "") and int(replay_prefix_tokens) < 0:
            raise ValueError("replay_prefix_tokens must be >= 0.")
        blind_period = int(
            getattr(self.args, "blind_accept_every_nth_draft_block", 0)
        )
        if blind_period == 1 or blind_period < 0:
            raise ValueError(
                "blind_accept_every_nth_draft_block must be 0 or >= 2."
            )
        if blind_period > 0:
            if not self.supports_periodic_blind_accept:
                raise ValueError(
                    "Periodic blind block acceptance is supported only by "
                    "DSpark evaluators."
                )
            if getattr(self.args, "acceptance_rule", None) != ACCEPTANCE_REJECTION_SAMPLING:
                raise ValueError(
                    "Periodic blind block acceptance requires "
                    "acceptance_rule='rejection_sampling'."
                )
            if int(getattr(self.args, "draft_proposal_blocks", 1)) != 1:
                raise ValueError(
                    "Periodic blind block acceptance requires "
                    "draft_proposal_blocks=1."
                )
            if float(getattr(self.args, "confidence_threshold", 0.0)) != 0.0:
                raise ValueError(
                    "Periodic blind block acceptance requires "
                    "confidence_threshold=0."
                )
            if any(
                value not in (None, "")
                for value in (
                    proposal_index,
                    proposal_position,
                    generated_position,
                )
            ):
                raise ValueError(
                    "Periodic blind block acceptance cannot be combined with "
                    "forced-rejection interventions."
                )

    def _load_replay_records(self) -> dict[int, list[int]]:
        samples_path = getattr(self.args, "replay_samples_jsonl", None)
        if not samples_path:
            return {}

        replay_sample_index = self._optional_int_arg("replay_sample_index")
        records: dict[int, list[int]] = {}
        with Path(samples_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sample_index = int(record["sample_index"])
                if (
                    replay_sample_index is not None
                    and sample_index != replay_sample_index
                ):
                    continue
                generated_token_ids = record.get("generated_token_ids")
                if not isinstance(generated_token_ids, list):
                    raise ValueError(
                        f"{samples_path}:{line_number} must contain "
                        "a list field `generated_token_ids`."
                    )
                records[sample_index] = [int(token_id) for token_id in generated_token_ids]

        if replay_sample_index is not None and replay_sample_index not in records:
            raise ValueError(
                f"replay sample_index={replay_sample_index} was not found in "
                f"{samples_path}."
            )
        return records

    def _build_token_diagnostics_recorder(self) -> TokenDiagnosticsRecorder | None:
        if not bool(getattr(self.args, "record_token_diagnostics", False)):
            return None
        if self.args.tensorboard_dir is None:
            raise ValueError(
                "--tensorboard-dir is required when token diagnostics are enabled."
            )
        return TokenDiagnosticsRecorder(
            tokenizer=self.tokenizer,
            tensorboard_dir=self.args.tensorboard_dir,
            step=self.args.step,
            rank=self.global_rank,
            world_size=self.world_size,
            mode=getattr(self.args, "token_diagnostics_mode", "all"),
            record_exact_nsd_rho=bool(
                getattr(self.args, "record_exact_nsd_rho", False)
            ),
            acceptance_rule=getattr(self.args, "acceptance_rule", "rejection_sampling"),
            acceptance_top_k=int(getattr(self.args, "acceptance_top_k", 2)),
            acceptance_top_p=float(getattr(self.args, "acceptance_top_p", 0.95)),
            dynamic_top_p_blend=float(
                getattr(self.args, "dynamic_top_p_blend", 0.0)
            ),
            dynamic_top_p_support=float(
                getattr(self.args, "dynamic_top_p_support", 0.95)
            ),
            dynamic_top_p_outer_alpha=float(
                getattr(self.args, "dynamic_top_p_outer_alpha", 0.0)
            ),
            dynamic_top_p_signal=str(
                getattr(
                    self.args,
                    "dynamic_top_p_signal",
                    DYNAMIC_TOP_P_SIGNAL_ENTROPY_DISAGREEMENT,
                )
            ),
            risk_budget_max=float(
                getattr(self.args, "risk_budget_max", DEFAULT_RISK_BUDGET_MAX)
            ),
            risk_budget_safe_top_p=float(
                getattr(
                    self.args,
                    "risk_budget_safe_top_p",
                    DEFAULT_RISK_BUDGET_SAFE_TOP_P,
                )
            ),
            cactus_delta=float(getattr(self.args, "cactus_delta", 1.0)),
            mars_relaxation_threshold=float(
                getattr(
                    self.args,
                    "mars_relaxation_threshold",
                    DEFAULT_MARS_RELAXATION_THRESHOLD,
                )
            ),
            target_sampling_top_p=float(
                getattr(self.args, "target_sampling_top_p", 1.0)
            ),
        )

    def start_dataset_diagnostics(self, dataset_name: str) -> None:
        if (
            self.token_diagnostics_recorder is not None
            and (
                self.token_diagnostics_tasks is None
                or dataset_name in self.token_diagnostics_tasks
            )
        ):
            self.token_diagnostics_recorder.start(dataset_name)

    def finish_dataset_diagnostics(self) -> Path | None:
        if self.token_diagnostics_recorder is None:
            return None
        return self.token_diagnostics_recorder.finish()

    def _post_verify(
        self,
        proposal: DraftProposal,
        verification: VerificationResult,
        *,
        start: int,
        proposal_index: int,
        dataset_name: str | None = None,
        sample_index: int | None = None,
    ) -> None:
        if self.token_diagnostics_recorder is not None:
            self.token_diagnostics_recorder.observe(
                proposal=proposal,
                verification=verification,
                start=start,
                proposal_index=proposal_index,
                sample_index=sample_index,
            )

    def replay_prefix_token_ids_for_sample(
        self,
        *,
        sample_index: int | None,
    ) -> torch.Tensor | None:
        if sample_index is None or not self.replay_records_by_sample_index:
            return None
        sample_index = int(sample_index)
        if sample_index not in self.replay_records_by_sample_index:
            return None
        token_ids = self.replay_records_by_sample_index[sample_index]
        prefix_tokens = self._optional_int_arg("replay_prefix_tokens")
        if prefix_tokens is None:
            prefix_tokens = len(token_ids)
        if prefix_tokens > len(token_ids):
            raise ValueError(
                f"Requested replay_prefix_tokens={prefix_tokens}, but sample "
                f"{sample_index} has only {len(token_ids)} generated tokens."
            )
        if prefix_tokens <= 0:
            return None
        return torch.tensor(
            token_ids[:prefix_tokens],
            dtype=torch.long,
            device=self.device,
        ).view(1, -1)

    def generation_intervention_kwargs(
        self,
        *,
        sample_index: int | None,
    ) -> dict[str, int | None]:
        force_sample_index = self._optional_int_arg("force_reject_sample_index")
        if (
            force_sample_index is not None
            and sample_index is not None
            and int(sample_index) != force_sample_index
        ):
            return {
                "force_reject_proposal_index": None,
                "force_reject_proposal_position": None,
                "force_reject_generated_position": None,
            }
        if force_sample_index is not None and sample_index is None:
            return {
                "force_reject_proposal_index": None,
                "force_reject_proposal_position": None,
                "force_reject_generated_position": None,
            }
        return {
            "force_reject_proposal_index": self._optional_int_arg(
                "force_reject_proposal_index"
            ),
            "force_reject_proposal_position": self._optional_int_arg(
                "force_reject_proposal_position"
            ),
            "force_reject_generated_position": self._optional_int_arg(
                "force_reject_generated_position"
            ),
        }

    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,
        stop_token_ids: list[int] | None,
        dataset_name: str | None = None,
        sample_index: int | None = None,
    ) -> SimpleNamespace:
        raise NotImplementedError

    def build_metrics_row(
        self,
        *,
        dataset_name: str,
        metric_summary: dict[str, int | float | list[int] | None],
    ) -> dict[str, object]:
        proposal_count = int(metric_summary["proposal_count"])
        proposal_token_count = int(metric_summary["proposal_length_sum"])
        accepted_draft_token_sum = int(
            metric_summary.get("accepted_draft_token_sum")
            or max(
                0,
                int(metric_summary["acceptance_length_sum"]) - proposal_count,
            )
        )
        if proposal_count == 0:
            acceptance_length = 0.0
            draft_tokens_per_proposal = 0.0
            verify_rate = 0.0
            accept_rates_by_position = [None] * self.max_proposal_tokens
        else:
            acceptance_length = (
                int(metric_summary["acceptance_length_sum"]) / proposal_count
            )
            draft_tokens_per_proposal = (
                int(metric_summary["proposal_length_sum"]) / proposal_count
            )
            verify_rate = int(metric_summary["acceptance_length_sum"]) / (
                int(metric_summary["proposal_length_sum"]) + proposal_count
            )
            proposals_at_pos = metric_summary["proposals_at_pos"]
            accepted_at_pos = metric_summary["accepted_at_pos"]
            assert isinstance(proposals_at_pos, list)
            assert isinstance(accepted_at_pos, list)
            accept_rates_by_position = []
            for pos_idx in range(self.max_proposal_tokens):
                position_proposal_count = proposals_at_pos[pos_idx]
                if position_proposal_count == 0:
                    accept_rates_by_position.append(None)
                    continue
                accept_rates_by_position.append(
                    accepted_at_pos[pos_idx] / position_proposal_count
                )
        draft_token_acceptance_rate = (
            accepted_draft_token_sum / proposal_token_count
            if proposal_token_count > 0
            else 0.0
        )
        risk_budget_rescue_count = int(
            metric_summary.get("risk_budget_rescue_count") or 0
        )
        risk_budget_decision_count = (
            risk_budget_rescue_count
            + int(
                metric_summary.get("risk_budget_reject_outside_top_p_count")
                or 0
            )
            + int(metric_summary.get("risk_budget_reject_budget_count") or 0)
        )
        risk_budget_rescue_acceptance_rate = (
            risk_budget_rescue_count / risk_budget_decision_count
            if risk_budget_decision_count > 0
            else 0.0
        )
        elapsed_seconds = metric_summary.get("elapsed_seconds")
        output_token_count = int(metric_summary.get("output_token_count") or 0)
        sample_count = int(metric_summary["sample_count"])
        if elapsed_seconds is None or float(elapsed_seconds) <= 0.0:
            throughput_tokens_per_second = None
            throughput_samples_per_second = None
        else:
            throughput_tokens_per_second = (
                output_token_count / float(elapsed_seconds)
            )
            throughput_samples_per_second = sample_count / float(elapsed_seconds)
        return {
            "dataset": dataset_name,
            "num_samples": sample_count,
            "draft_tokens_per_proposal": draft_tokens_per_proposal,
            "acceptance_length": acceptance_length,
            "verify_rate": verify_rate,
            "accept_rates_by_position": accept_rates_by_position,
            "draft_token_acceptance_rate": draft_token_acceptance_rate,
            "risk_budget_rescue_acceptance_rate": (
                risk_budget_rescue_acceptance_rate
            ),
            "output_token_count": output_token_count,
            "elapsed_seconds": elapsed_seconds,
            "throughput_tokens_per_second": throughput_tokens_per_second,
            "throughput_samples_per_second": throughput_samples_per_second,
            "logical_draft_block_count": int(
                metric_summary.get("logical_draft_block_count") or proposal_count
            ),
            "blind_accept_event_count": int(
                metric_summary.get("blind_accept_event_count") or 0
            ),
            "blind_accepted_token_count": int(
                metric_summary.get("blind_accepted_token_count") or 0
            ),
            "risk_budget_rescue_count": int(
                metric_summary.get("risk_budget_rescue_count") or 0
            ),
            "risk_budget_reject_outside_top_p_count": int(
                metric_summary.get("risk_budget_reject_outside_top_p_count") or 0
            ),
            "risk_budget_reject_budget_count": int(
                metric_summary.get("risk_budget_reject_budget_count") or 0
            ),
        }

    def run_dataset(
        self,
        *,
        dataset_name: str,
        max_samples: int | None,
    ) -> list[SimpleNamespace]:
        seed_all(int(self.args.seed))
        dataset = load_and_process_dataset(
            dataset_name,
            dataset_root=getattr(self.args, "dataset_root", DEFAULT_DATASET_ROOT),
        )

        if max_samples is not None and len(dataset) > max_samples:
            rng = random.Random(int(getattr(self.args, "dataset_seed", self.args.seed)))
            dataset = list(dataset)
            rng.shuffle(dataset)
            dataset = dataset[:max_samples]

        stop_token_ids = resolve_stop_token_ids(self.target_model, self.tokenizer)
        responses = []
        prediction_dir = self._prediction_dataset_dir(dataset_name)
        prediction_path = None
        prediction_inprogress_path = None
        prediction_handle = None
        prediction_count = 0
        completed = False
        if prediction_dir is not None:
            prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = (
                prediction_dir
                / f"predictions.rank{int(self.global_rank):05d}.jsonl"
            )
            prediction_inprogress_path = prediction_path.with_suffix(
                prediction_path.suffix + ".inprogress"
            )
            prediction_handle = prediction_inprogress_path.open(
                "w",
                encoding="utf-8",
            )

        selected_indices = getattr(self.args, "sample_indices", None)
        if selected_indices is None:
            dataset_indices = list(range(len(dataset)))
        else:
            out_of_range = [
                int(index)
                for index in selected_indices
                if not 0 <= int(index) < len(dataset)
            ]
            if out_of_range:
                raise ValueError(
                    f"sample indices {out_of_range} are outside dataset "
                    f"{dataset_name!r} with {len(dataset)} selected rows."
                )
            dataset_indices = [int(index) for index in selected_indices]

        try:
            for ordinal, idx in enumerate(dataset_indices):
                if ordinal % self.world_size != self.global_rank:
                    continue
                seed_all(int(self.args.seed) + idx)
                self._current_dataset_ordinal = int(ordinal)
                instance = dataset[idx]
                input_ids = encode_evaluation_instance(
                    self.tokenizer,
                    instance,
                ).to(self.device)
                response = self.generate_one_sample(
                    input_ids=input_ids,
                    stop_token_ids=stop_token_ids,
                    dataset_name=dataset_name,
                    sample_index=idx,
                )
                response.sample_index = idx
                responses.append(response)

                if prediction_handle is not None:
                    record = self._build_prediction_record(
                        dataset_name=dataset_name,
                        sample_index=idx,
                        instance=instance,
                        response=response,
                    )
                    prediction_handle.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            default=self._prediction_json_default,
                        )
                        + "\n"
                    )
                    prediction_handle.flush()
                    prediction_count += 1
            completed = True
        finally:
            if prediction_handle is not None:
                prediction_handle.close()

        if completed and prediction_path is not None:
            assert prediction_inprogress_path is not None
            os.replace(prediction_inprogress_path, prediction_path)
            manifest_path = prediction_path.with_name(
                f"predictions_manifest.rank{int(self.global_rank):05d}.json"
            )
            manifest = {
                "schema_version": 1,
                "dataset": str(dataset_name),
                "rank": int(self.global_rank),
                "world_size": int(self.world_size),
                "record_count": int(prediction_count),
                "prediction_file": prediction_path.name,
                "complete": True,
            }
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")

        return responses

    def allreduce_response_metrics(
        self,
        responses: list[SimpleNamespace],
        *,
        local_elapsed_seconds: float | None = None,
    ) -> dict[str, int | float | list[int] | None]:
        metric_summary: dict[str, int | float | list[int] | None] = {
            "sample_count": len(responses),
            "proposal_count": 0,
            "acceptance_length_sum": 0,
            "proposal_length_sum": 0,
            "accepted_draft_token_sum": 0,
            "output_token_count": sum(
                max(0, int(getattr(response, "num_output_tokens", 0)))
                for response in responses
            ),
            "logical_draft_block_count": sum(
                sum(
                    int(value)
                    for value in getattr(
                        response,
                        "logical_draft_block_counts",
                        [],
                    )
                )
                for response in responses
            ),
            "blind_accept_event_count": sum(
                int(getattr(response, "blind_accept_event_count", 0))
                for response in responses
            ),
            "blind_accepted_token_count": sum(
                int(getattr(response, "blind_accepted_token_count", 0))
                for response in responses
            ),
            "risk_budget_rescue_count": sum(
                int(getattr(response, "risk_budget_rescue_count", 0))
                for response in responses
            ),
            "risk_budget_reject_outside_top_p_count": sum(
                int(
                    getattr(
                        response,
                        "risk_budget_reject_outside_top_p_count",
                        0,
                    )
                )
                for response in responses
            ),
            "risk_budget_reject_budget_count": sum(
                int(getattr(response, "risk_budget_reject_budget_count", 0))
                for response in responses
            ),
            "proposals_at_pos": [0] * self.max_proposal_tokens,
            "accepted_at_pos": [0] * self.max_proposal_tokens,
        }
        proposals_at_pos = metric_summary["proposals_at_pos"]
        accepted_at_pos = metric_summary["accepted_at_pos"]
        assert isinstance(proposals_at_pos, list)
        assert isinstance(accepted_at_pos, list)

        for response in responses:
            acceptance_lengths = getattr(response, "acceptance_lengths", None)
            proposal_lengths = getattr(response, "proposal_lengths", None)
            accepted_draft_lengths = getattr(response, "accepted_draft_lengths", None)
            if (
                acceptance_lengths is None
                or proposal_lengths is None
                or accepted_draft_lengths is None
            ):
                continue
            assert (
                len(acceptance_lengths)
                == len(proposal_lengths)
                == len(accepted_draft_lengths)
            ), (
                "acceptance_lengths, proposal_lengths and accepted_draft_lengths "
                "must have the same length."
            )
            for acceptance_length, proposal_length, accepted_draft_length in zip(
                acceptance_lengths,
                proposal_lengths,
                accepted_draft_lengths,
            ):
                accepted_draft_length = int(accepted_draft_length)
                metric_summary["proposal_count"] += 1
                metric_summary["acceptance_length_sum"] += int(acceptance_length)
                metric_summary["proposal_length_sum"] += int(proposal_length)
                metric_summary["accepted_draft_token_sum"] += (
                    accepted_draft_length
                )
                for pos_idx in range(self.max_proposal_tokens):
                    if proposal_length > pos_idx:
                        proposals_at_pos[pos_idx] += 1
                    if accepted_draft_length > pos_idx:
                        accepted_at_pos[pos_idx] += 1

        scalar_tensor = torch.tensor(
            [
                int(metric_summary["sample_count"]),
                int(metric_summary["proposal_count"]),
                int(metric_summary["acceptance_length_sum"]),
                int(metric_summary["proposal_length_sum"]),
                int(metric_summary["output_token_count"]),
                int(metric_summary["logical_draft_block_count"]),
                int(metric_summary["blind_accept_event_count"]),
                int(metric_summary["blind_accepted_token_count"]),
                int(metric_summary["risk_budget_rescue_count"]),
                int(metric_summary["risk_budget_reject_outside_top_p_count"]),
                int(metric_summary["risk_budget_reject_budget_count"]),
                int(metric_summary["accepted_draft_token_sum"]),
            ],
            device=self.device,
            dtype=torch.int64,
        )
        dist.all_reduce(scalar_tensor, op=dist.ReduceOp.SUM)

        position_tensor = torch.tensor(
            proposals_at_pos + accepted_at_pos,
            device=self.device,
            dtype=torch.int64,
        )
        if position_tensor.numel() > 0:
            dist.all_reduce(position_tensor, op=dist.ReduceOp.SUM)
        elapsed_seconds = None
        if local_elapsed_seconds is not None:
            elapsed_tensor = torch.tensor(
                [float(local_elapsed_seconds)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            elapsed_seconds = float(elapsed_tensor[0].item())
        return {
            "sample_count": int(scalar_tensor[0].item()),
            "proposal_count": int(scalar_tensor[1].item()),
            "acceptance_length_sum": int(scalar_tensor[2].item()),
            "proposal_length_sum": int(scalar_tensor[3].item()),
            "output_token_count": int(scalar_tensor[4].item()),
            "logical_draft_block_count": int(scalar_tensor[5].item()),
            "blind_accept_event_count": int(scalar_tensor[6].item()),
            "blind_accepted_token_count": int(scalar_tensor[7].item()),
            "risk_budget_rescue_count": int(scalar_tensor[8].item()),
            "risk_budget_reject_outside_top_p_count": int(
                scalar_tensor[9].item()
            ),
            "risk_budget_reject_budget_count": int(scalar_tensor[10].item()),
            "accepted_draft_token_sum": int(scalar_tensor[11].item()),
            "elapsed_seconds": elapsed_seconds,
            "proposals_at_pos": position_tensor[
                : self.max_proposal_tokens
            ].tolist(),
            "accepted_at_pos": position_tensor[
                self.max_proposal_tokens :
            ].tolist(),
        }

    def log_tensorboard(self) -> None:
        from torch.utils.tensorboard import SummaryWriter

        assert self.args.tensorboard_dir is not None
        assert self.args.step is not None
        Path(self.args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=self.args.tensorboard_dir)
        for metrics in self.metrics_rows:
            dataset_name = metrics["dataset"]
            writer.add_scalar(
                f"eval/{dataset_name}/draft_tokens_per_proposal",
                float(metrics["draft_tokens_per_proposal"]),
                global_step=self.args.step,
            )
            writer.add_scalar(
                f"eval/{dataset_name}/accept_length",
                float(metrics["acceptance_length"]),
                global_step=self.args.step,
            )
            writer.add_scalar(
                f"eval/{dataset_name}/verify_rate",
                float(metrics["verify_rate"]),
                global_step=self.args.step,
            )
            if int(metrics.get("blind_accept_event_count", 0)) > 0:
                writer.add_scalar(
                    f"eval/{dataset_name}/blind_accept_blocks",
                    int(metrics["blind_accept_event_count"]),
                    global_step=self.args.step,
                )
            if any(
                int(metrics.get(key, 0)) > 0
                for key in (
                    "risk_budget_rescue_count",
                    "risk_budget_reject_outside_top_p_count",
                    "risk_budget_reject_budget_count",
                )
            ):
                writer.add_scalar(
                    f"eval/{dataset_name}/risk_budget_rescues",
                    int(metrics["risk_budget_rescue_count"]),
                    global_step=self.args.step,
                )
                writer.add_scalar(
                    f"eval/{dataset_name}/risk_budget_reject_outside_top_p",
                    int(metrics["risk_budget_reject_outside_top_p_count"]),
                    global_step=self.args.step,
                )
                writer.add_scalar(
                    f"eval/{dataset_name}/risk_budget_reject_budget",
                    int(metrics["risk_budget_reject_budget_count"]),
                    global_step=self.args.step,
                )
                writer.add_scalar(
                    f"eval/{dataset_name}/risk_budget_rescue_acceptance_rate",
                    float(metrics["risk_budget_rescue_acceptance_rate"]),
                    global_step=self.args.step,
                )
                writer.add_scalar(
                    f"eval/{dataset_name}/blind_accepted_tokens",
                    int(metrics["blind_accepted_token_count"]),
                    global_step=self.args.step,
                )
            throughput = metrics.get("throughput_tokens_per_second")
            if throughput is not None:
                writer.add_scalar(
                    f"eval/{dataset_name}/throughput_output_tokens_per_second",
                    float(throughput),
                    global_step=self.args.step,
                )
                writer.add_scalar(
                    f"eval/{dataset_name}/throughput_samples_per_second",
                    float(metrics["throughput_samples_per_second"]),
                    global_step=self.args.step,
                )
            for pos_idx, accept_rate in enumerate(metrics["accept_rates_by_position"]):
                if accept_rate is None:
                    continue
                writer.add_scalar(
                    f"eval/{dataset_name}/accept_rate@{pos_idx}",
                    float(accept_rate),
                    global_step=self.args.step,
                )
        writer.close()

    def print_results(self) -> None:
        if dist.get_rank() != 0:
            return
        print(
            build_results_table(
                rows=self.metrics_rows,
                model_name_or_path=self.args.target_name_or_path,
                draft_name_or_path=self.args.draft_name_or_path,
            ),
            flush=True,
        )

    def print_dataset_result(self, metrics_row: dict[str, object]) -> None:
        if dist.get_rank() != 0:
            return
        print(
            build_results_table(
                rows=[metrics_row],
                model_name_or_path=self.args.target_name_or_path,
                draft_name_or_path=self.args.draft_name_or_path,
                header=False,
            ),
            flush=True,
        )

    def record_dataset_metrics(
        self,
        *,
        dataset_name: str,
        metric_summary: dict[str, int | float | list[int] | None],
    ) -> dict[str, object] | None:
        if dist.get_rank() != 0 or int(metric_summary["sample_count"]) <= 0:
            return None
        metrics_row = self.build_metrics_row(
            dataset_name=dataset_name,
            metric_summary=metric_summary,
        )
        self.metrics_rows.append(metrics_row)
        self.print_dataset_result(metrics_row)
        return metrics_row

    def write_throughput_metrics(self) -> None:
        if (
            dist.get_rank() != 0
            or not bool(getattr(self.args, "record_throughput", False))
        ):
            return
        timed_rows = [
            dict(row)
            for row in self.metrics_rows
            if row.get("elapsed_seconds") is not None
        ]
        total_output_tokens = sum(
            int(row["output_token_count"]) for row in timed_rows
        )
        total_samples = sum(int(row["num_samples"]) for row in timed_rows)
        total_elapsed_seconds = sum(
            float(row["elapsed_seconds"]) for row in timed_rows
        )
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "timing_scope": (
                "Per-dataset inference loop including prompt encoding and "
                "prediction dumping; excludes model loading and post-hoc scoring."
            ),
            "throughput_definition": (
                "sum of generated output tokens divided by the slowest-rank "
                "wall-clock inference time"
            ),
            "target_model": str(self.args.target_name_or_path),
            "draft_model": str(self.args.draft_name_or_path),
            "seed": int(self.args.seed),
            "decoding_config": {
                **(
                    {"autoregressive_top_k": getattr(self.args, "autoregressive_top_k", None)}
                    if getattr(self.args, "acceptance_rule", None) == "autoregressive"
                    else {}
                ),
                "acceptance_rule": str(
                    getattr(
                        self.args,
                        "acceptance_rule",
                        ACCEPTANCE_REJECTION_SAMPLING,
                    )
                ),
                "acceptance_top_k": int(
                    getattr(self.args, "acceptance_top_k", 2)
                ),
                "acceptance_top_p": float(
                    getattr(self.args, "acceptance_top_p", 0.95)
                ),
                "dynamic_top_p_blend": float(
                    getattr(self.args, "dynamic_top_p_blend", 0.0)
                ),
                "dynamic_top_p_support": float(
                    getattr(self.args, "dynamic_top_p_support", 0.95)
                ),
                "dynamic_top_p_outer_alpha": float(
                    getattr(self.args, "dynamic_top_p_outer_alpha", 0.0)
                ),
                "dynamic_top_p_signal": str(
                    getattr(
                        self.args,
                        "dynamic_top_p_signal",
                        DYNAMIC_TOP_P_SIGNAL_ENTROPY_DISAGREEMENT,
                    )
                ),
                "risk_budget_max": float(
                    getattr(
                        self.args,
                        "risk_budget_max",
                        DEFAULT_RISK_BUDGET_MAX,
                    )
                ),
                "risk_budget_safe_top_p": float(
                    getattr(
                        self.args,
                        "risk_budget_safe_top_p",
                        DEFAULT_RISK_BUDGET_SAFE_TOP_P,
                    )
                ),
                "cactus_delta": float(
                    getattr(self.args, "cactus_delta", 1.0)
                ),
                "mars_relaxation_threshold": float(
                    getattr(
                        self.args,
                        "mars_relaxation_threshold",
                        DEFAULT_MARS_RELAXATION_THRESHOLD,
                    )
                ),
                "target_sampling_top_p": float(
                    getattr(self.args, "target_sampling_top_p", 1.0)
                ),
                "temperature": float(getattr(self.args, "temperature", 1.0)),
                "record_verification_timing": bool(
                    getattr(self.args, "record_verification_timing", False)
                ),
                "verification_timing_warmup_samples": int(
                    getattr(self.args, "verification_timing_warmup_samples", 1)
                ),
                "asd_budget": float(getattr(self.args, "asd_budget", 8.0)),
                "asd_gate": float(getattr(self.args, "asd_gate", 0.25)),
                "asd_max_exceptions": int(getattr(self.args, "asd_max_exceptions", 2)),
                "asd_value_scheme": str(getattr(self.args, "asd_value_scheme", "suffix_length")),
            },
            "datasets": {
                str(row["dataset"]): row
                for row in timed_rows
            },
            "aggregate": {
                "num_samples": total_samples,
                "output_token_count": total_output_tokens,
                "elapsed_seconds": total_elapsed_seconds,
                "throughput_tokens_per_second": (
                    total_output_tokens / total_elapsed_seconds
                    if total_elapsed_seconds > 0.0
                    else None
                ),
                "throughput_samples_per_second": (
                    total_samples / total_elapsed_seconds
                    if total_elapsed_seconds > 0.0
                    else None
                ),
            },
        }
        output_path = Path(self.args.tensorboard_dir) / "throughput_metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        inprogress_path = output_path.with_suffix(output_path.suffix + ".inprogress")
        with inprogress_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(inprogress_path, output_path)

    def start_throughput_timer(self) -> float | None:
        if not bool(getattr(self.args, "record_throughput", False)):
            return None
        dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def stop_throughput_timer(
        self,
        inference_start: float | None,
    ) -> float | None:
        if inference_start is None:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - inference_start

    def report_results(self) -> None:
        if dist.get_rank() == 0 and self.metrics_rows:
            if self.args.tensorboard_dir is not None:
                self.log_tensorboard()
            self.write_throughput_metrics()
        self.print_results()

    def evaluate(self) -> None:
        for dataset_name, max_samples in self.tasks:
            self.start_dataset_diagnostics(dataset_name)
            inference_start = self.start_throughput_timer()
            responses = self.run_dataset(
                dataset_name=dataset_name,
                max_samples=max_samples,
            )
            local_elapsed_seconds = self.stop_throughput_timer(inference_start)
            metric_summary = self.allreduce_response_metrics(
                responses,
                local_elapsed_seconds=local_elapsed_seconds,
            )
            self.finish_dataset_diagnostics()
            self.record_dataset_metrics(
                dataset_name=dataset_name,
                metric_summary=metric_summary,
            )

        self.report_results()

    def clean_up(self) -> None:
        dist.destroy_process_group()
