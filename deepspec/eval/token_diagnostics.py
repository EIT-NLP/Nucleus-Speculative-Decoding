from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, TextIO

import torch
import torch.distributed as dist


TOKEN_DIAGNOSTICS_MODES = {"all", "rejected", "accepted"}
NSD_TARGET_TOP_P_ACCEPTANCE_RULE = "rejection_sampling_then_target_top_p"


def _step_artifact_name(step: int | None) -> str:
    if step is None:
        return "step_unknown"
    return f"step_{int(step)}"


def _tensor_entropy(probs: torch.Tensor) -> float:
    probs = probs.float().clamp_min(1e-20)
    return float(-(probs * probs.log()).sum().item())


def _total_variation_distance(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> float:
    """Return the full-vocabulary target/draft total variation distance."""
    if target_probs.shape != draft_probs.shape:
        raise ValueError(
            "target and draft probabilities must have the same shape, "
            f"got {tuple(target_probs.shape)} and {tuple(draft_probs.shape)}."
        )
    distance = 0.5 * (target_probs.float() - draft_probs.float()).abs().sum()
    return float(distance.clamp(0.0, 1.0).item())


def _target_nucleus_mask(
    target_probs: torch.Tensor,
    target_logits: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """Return the full-vocabulary nucleus used by target-top-p acceptance.

    Membership matches the verifier's strict-higher-logit definition, including
    all exactly tied tokens when the mass of strictly higher logits is at most
    ``top_p``.
    """
    if target_probs.ndim != 2 or target_logits.ndim != 2:
        raise ValueError("target nucleus inputs must have shape [positions, vocab]")
    if target_probs.shape != target_logits.shape:
        raise ValueError(
            "target probabilities and logits must have matching shapes, got "
            f"{tuple(target_probs.shape)} and {tuple(target_logits.shape)}"
        )

    sorted_logits, sorted_indices = torch.sort(
        target_logits.float(),
        dim=-1,
        descending=True,
    )
    sorted_probs = target_probs.float().gather(-1, sorted_indices)
    cumulative_before = sorted_probs.cumsum(dim=-1) - sorted_probs

    group_start = torch.ones_like(sorted_logits, dtype=torch.bool)
    group_start[:, 1:] = sorted_logits[:, 1:] != sorted_logits[:, :-1]
    positions = torch.arange(
        sorted_logits.shape[-1],
        device=sorted_logits.device,
        dtype=torch.long,
    ).unsqueeze(0).expand_as(sorted_indices)
    group_start_positions = torch.where(group_start, positions, 0)
    group_start_positions = torch.cummax(group_start_positions, dim=-1).values
    strictly_higher_mass = cumulative_before.gather(-1, group_start_positions)
    keep_sorted = strictly_higher_mass <= float(top_p)

    nucleus_mask = torch.zeros_like(keep_sorted)
    return nucleus_mask.scatter(-1, sorted_indices, keep_sorted)


def _exact_nsd_nucleus_metrics(
    *,
    target_probs: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    top_p: float,
) -> dict[str, torch.Tensor]:
    """Compute exact one-step NSD versus target-nucleus agreement metrics."""
    if target_probs.shape != draft_probs.shape:
        raise ValueError(
            "target and draft probabilities must have matching shapes, got "
            f"{tuple(target_probs.shape)} and {tuple(draft_probs.shape)}"
        )

    target = target_probs.float()
    draft = draft_probs.float()
    nucleus_mask = _target_nucleus_mask(target, target_logits, top_p)
    nucleus_float = nucleus_mask.to(target.dtype)
    nucleus_mass = (target * nucleus_float).sum(dim=-1)
    nucleus_size = nucleus_mask.sum(dim=-1)
    target_nucleus = target * nucleus_float
    target_nucleus = target_nucleus / nucleus_mass.unsqueeze(-1).clamp_min(1e-8)

    standard_accept_probability = torch.clamp(
        target / draft.clamp_min(1e-8),
        max=1.0,
    )
    nsd_accept_probability = torch.where(
        nucleus_mask,
        torch.ones_like(standard_accept_probability),
        standard_accept_probability,
    )
    committed_proposal_mass = draft * nsd_accept_probability
    rejection_mass = (
        1.0 - committed_proposal_mass.sum(dim=-1)
    ).clamp(min=0.0, max=1.0)

    residual = (target - draft).clamp_min(0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = torch.where(residual_mass <= 1e-8, target, residual)
    residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    induced_nsd = committed_proposal_mass + rejection_mass.unsqueeze(-1) * residual
    induced_nsd = induced_nsd / induced_nsd.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)

    nsd_to_target_tv = 0.5 * (induced_nsd - target).abs().sum(dim=-1)
    nsd_to_nucleus_tv = 0.5 * (
        induced_nsd - target_nucleus
    ).abs().sum(dim=-1)
    rescue_mass = (
        (draft - target).clamp_min(0.0) * nucleus_float
    ).sum(dim=-1)

    return {
        "target_nucleus_mass": nucleus_mass.clamp(0.0, 1.0),
        "target_nucleus_size": nucleus_size,
        "rescue_mass": rescue_mass.clamp(0.0, 1.0),
        "rejection_mass": rejection_mass,
        "nsd_to_target_tv": nsd_to_target_tv.clamp(0.0, 1.0),
        "nsd_to_nucleus_tv": nsd_to_nucleus_tv.clamp(0.0, 1.0),
        "rho": (1.0 - nsd_to_nucleus_tv).clamp(0.0, 1.0),
    }


def _target_rank_and_cumulative_prob(
    target_probs: torch.Tensor,
    target_logits: torch.Tensor,
    token_id: int,
) -> tuple[int, float, float]:
    token_logit = target_logits[token_id]
    higher_prob_mask = target_logits > token_logit
    target_rank = int(higher_prob_mask.sum().item()) + 1
    cumulative_before = min(
        max(float(target_probs[higher_prob_mask].sum().item()), 0.0),
        1.0,
    )
    token_prob = float(target_probs[token_id].item())
    cumulative_at_rank = min(cumulative_before + token_prob, 1.0)
    return target_rank, cumulative_before, cumulative_at_rank


def _mask_value(mask: torch.Tensor | None, position: int) -> bool | None:
    if mask is None:
        return None
    return bool(mask[0, position].item())


def _tensor_float_value(
    tensor: torch.Tensor | None,
    position: int,
) -> float | None:
    if tensor is None:
        return None
    return float(tensor[0, position].float().item())


def _tensor_int_value(
    tensor: torch.Tensor | None,
    position: int,
) -> int | None:
    if tensor is None:
        return None
    return int(tensor[0, position].item())


def _sigmoid_float(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class TokenDiagnosticsRecorder:
    """Write per-draft-token probability and entropy diagnostics.

    The recorder is intentionally rank-local.  Large runs can keep one JSONL
    shard per distributed rank without gathering all token-level rows on rank 0.
    """

    def __init__(
        self,
        *,
        tokenizer,
        tensorboard_dir: str,
        step: int | None,
        rank: int,
        world_size: int,
        mode: str,
        record_exact_nsd_rho: bool = False,
        acceptance_rule: str,
        acceptance_top_k: int,
        acceptance_top_p: float,
        dynamic_top_p_blend: float = 0.0,
        dynamic_top_p_support: float = 0.95,
        dynamic_top_p_outer_alpha: float = 0.0,
        dynamic_top_p_signal: str = "entropy_disagreement",
        risk_budget_max: float = 1.0,
        risk_budget_safe_top_p: float = 0.90,
        cactus_delta: float = 1.0,
        mars_relaxation_threshold: float = 0.90,
        target_sampling_top_p: float = 1.0,
    ) -> None:
        mode = str(mode)
        if mode not in TOKEN_DIAGNOSTICS_MODES:
            raise ValueError(
                f"token diagnostics mode must be one of "
                f"{sorted(TOKEN_DIAGNOSTICS_MODES)}, got {mode!r}."
            )
        self.tokenizer = tokenizer
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.mode = mode
        self.record_exact_nsd_rho = bool(record_exact_nsd_rho)
        self.acceptance_rule = str(acceptance_rule)
        self.acceptance_top_k = int(acceptance_top_k)
        self.acceptance_top_p = float(acceptance_top_p)
        self.dynamic_top_p_blend = float(dynamic_top_p_blend)
        self.dynamic_top_p_support = float(dynamic_top_p_support)
        self.dynamic_top_p_outer_alpha = float(dynamic_top_p_outer_alpha)
        self.dynamic_top_p_signal = str(dynamic_top_p_signal)
        self.risk_budget_max = float(risk_budget_max)
        self.risk_budget_safe_top_p = float(risk_budget_safe_top_p)
        self.cactus_delta = float(cactus_delta)
        self.mars_relaxation_threshold = float(mars_relaxation_threshold)
        self.target_sampling_top_p = float(target_sampling_top_p)
        self.artifact_root = (
            Path(tensorboard_dir) / "artifacts" / _step_artifact_name(step)
        )
        self.dataset_name: str | None = None
        self.dataset_dir: Path | None = None
        self.output_path: Path | None = None
        self.handle: TextIO | None = None
        self.record_count = 0

    def start(self, dataset_name: str) -> None:
        if self.handle is not None:
            raise RuntimeError("TokenDiagnosticsRecorder.start called twice.")
        self.dataset_name = str(dataset_name)
        self.dataset_dir = self.artifact_root / self.dataset_name
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.dataset_dir / f"token_diagnostics.rank{self.rank:05d}.jsonl"
        self.handle = self.output_path.open("w", encoding="utf-8")
        self.record_count = 0

    def _should_record(
        self,
        *,
        proposal_position: int,
        accepted_draft_tokens: int,
        rejected_position: int | None,
    ) -> bool:
        if self.mode == "all":
            return True
        if self.mode == "accepted":
            return proposal_position < accepted_draft_tokens
        return rejected_position is not None and proposal_position == rejected_position

    def observe(
        self,
        *,
        proposal: Any,
        verification: Any,
        start: int,
        proposal_index: int,
        sample_index: int | None,
    ) -> None:
        if self.handle is None:
            return
        effective_length = int(verification.effective_proposal_length)
        if effective_length <= 0:
            return
        if proposal.draft_probs is None:
            return

        draft_token_count = min(int(proposal.draft_token_count), effective_length)
        accepted_draft_tokens = int(verification.accepted_draft_tokens)
        rejected_position = (
            accepted_draft_tokens
            if accepted_draft_tokens < effective_length
            else None
        )
        vocab_size = int(verification.target_probs.shape[-1])
        log_vocab_size = math.log(vocab_size) if vocab_size > 1 else 1.0
        confidence_logits = getattr(proposal, "confidence_logits", None)
        reached_position_count = (
            draft_token_count
            if rejected_position is None
            else min(rejected_position + 1, draft_token_count)
        )
        exact_nsd_metrics: dict[str, torch.Tensor] | None = None
        if (
            self.record_exact_nsd_rho
            and self.acceptance_rule in {NSD_TARGET_TOP_P_ACCEPTANCE_RULE, "cleanNSD"}
            and reached_position_count > 0
        ):
            exact_nsd_metrics = _exact_nsd_nucleus_metrics(
                target_probs=verification.target_probs[
                    0, :reached_position_count, :
                ],
                target_logits=verification.target_output.logits[
                    0, :reached_position_count, :
                ],
                draft_probs=proposal.draft_probs[
                    0, :reached_position_count, :
                ],
                top_p=self.acceptance_top_p,
            )

        for proposal_position in range(draft_token_count):
            if not self._should_record(
                proposal_position=proposal_position,
                accepted_draft_tokens=accepted_draft_tokens,
                rejected_position=rejected_position,
            ):
                continue

            token_id = int(
                proposal.verify_input_ids[0, proposal_position + 1].item()
            )
            target_probs = verification.target_probs[0, proposal_position].float()
            target_logits = verification.target_output.logits[
                0,
                proposal_position,
            ].float()
            (
                target_rank,
                target_cumulative_prob_before,
                target_cumulative_prob_at_rank,
            ) = _target_rank_and_cumulative_prob(
                target_probs,
                target_logits,
                token_id,
            )
            target_verify_prob = float(target_probs[token_id].item())
            draft_probs = proposal.draft_probs[0, proposal_position].float()
            draft_token_prob = float(draft_probs[token_id].item())
            target_entropy = _tensor_entropy(target_probs)
            draft_entropy = _tensor_entropy(draft_probs)
            draft_target_tv_distance = _total_variation_distance(
                target_probs,
                draft_probs,
            )
            nsd_target_nucleus_mass = None
            nsd_target_nucleus_size = None
            nsd_nucleus_rescue_mass = None
            nsd_rejection_mass = None
            nsd_to_target_tv_distance = None
            nsd_to_target_nucleus_tv_distance = None
            nsd_nucleus_rho = None
            if (
                exact_nsd_metrics is not None
                and proposal_position < reached_position_count
            ):
                nsd_target_nucleus_mass = float(
                    exact_nsd_metrics["target_nucleus_mass"][
                        proposal_position
                    ].item()
                )
                nsd_target_nucleus_size = int(
                    exact_nsd_metrics["target_nucleus_size"][
                        proposal_position
                    ].item()
                )
                nsd_nucleus_rescue_mass = float(
                    exact_nsd_metrics["rescue_mass"][proposal_position].item()
                )
                nsd_rejection_mass = float(
                    exact_nsd_metrics["rejection_mass"][proposal_position].item()
                )
                nsd_to_target_tv_distance = float(
                    exact_nsd_metrics["nsd_to_target_tv"][
                        proposal_position
                    ].item()
                )
                nsd_to_target_nucleus_tv_distance = float(
                    exact_nsd_metrics["nsd_to_nucleus_tv"][
                        proposal_position
                    ].item()
                )
                nsd_nucleus_rho = float(
                    exact_nsd_metrics["rho"][proposal_position].item()
                )
            standard_accept_probability = min(
                target_verify_prob / max(draft_token_prob, 1e-8),
                1.0,
            )
            target_to_draft_probability_ratio = (
                target_verify_prob / max(draft_token_prob, 1e-8)
            )
            cactus_adjusted_target_probability = _tensor_float_value(
                getattr(verification, "cactus_adjusted_selected_probs", None),
                proposal_position,
            )
            cactus_accept_probability = _tensor_float_value(
                getattr(verification, "cactus_accept_prob", None),
                proposal_position,
            )
            standard_uniform_random = _tensor_float_value(
                getattr(verification, "standard_uniform_random", None),
                proposal_position,
            )
            standard_accept = _mask_value(
                getattr(verification, "standard_accept_mask", None),
                proposal_position,
            )
            mars_logit_ratio = _tensor_float_value(
                getattr(verification, "mars_top2_logit_ratio", None),
                proposal_position,
            )
            mars_relaxation = _mask_value(
                getattr(verification, "mars_relaxation_mask", None),
                proposal_position,
            )
            mars_inside_target_top2 = _mask_value(
                getattr(verification, "mars_inside_target_top2", None),
                proposal_position,
            )
            mars_relaxed_accept = _mask_value(
                getattr(verification, "mars_relaxed_accept_mask", None),
                proposal_position,
            )
            mars_relaxed_reject = _mask_value(
                getattr(verification, "mars_relaxed_reject_mask", None),
                proposal_position,
            )
            target_top_p_accept = _mask_value(
                getattr(verification, "target_top_p_accept_mask", None),
                proposal_position,
            )
            asd = getattr(verification, "asd_decision", None)
            asd_mismatch = _mask_value(getattr(asd, "mismatches", None), proposal_position)
            asd_regret = _tensor_float_value(getattr(asd, "regrets", None), proposal_position)
            asd_value = _tensor_float_value(getattr(asd, "values", None), proposal_position)
            if (
                self.acceptance_rule in {"target_top_p", "onlytopp"}
                and target_top_p_accept is not None
            ):
                effective_accept_probability = float(target_top_p_accept)
            elif self.acceptance_rule == "asd":
                effective_accept_probability = float(bool(_mask_value(
                    getattr(verification, "accept_mask", None), proposal_position)))
            elif cactus_accept_probability is not None:
                effective_accept_probability = cactus_accept_probability
            elif mars_relaxation is True:
                effective_accept_probability = float(bool(mars_inside_target_top2))
            else:
                effective_accept_probability = standard_accept_probability
            inside_initial_top_p = _mask_value(
                getattr(verification, "dynamic_top_p_initial_accept_mask", None),
                proposal_position,
            )
            dynamic_top_p_accept = _mask_value(
                getattr(verification, "dynamic_top_p_accept_mask", None),
                proposal_position,
            )
            dynamic_top_p_rescue = _mask_value(
                getattr(verification, "dynamic_top_p_rescue_mask", None),
                proposal_position,
            )
            risk_budget_rescue = _mask_value(
                getattr(verification, "risk_budget_rescue_mask", None),
                proposal_position,
            )
            risk_budget_rescue_eligible = _mask_value(
                getattr(
                    verification,
                    "risk_budget_rescue_eligibility_mask",
                    None,
                ),
                proposal_position,
            )
            risk_budget_reject_outside_top_p = _mask_value(
                getattr(
                    verification,
                    "risk_budget_reject_outside_top_p_mask",
                    None,
                ),
                proposal_position,
            )
            risk_budget_reject_budget = _mask_value(
                getattr(verification, "risk_budget_reject_budget_mask", None),
                proposal_position,
            )

            confidence_logit = None
            confidence_score = None
            if confidence_logits is not None:
                confidence_logit = float(
                    confidence_logits[0, proposal_position].float().item()
                )
                confidence_score = _sigmoid_float(confidence_logit)

            actual_rejected = (
                rejected_position is not None
                and proposal_position == rejected_position
            )
            actually_accepted = proposal_position < accepted_draft_tokens
            blind_accept_prefix_length = int(
                getattr(verification, "blind_accept_prefix_length", 0)
            )
            draft_block_size = int(getattr(proposal, "draft_block_size", 0))
            if actual_rejected and risk_budget_reject_outside_top_p:
                final_decision = "REJECT_OUTSIDE_TOP_P"
            elif actual_rejected and risk_budget_reject_budget:
                final_decision = "REJECT_RISK_BUDGET"
            elif actual_rejected and mars_relaxed_reject:
                final_decision = "MARS_RELAX_REJECT"
            elif actual_rejected:
                final_decision = "REJECT"
            elif actually_accepted and risk_budget_rescue:
                final_decision = "RESCUE_ACCEPT"
            elif actually_accepted and dynamic_top_p_rescue:
                final_decision = "RESCUE_ACCEPT"
            elif actually_accepted and mars_relaxed_accept:
                final_decision = "MARS_RELAX_ACCEPT"
            elif actually_accepted and self.acceptance_rule == "asd":
                final_decision = "ASD_EXCEPTION_ACCEPT" if asd_mismatch else "ASD_GREEDY_ACCEPT"
            elif actually_accepted and standard_accept:
                final_decision = "STANDARD_ACCEPT"
            elif actually_accepted:
                final_decision = "ACCEPT"
            else:
                final_decision = "NOT_REACHED"
            record = {
                "dataset": self.dataset_name,
                "sample_index": int(sample_index) if sample_index is not None else None,
                "rank": self.rank,
                "world_size": self.world_size,
                "proposal_index": int(proposal_index),
                "proposal_position": int(proposal_position),
                "block_id": int(proposal_index),
                "position_in_block": int(proposal_position),
                "generated_position": int(start) + proposal_position + 1,
                "draft_token_id": token_id,
                "asd_mismatch": asd_mismatch,
                "asd_regret": asd_regret,
                "asd_suffix_value": asd_value,
                "asd_request_ledger_after_block": getattr(verification, "asd_statistics", None),
                "draft_token_text": self.tokenizer.decode(
                    [token_id],
                    clean_up_tokenization_spaces=False,
                ),
                "draft_token_prob": draft_token_prob,
                "draft_probability": draft_token_prob,
                "target_verify_prob": target_verify_prob,
                "target_probability": target_verify_prob,
                "target_to_draft_probability_ratio": (
                    target_to_draft_probability_ratio
                ),
                "target_rank": int(target_rank),
                "target_cumulative_prob_before": target_cumulative_prob_before,
                "target_cumulative_prob_at_rank": target_cumulative_prob_at_rank,
                "target_entropy": target_entropy,
                "draft_entropy": draft_entropy,
                "target_entropy_normalized": target_entropy / log_vocab_size,
                "draft_entropy_normalized": draft_entropy / log_vocab_size,
                "accept_probability": standard_accept_probability,
                "standard_accept_probability": standard_accept_probability,
                "uniform_random_value": standard_uniform_random,
                "standard_decision": standard_accept,
                "cactus_adjusted_target_probability": (
                    cactus_adjusted_target_probability
                ),
                "cactus_accept_probability": cactus_accept_probability,
                "mars_top2_logit_ratio": mars_logit_ratio,
                "mars_relaxation": mars_relaxation,
                "mars_inside_target_top2": mars_inside_target_top2,
                "mars_relaxed_accept": mars_relaxed_accept,
                "mars_relaxed_reject": mars_relaxed_reject,
                "effective_accept_probability": effective_accept_probability,
                "standard_accept": standard_accept,
                "standard_rejected": (
                    None if standard_accept is None else not standard_accept
                ),
                "target_top_p_accept": target_top_p_accept,
                "target_top_k_top_p_accept": _mask_value(
                    getattr(verification, "target_top_k_top_p_accept_mask", None),
                    proposal_position,
                ),
                "inside_initial_top_p": inside_initial_top_p,
                "initial_nucleus_size": _tensor_int_value(
                    getattr(
                        verification,
                        "dynamic_top_p_initial_nucleus_size",
                        None,
                    ),
                    proposal_position,
                ),
                "dynamic_target_entropy": _tensor_float_value(
                    getattr(verification, "dynamic_top_p_entropy", None),
                    proposal_position,
                ),
                "draft_target_tv_distance": draft_target_tv_distance,
                "nsd_target_nucleus_mass": nsd_target_nucleus_mass,
                "nsd_target_nucleus_size": nsd_target_nucleus_size,
                "nsd_nucleus_rescue_mass": nsd_nucleus_rescue_mass,
                "nsd_rejection_mass": nsd_rejection_mass,
                "nsd_to_target_tv_distance": nsd_to_target_tv_distance,
                "nsd_to_target_nucleus_tv_distance": (
                    nsd_to_target_nucleus_tv_distance
                ),
                "nsd_nucleus_rho": nsd_nucleus_rho,
                "dynamic_top_p": _tensor_float_value(
                    getattr(verification, "dynamic_top_p_threshold", None),
                    proposal_position,
                ),
                "dynamic_nucleus_size": _tensor_int_value(
                    getattr(verification, "dynamic_top_p_nucleus_size", None),
                    proposal_position,
                ),
                "inside_dynamic_top_p": dynamic_top_p_accept,
                "dynamic_top_p_rescue": dynamic_top_p_rescue,
                "risk_budget_target_prefix_mass": _tensor_float_value(
                    getattr(
                        verification,
                        "risk_budget_target_prefix_mass",
                        None,
                    ),
                    proposal_position,
                ),
                "risk_budget_inside_target_top_p": _mask_value(
                    getattr(
                        verification,
                        "risk_budget_inside_target_top_p",
                        None,
                    ),
                    proposal_position,
                ),
                "risk_budget_token_risk": _tensor_float_value(
                    getattr(verification, "risk_budget_token_risk", None),
                    proposal_position,
                ),
                "risk_budget_block_risk_before": _tensor_float_value(
                    getattr(
                        verification,
                        "risk_budget_block_risk_before",
                        None,
                    ),
                    proposal_position,
                ),
                "risk_budget_candidate_block_risk": _tensor_float_value(
                    getattr(
                        verification,
                        "risk_budget_candidate_block_risk",
                        None,
                    ),
                    proposal_position,
                ),
                "risk_budget_block_risk_after": _tensor_float_value(
                    getattr(
                        verification,
                        "risk_budget_block_risk_after",
                        None,
                    ),
                    proposal_position,
                ),
                "risk_budget_rescue": risk_budget_rescue,
                "risk_budget_rescue_eligible": risk_budget_rescue_eligible,
                "risk_budget_reject_outside_top_p": (
                    risk_budget_reject_outside_top_p
                ),
                "risk_budget_reject_budget": risk_budget_reject_budget,
                "target_prefix_mass": _tensor_float_value(
                    getattr(
                        verification,
                        "risk_budget_target_prefix_mass",
                        None,
                    ),
                    proposal_position,
                ),
                "inside_target_top_p_095": _mask_value(
                    getattr(
                        verification,
                        "risk_budget_inside_target_top_p",
                        None,
                    ),
                    proposal_position,
                ),
                "token_risk": _tensor_float_value(
                    getattr(verification, "risk_budget_token_risk", None),
                    proposal_position,
                ),
                "block_risk_before": _tensor_float_value(
                    getattr(
                        verification,
                        "risk_budget_block_risk_before",
                        None,
                    ),
                    proposal_position,
                ),
                "candidate_block_risk": _tensor_float_value(
                    getattr(
                        verification,
                        "risk_budget_candidate_block_risk",
                        None,
                    ),
                    proposal_position,
                ),
                "risk_budget": self.risk_budget_max,
                "block_risk_after": _tensor_float_value(
                    getattr(
                        verification,
                        "risk_budget_block_risk_after",
                        None,
                    ),
                    proposal_position,
                ),
                "cactus_accept": _mask_value(
                    getattr(verification, "cactus_accept_mask", None),
                    proposal_position,
                ),
                "final_accept": _mask_value(
                    getattr(verification, "accept_mask", None),
                    proposal_position,
                ),
                "prefix_accept": _mask_value(
                    getattr(verification, "accept_prefix_mask", None),
                    proposal_position,
                ),
                "actually_accepted": actually_accepted,
                "actual_rejected": actual_rejected,
                "final_decision": final_decision,
                "replacement_token_id": (
                    int(verification.next_token[0].item())
                    if actual_rejected and verification.next_token is not None
                    else None
                ),
                "reached_by_verification": (
                    rejected_position is None or proposal_position <= rejected_position
                ),
                "forced_reject_applied": (
                    getattr(verification, "forced_reject_position", None)
                    == proposal_position
                ),
                "blind_accept_applied": (
                    proposal_position < blind_accept_prefix_length
                ),
                "logical_block_offset": (
                    proposal_position // draft_block_size
                    if draft_block_size > 0
                    else 0
                ),
                "logical_block_count": int(
                    getattr(verification, "logical_draft_block_count", 1)
                ),
                "acceptance_rule": self.acceptance_rule,
                "record_exact_nsd_rho": self.record_exact_nsd_rho,
                "acceptance_top_k": self.acceptance_top_k,
                "acceptance_top_p": self.acceptance_top_p,
                "dynamic_top_p_blend": self.dynamic_top_p_blend,
                "dynamic_top_p_support": self.dynamic_top_p_support,
                "dynamic_top_p_outer_alpha": self.dynamic_top_p_outer_alpha,
                "dynamic_top_p_signal": self.dynamic_top_p_signal,
                "risk_budget_max": self.risk_budget_max,
                "risk_budget_safe_top_p": self.risk_budget_safe_top_p,
                "cactus_delta": self.cactus_delta,
                "mars_relaxation_threshold": self.mars_relaxation_threshold,
                "target_sampling_top_p": self.target_sampling_top_p,
                "confidence_logit": confidence_logit,
                "confidence_score": confidence_score,
            }
            self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.record_count += 1

    def finish(self) -> Path | None:
        output_path = self.output_path
        record_count = self.record_count
        if self.handle is not None:
            self.handle.close()
        self.dataset_name = None
        self.dataset_dir = None
        self.output_path = None
        self.handle = None
        self.record_count = 0
        if dist.get_rank() == 0 and output_path is not None:
            pattern = output_path.with_name("token_diagnostics.rank*.jsonl")
            print(
                f"Wrote token diagnostics to {pattern} "
                f"({record_count} records on rank 0)",
                flush=True,
            )
        return output_path


__all__ = ["TOKEN_DIAGNOSTICS_MODES", "TokenDiagnosticsRecorder"]
