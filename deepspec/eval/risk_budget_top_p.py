from __future__ import annotations

import math
from dataclasses import dataclass

import torch


DEFAULT_RISK_BUDGET_SAFE_TOP_P = 0.90
DEFAULT_RISK_BUDGET_MAX = 1.0


def validate_risk_budget_max(value: float) -> float:
    risk_budget = float(value)
    if not math.isfinite(risk_budget) or risk_budget < 0.0:
        raise ValueError(
            "risk_budget_max must be finite and nonnegative, "
            f"got {value}."
        )
    return risk_budget


def validate_risk_budget_top_p_thresholds(
    *,
    safe_top_p: float,
    target_top_p: float,
) -> tuple[float, float]:
    safe_top_p = float(safe_top_p)
    target_top_p = float(target_top_p)
    if not math.isfinite(safe_top_p) or not 0.0 <= safe_top_p < 1.0:
        raise ValueError(
            "risk_budget_safe_top_p must be finite and in [0, 1), "
            f"got {safe_top_p}."
        )
    if not math.isfinite(target_top_p) or not 0.0 < target_top_p <= 1.0:
        raise ValueError(
            "target_top_p must be finite and in (0, 1], "
            f"got {target_top_p}."
        )
    if safe_top_p >= target_top_p:
        raise ValueError(
            "risk_budget_safe_top_p must be smaller than target_top_p, "
            f"got {safe_top_p} and {target_top_p}."
        )
    return safe_top_p, target_top_p


@dataclass(frozen=True)
class RiskBudgetTopPResult:
    """Per-position decisions and block-risk state for one draft block."""

    target_prefix_mass: torch.Tensor
    inside_target_top_p: torch.Tensor
    rescue_eligibility_mask: torch.Tensor
    token_risk: torch.Tensor
    block_risk_before: torch.Tensor
    candidate_block_risk: torch.Tensor
    block_risk_after: torch.Tensor
    rescue_mask: torch.Tensor
    reject_outside_top_p_mask: torch.Tensor
    reject_risk_budget_mask: torch.Tensor
    accept_mask: torch.Tensor


def _validate_inputs(
    *,
    target_probs: torch.Tensor,
    target_logits: torch.Tensor,
    proposed_tokens: torch.Tensor,
    standard_accept_mask: torch.Tensor,
) -> None:
    if target_probs.shape != target_logits.shape:
        raise ValueError(
            "target_probs and target_logits must have the same shape, "
            f"got {tuple(target_probs.shape)} and {tuple(target_logits.shape)}."
        )
    if target_probs.ndim < 2:
        raise ValueError("target probabilities must include a vocabulary dimension.")
    expected_shape = target_probs.shape[:-1]
    if proposed_tokens.shape != expected_shape:
        raise ValueError(
            "proposed_tokens shape must match probability leading dimensions, "
            f"got {tuple(proposed_tokens.shape)} and {tuple(expected_shape)}."
        )
    if standard_accept_mask.shape != expected_shape:
        raise ValueError(
            "standard_accept_mask shape must match probability leading dimensions, "
            f"got {tuple(standard_accept_mask.shape)} and {tuple(expected_shape)}."
        )


def compute_target_prefix_mass(
    *,
    target_probs: torch.Tensor,
    target_logits: torch.Tensor,
    proposed_tokens: torch.Tensor,
) -> torch.Tensor:
    """Return probability mass ranked before each proposal.

    Higher-logit tokens precede the proposal. Equal-logit tokens are ordered by
    ascending token ID, matching the specification's deterministic tie break.
    """

    if target_probs.shape != target_logits.shape:
        raise ValueError("target_probs and target_logits must have the same shape.")
    if proposed_tokens.shape != target_probs.shape[:-1]:
        raise ValueError(
            "proposed_tokens shape must match probability leading dimensions."
        )

    proposed_logits = target_logits.gather(
        dim=-1,
        index=proposed_tokens.unsqueeze(-1),
    ).squeeze(-1)
    higher_logit_mask = target_logits > proposed_logits.unsqueeze(-1)
    token_ids = torch.arange(
        target_probs.shape[-1],
        device=target_probs.device,
        dtype=proposed_tokens.dtype,
    )
    tied_earlier_id_mask = (
        target_logits == proposed_logits.unsqueeze(-1)
    ) & (token_ids < proposed_tokens.unsqueeze(-1))
    ranked_before_mask = higher_logit_mask | tied_earlier_id_mask
    return target_probs.masked_fill(~ranked_before_mask, 0.0).sum(dim=-1)


def compute_risk_budget_top_p(
    *,
    target_probs: torch.Tensor,
    target_logits: torch.Tensor,
    proposed_tokens: torch.Tensor,
    standard_accept_mask: torch.Tensor,
    risk_budget_max: float = DEFAULT_RISK_BUDGET_MAX,
    risk_budget_safe_top_p: float = DEFAULT_RISK_BUDGET_SAFE_TOP_P,
    target_top_p: float = 0.95,
) -> RiskBudgetTopPResult:
    """Apply sequential block-level risk accounting after standard rejection."""

    _validate_inputs(
        target_probs=target_probs,
        target_logits=target_logits,
        proposed_tokens=proposed_tokens,
        standard_accept_mask=standard_accept_mask,
    )
    target_prefix_mass = compute_target_prefix_mass(
        target_probs=target_probs.float(),
        target_logits=target_logits.float(),
        proposed_tokens=proposed_tokens,
    )
    inside_target_top_p = target_prefix_mass < target_top_p
    return apply_block_risk_budget(
        target_prefix_mass=target_prefix_mass,
        inside_target_top_p=inside_target_top_p,
        rescue_eligibility_mask=inside_target_top_p,
        standard_accept_mask=standard_accept_mask,
        risk_budget_max=risk_budget_max,
        risk_budget_safe_top_p=risk_budget_safe_top_p,
        risk_reference_top_p=target_top_p,
    )


def apply_block_risk_budget(
    *,
    target_prefix_mass: torch.Tensor,
    inside_target_top_p: torch.Tensor,
    rescue_eligibility_mask: torch.Tensor,
    standard_accept_mask: torch.Tensor,
    risk_budget_max: float = DEFAULT_RISK_BUDGET_MAX,
    risk_budget_safe_top_p: float = DEFAULT_RISK_BUDGET_SAFE_TOP_P,
    risk_reference_top_p: float = 0.95,
) -> RiskBudgetTopPResult:
    """Apply sequential risk accounting to an arbitrary rescue gate.

    `risk_reference_top_p` controls the risk scale independently of the rescue
    gate. This lets an adaptive gate extend beyond fixed Top-P=0.95 while
    charging those outer-shell tokens more than one unit of risk.
    """

    expected_shape = target_prefix_mass.shape
    for name, tensor in (
        ("inside_target_top_p", inside_target_top_p),
        ("rescue_eligibility_mask", rescue_eligibility_mask),
        ("standard_accept_mask", standard_accept_mask),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {tuple(expected_shape)}, "
                f"got {tuple(tensor.shape)}."
            )
    if target_prefix_mass.ndim < 1:
        raise ValueError("target_prefix_mass must include a block dimension.")

    risk_budget_max = validate_risk_budget_max(risk_budget_max)
    risk_budget_safe_top_p, risk_reference_top_p = (
        validate_risk_budget_top_p_thresholds(
            safe_top_p=risk_budget_safe_top_p,
            target_top_p=risk_reference_top_p,
        )
    )
    token_risk = (
        (target_prefix_mass - risk_budget_safe_top_p)
        / (risk_reference_top_p - risk_budget_safe_top_p)
    ).clamp_min(0.0)

    standard_accept_mask = standard_accept_mask.bool()
    inside_target_top_p = inside_target_top_p.bool()
    rescue_eligibility_mask = rescue_eligibility_mask.bool()
    leading_shape = standard_accept_mask.shape
    block_risk_before = torch.zeros_like(target_prefix_mass)
    candidate_block_risk = torch.zeros_like(target_prefix_mass)
    block_risk_after = torch.zeros_like(target_prefix_mass)
    rescue_mask = torch.zeros_like(standard_accept_mask)
    reject_outside_top_p_mask = torch.zeros_like(standard_accept_mask)
    reject_risk_budget_mask = torch.zeros_like(standard_accept_mask)
    accept_mask = torch.zeros_like(standard_accept_mask)

    running_risk = torch.zeros(
        leading_shape[:-1],
        device=target_prefix_mass.device,
        dtype=target_prefix_mass.dtype,
    )
    active = torch.ones(
        leading_shape[:-1],
        device=standard_accept_mask.device,
        dtype=torch.bool,
    )
    for position in range(leading_shape[-1]):
        position_standard_accept = standard_accept_mask[..., position]
        position_rescue_eligible = rescue_eligibility_mask[..., position]
        position_token_risk = token_risk[..., position]

        block_risk_before[..., position] = torch.where(
            active,
            running_risk,
            torch.zeros_like(running_risk),
        )
        position_candidate_risk = running_risk + torch.where(
            position_standard_accept,
            torch.zeros_like(position_token_risk),
            position_token_risk,
        )
        candidate_block_risk[..., position] = torch.where(
            active,
            position_candidate_risk,
            torch.zeros_like(position_candidate_risk),
        )

        position_rescue = (
            active
            & ~position_standard_accept
            & position_rescue_eligible
            & (position_candidate_risk <= risk_budget_max)
        )
        position_reject_outside = (
            active & ~position_standard_accept & ~position_rescue_eligible
        )
        position_reject_budget = (
            active
            & ~position_standard_accept
            & position_rescue_eligible
            & (position_candidate_risk > risk_budget_max)
        )
        position_accept = active & (position_standard_accept | position_rescue)

        rescue_mask[..., position] = position_rescue
        reject_outside_top_p_mask[..., position] = position_reject_outside
        reject_risk_budget_mask[..., position] = position_reject_budget
        accept_mask[..., position] = position_accept
        running_risk = torch.where(
            position_rescue,
            position_candidate_risk,
            running_risk,
        )
        block_risk_after[..., position] = torch.where(
            position_accept,
            running_risk,
            torch.zeros_like(running_risk),
        )
        active = position_accept

    return RiskBudgetTopPResult(
        target_prefix_mass=target_prefix_mass,
        inside_target_top_p=inside_target_top_p,
        rescue_eligibility_mask=rescue_eligibility_mask,
        token_risk=token_risk,
        block_risk_before=block_risk_before,
        candidate_block_risk=candidate_block_risk,
        block_risk_after=block_risk_after,
        rescue_mask=rescue_mask,
        reject_outside_top_p_mask=reject_outside_top_p_mask,
        reject_risk_budget_mask=reject_risk_budget_mask,
        accept_mask=accept_mask,
    )


__all__ = [
    "DEFAULT_RISK_BUDGET_MAX",
    "DEFAULT_RISK_BUDGET_SAFE_TOP_P",
    "RiskBudgetTopPResult",
    "apply_block_risk_budget",
    "compute_risk_budget_top_p",
    "compute_target_prefix_mass",
    "validate_risk_budget_max",
    "validate_risk_budget_top_p_thresholds",
]
