from __future__ import annotations

import math
from dataclasses import dataclass

import torch


DYNAMIC_TOP_P_INITIAL_THRESHOLD = 0.95
DYNAMIC_TOP_P_MIN_THRESHOLD = 0.40
DYNAMIC_TOP_P_MAX_THRESHOLD = 0.95
DYNAMIC_TOP_P_MAX_SUPPORT = 1.0
DYNAMIC_TOP_P_SIGNAL_ENTROPY_DISAGREEMENT = "entropy_disagreement"
DYNAMIC_TOP_P_SIGNAL_ENTROPY_ONLY = "entropy_only"
DYNAMIC_TOP_P_SIGNALS = frozenset(
    {
        DYNAMIC_TOP_P_SIGNAL_ENTROPY_DISAGREEMENT,
        DYNAMIC_TOP_P_SIGNAL_ENTROPY_ONLY,
    }
)


def validate_dynamic_top_p_blend(value: float) -> float:
    blend = float(value)
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError(
            "dynamic_top_p_blend must be finite and in [0, 1], "
            f"got {value}."
        )
    return blend


def validate_dynamic_top_p_support(value: float) -> float:
    support = float(value)
    if (
        not math.isfinite(support)
        or not DYNAMIC_TOP_P_INITIAL_THRESHOLD <= support <= DYNAMIC_TOP_P_MAX_SUPPORT
    ):
        raise ValueError(
            "dynamic_top_p_support must be finite and in [0.95, 1], "
            f"got {value}."
        )
    return support


def validate_dynamic_top_p_outer_alpha(value: float) -> float:
    outer_alpha = float(value)
    if not math.isfinite(outer_alpha) or not 0.0 <= outer_alpha <= 1.0:
        raise ValueError(
            "dynamic_top_p_outer_alpha must be finite and in [0, 1], "
            f"got {value}."
        )
    return outer_alpha


def validate_dynamic_top_p_signal(value: str) -> str:
    signal = str(value)
    if signal not in DYNAMIC_TOP_P_SIGNALS:
        raise ValueError(
            "dynamic_top_p_signal must be one of "
            f"{sorted(DYNAMIC_TOP_P_SIGNALS)}, got {value!r}."
        )
    return signal


@dataclass(frozen=True)
class DynamicTopPResult:
    """Per-position statistics used by Dynamic Top-P Rescue."""

    inside_initial_nucleus: torch.Tensor
    initial_nucleus_size: torch.Tensor
    normalized_nucleus_entropy: torch.Tensor
    draft_target_tv_distance: torch.Tensor
    dynamic_top_p: torch.Tensor
    inside_dynamic_nucleus: torch.Tensor
    dynamic_nucleus_size: torch.Tensor


def _validate_inputs(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    proposed_tokens: torch.Tensor,
) -> None:
    if target_probs.shape != draft_probs.shape:
        raise ValueError(
            "target_probs and draft_probs must have the same shape, "
            f"got {tuple(target_probs.shape)} and {tuple(draft_probs.shape)}."
        )
    if target_probs.ndim < 2:
        raise ValueError(
            "target_probs and draft_probs must include a vocabulary dimension."
        )
    if proposed_tokens.shape != target_probs.shape[:-1]:
        raise ValueError(
            "proposed_tokens shape must match the probability leading dimensions, "
            f"got {tuple(proposed_tokens.shape)} and {tuple(target_probs.shape[:-1])}."
        )


def _sorted_nucleus_mask(
    cumulative_before: torch.Tensor,
    threshold: float | torch.Tensor,
) -> torch.Tensor:
    if torch.is_tensor(threshold):
        threshold = threshold.to(
            device=cumulative_before.device,
            dtype=cumulative_before.dtype,
        ).unsqueeze(-1)
    else:
        threshold = float(threshold)


    return cumulative_before < threshold


def compute_dynamic_top_p(
    *,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    proposed_tokens: torch.Tensor,
    dynamic_top_p_blend: float = 0.0,
    dynamic_top_p_support: float = DYNAMIC_TOP_P_INITIAL_THRESHOLD,
    dynamic_top_p_outer_alpha: float = 0.0,
    dynamic_top_p_signal: str = DYNAMIC_TOP_P_SIGNAL_ENTROPY_DISAGREEMENT,
) -> DynamicTopPResult:
    """Compute the hard rescue nucleus and its diagnostics.

    The caller remains responsible for applying these results only to tokens
    rejected by the standard speculative-decoding acceptance test.
    """

    _validate_inputs(target_probs, draft_probs, proposed_tokens)
    dynamic_top_p_blend = validate_dynamic_top_p_blend(dynamic_top_p_blend)
    dynamic_top_p_support = validate_dynamic_top_p_support(dynamic_top_p_support)
    dynamic_top_p_outer_alpha = validate_dynamic_top_p_outer_alpha(
        dynamic_top_p_outer_alpha
    )
    dynamic_top_p_signal = validate_dynamic_top_p_signal(dynamic_top_p_signal)

    target_probs_float = target_probs.float()
    draft_probs_float = draft_probs.float()
    sorted_probs, sorted_indices = torch.sort(
        target_probs_float,
        dim=-1,
        descending=True,
    )
    cumulative_before = sorted_probs.cumsum(dim=-1) - sorted_probs

    entropy_core_sorted_mask = _sorted_nucleus_mask(
        cumulative_before,
        DYNAMIC_TOP_P_INITIAL_THRESHOLD,
    )
    entropy_core_nucleus_size = entropy_core_sorted_mask.sum(dim=-1)
    entropy_core_mass = sorted_probs.masked_fill(
        ~entropy_core_sorted_mask,
        0.0,
    ).sum(
        dim=-1,
        keepdim=True,
    )
    normalized_entropy_core_probs = torch.where(
        entropy_core_sorted_mask,
        sorted_probs / entropy_core_mass.clamp_min(1e-20),
        torch.zeros_like(sorted_probs),
    )
    entropy_terms = (
        normalized_entropy_core_probs
        * normalized_entropy_core_probs.clamp_min(1e-20).log()
    )
    nucleus_entropy = -entropy_terms.sum(dim=-1)
    normalized_nucleus_entropy = torch.where(
        entropy_core_nucleus_size > 1,
        nucleus_entropy
        / entropy_core_nucleus_size.float().log().clamp_min(1e-20),
        torch.zeros_like(nucleus_entropy),
    ).clamp_(0.0, 1.0)

    support_sorted_mask = _sorted_nucleus_mask(
        cumulative_before,
        dynamic_top_p_support,
    )
    support_nucleus_size = support_sorted_mask.sum(dim=-1)

    draft_target_tv_distance = (
        0.5 * (target_probs_float - draft_probs_float).abs().sum(dim=-1)
    ).clamp_(0.0, 1.0)
    raw_dynamic_top_p = 0.80 + 0.15 * normalized_nucleus_entropy
    outer_agreement_weight: torch.Tensor | float = 1.0
    if dynamic_top_p_signal == DYNAMIC_TOP_P_SIGNAL_ENTROPY_DISAGREEMENT:
        raw_dynamic_top_p = raw_dynamic_top_p - draft_target_tv_distance * (
            0.05 + 0.35 * (1.0 - normalized_nucleus_entropy)
        )
        outer_agreement_weight = 1.0 - draft_target_tv_distance
    raw_dynamic_top_p = raw_dynamic_top_p.clamp(
        DYNAMIC_TOP_P_MIN_THRESHOLD,
        DYNAMIC_TOP_P_MAX_THRESHOLD,
    )
    inner_dynamic_top_p = raw_dynamic_top_p + dynamic_top_p_blend * (
        DYNAMIC_TOP_P_MAX_THRESHOLD - raw_dynamic_top_p
    )
    outer_weight = (
        dynamic_top_p_outer_alpha
        * normalized_nucleus_entropy
        * outer_agreement_weight
    )
    dynamic_top_p = inner_dynamic_top_p + outer_weight * (
        dynamic_top_p_support - inner_dynamic_top_p
    )

    dynamic_sorted_mask = _sorted_nucleus_mask(cumulative_before, dynamic_top_p)
    dynamic_nucleus_size = dynamic_sorted_mask.sum(dim=-1)

    support_nucleus_mask = torch.zeros_like(
        support_sorted_mask,
        dtype=torch.bool,
    ).scatter_(dim=-1, index=sorted_indices, src=support_sorted_mask)
    dynamic_nucleus_mask = torch.zeros_like(
        dynamic_sorted_mask,
        dtype=torch.bool,
    ).scatter_(dim=-1, index=sorted_indices, src=dynamic_sorted_mask)
    proposed_token_indices = proposed_tokens.unsqueeze(-1)

    return DynamicTopPResult(
        inside_initial_nucleus=support_nucleus_mask.gather(
            dim=-1,
            index=proposed_token_indices,
        ).squeeze(-1),
        initial_nucleus_size=support_nucleus_size,
        normalized_nucleus_entropy=normalized_nucleus_entropy,
        draft_target_tv_distance=draft_target_tv_distance,
        dynamic_top_p=dynamic_top_p,
        inside_dynamic_nucleus=dynamic_nucleus_mask.gather(
            dim=-1,
            index=proposed_token_indices,
        ).squeeze(-1),
        dynamic_nucleus_size=dynamic_nucleus_size,
    )


__all__ = [
    "DYNAMIC_TOP_P_INITIAL_THRESHOLD",
    "DYNAMIC_TOP_P_MAX_THRESHOLD",
    "DYNAMIC_TOP_P_MAX_SUPPORT",
    "DYNAMIC_TOP_P_MIN_THRESHOLD",
    "DYNAMIC_TOP_P_SIGNALS",
    "DYNAMIC_TOP_P_SIGNAL_ENTROPY_DISAGREEMENT",
    "DYNAMIC_TOP_P_SIGNAL_ENTROPY_ONLY",
    "DynamicTopPResult",
    "compute_dynamic_top_p",
    "validate_dynamic_top_p_blend",
    "validate_dynamic_top_p_outer_alpha",
    "validate_dynamic_top_p_signal",
    "validate_dynamic_top_p_support",
]
