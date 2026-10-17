from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from deepspec.utils.sampling import gather_token_probs


DEFAULT_MARS_RELAXATION_THRESHOLD = 0.90





def validate_mars_relaxation_threshold(value: float) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "mars_relaxation_threshold must be finite and in [0, 1], "
            f"got {value!r}."
        )
    return threshold


@dataclass
class MarsAcceptanceResult:
    acceptance_probability: torch.Tensor
    uniform_random: torch.Tensor
    standard_accept_mask: torch.Tensor
    top2_logit_ratio: torch.Tensor
    relaxation_mask: torch.Tensor
    inside_target_top2: torch.Tensor
    relaxed_accept_mask: torch.Tensor
    relaxed_reject_mask: torch.Tensor
    accept_mask: torch.Tensor
    accept_prefix_mask: torch.Tensor
    accepted_draft_tokens: int


def compute_mars_acceptance(
    *,
    target_probs: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    proposed_tokens: torch.Tensor,
    relaxation_threshold: float = DEFAULT_MARS_RELAXATION_THRESHOLD,
    uniform_random: torch.Tensor | None = None,
) -> MarsAcceptanceResult:
    """Apply the MARS margin-aware verification switch to one draft path.

    This follows the reference implementation: when the ratio of the target's
    second-largest logit to its largest logit exceeds the threshold, verify by
    target Top-2 membership; otherwise use standard rejection sampling.
    """
    relaxation_threshold = validate_mars_relaxation_threshold(
        relaxation_threshold
    )
    if target_probs.shape != target_logits.shape:
        raise ValueError("target_probs and target_logits must have the same shape")
    if target_probs.shape != draft_probs.shape:
        raise ValueError("target_probs and draft_probs must have the same shape")
    if target_probs.shape[:-1] != proposed_tokens.shape:
        raise ValueError("proposed_tokens must match the probability sequence shape")
    if int(target_probs.shape[-1]) < 2:
        raise ValueError("MARS requires a vocabulary containing at least two tokens")

    selected_target_probs = gather_token_probs(target_probs, proposed_tokens)
    selected_draft_probs = gather_token_probs(
        draft_probs,
        proposed_tokens,
    ).clamp_min(1e-8)
    acceptance_probability = torch.clamp(
        selected_target_probs / selected_draft_probs,
        max=1.0,
    )
    if uniform_random is None:
        uniform_random = torch.rand_like(acceptance_probability)
    if uniform_random.shape != acceptance_probability.shape:
        raise ValueError("uniform_random must match proposed_tokens")
    standard_accept_mask = uniform_random < acceptance_probability

    top2_logits, top2_indices = torch.topk(target_logits, k=2, dim=-1)

    top2_logit_ratio = top2_logits[..., 1] / (top2_logits[..., 0] + 1e-10)
    relaxation_mask = top2_logit_ratio > relaxation_threshold
    inside_target_top2 = (
        top2_indices == proposed_tokens.unsqueeze(-1)
    ).any(dim=-1)
    relaxed_accept_mask = relaxation_mask & inside_target_top2
    relaxed_reject_mask = relaxation_mask & ~inside_target_top2
    accept_mask = torch.where(
        relaxation_mask,
        inside_target_top2,
        standard_accept_mask,
    )
    accept_prefix_mask = accept_mask.to(torch.int64).cumprod(dim=1)
    accepted_draft_tokens = int(accept_prefix_mask[0].sum().item())

    return MarsAcceptanceResult(
        acceptance_probability=acceptance_probability,
        uniform_random=uniform_random,
        standard_accept_mask=standard_accept_mask,
        top2_logit_ratio=top2_logit_ratio,
        relaxation_mask=relaxation_mask,
        inside_target_top2=inside_target_top2,
        relaxed_accept_mask=relaxed_accept_mask,
        relaxed_reject_mask=relaxed_reject_mask,
        accept_mask=accept_mask,
        accept_prefix_mask=accept_prefix_mask,
        accepted_draft_tokens=accepted_draft_tokens,
    )


__all__ = [
    "DEFAULT_MARS_RELAXATION_THRESHOLD",
    "MarsAcceptanceResult",
    "compute_mars_acceptance",
    "validate_mars_relaxation_threshold",
]
