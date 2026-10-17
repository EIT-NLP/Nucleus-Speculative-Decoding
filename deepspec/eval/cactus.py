from __future__ import annotations

import math

import torch





def validate_cactus_delta(delta: float) -> float:
    """Return a validated finite, nonnegative Cactus KL budget."""
    value = float(delta)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "cactus_delta must be finite and >= 0, "
            f"got {delta!r}."
        )
    return value


def cactus_adjusted_selected_probs(
    target_probs: torch.Tensor,
    proposed_tokens: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Compute Cactus's adjusted probability for each proposed token.

    ``target_probs`` has shape ``[..., vocab_size]`` and
    ``proposed_tokens`` has the corresponding leading shape ``[...]``.
    """
    delta = validate_cactus_delta(delta)
    selected_target_probs = target_probs.gather(
        dim=-1,
        index=proposed_tokens.unsqueeze(-1),
    ).squeeze(-1)
    variance = selected_target_probs * (1.0 - selected_target_probs)
    bonus = torch.sqrt(torch.clamp(2.0 * delta * variance, min=0.0))
    return torch.clamp(selected_target_probs + bonus, min=0.0, max=1.0)


def cactus_adjusted_distribution(
    target_probs: torch.Tensor,
    proposed_tokens: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Construct Cactus's proposal-conditioned adjusted distribution."""
    adjusted_selected_probs = cactus_adjusted_selected_probs(
        target_probs,
        proposed_tokens,
        delta,
    )
    selected_target_probs = target_probs.gather(
        dim=-1,
        index=proposed_tokens.unsqueeze(-1),
    ).squeeze(-1)

    remaining_target_mass = 1.0 - selected_target_probs
    remaining_adjusted_mass = 1.0 - adjusted_selected_probs
    scale = torch.where(
        remaining_target_mass > 0.0,
        remaining_adjusted_mass / remaining_target_mass,
        torch.ones_like(remaining_target_mass),
    )
    adjusted_probs = target_probs * scale.unsqueeze(-1)
    adjusted_probs = adjusted_probs.scatter(
        dim=-1,
        index=proposed_tokens.unsqueeze(-1),
        src=adjusted_selected_probs.unsqueeze(-1),
    )
    return torch.clamp(adjusted_probs, min=0.0, max=1.0)


__all__ = [
    "cactus_adjusted_distribution",
    "cactus_adjusted_selected_probs",
    "validate_cactus_delta",
]
