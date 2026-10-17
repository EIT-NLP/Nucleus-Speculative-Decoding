"""ASD-style budgeted contiguous-prefix verification.

The temperature-zero path is an independent implementation of Algorithm 1 in
arXiv:2608.03447.  At nonzero temperature the surrounding evaluator uses the
same target-mode regret budget but samples proposals and recovery tokens; that
is an experimental sampling adaptation, not the paper's greedy algorithm.
``suffix_length`` follows the paper; ``normalized_suffix`` exposes the supplied
ASD DSpark repo variant.  The ledger is owned by one generation request, never
reset between blocks.
"""
from dataclasses import dataclass
import math

import torch


def validate_asd(budget=8.0, gate=0.25, cap=2, value_scheme="suffix_length"):
    for name, value in (("asd_budget", budget), ("asd_gate", gate)):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if isinstance(cap, bool) or int(cap) != cap or int(cap) < 0:
        raise ValueError("asd_max_exceptions must be a nonnegative integer")
    if value_scheme not in {"suffix_length", "normalized_suffix"}:
        raise ValueError("Unknown ASD value scheme")


@dataclass
class ASDState:
    budget: float = 8.0
    spent: float = 0.0
    relaxed_tokens: int = 0
    blocks: int = 0

    def __post_init__(self):
        validate_asd(self.budget)
        if not math.isfinite(self.spent) or not 0 <= self.spent <= self.budget:
            raise ValueError("ASD spent must lie within the request budget")

    def statistics(self):
        return {"budget": self.budget, "spent": self.spent,
                "remaining": max(0.0, self.budget - self.spent),
                "relaxed_tokens": self.relaxed_tokens, "blocks": self.blocks}


@dataclass
class ASDDecision:
    accepted_draft_tokens: int
    accept_mask: torch.Tensor
    accept_prefix_mask: torch.Tensor
    regrets: torch.Tensor
    values: torch.Tensor
    mismatches: torch.Tensor
    terminated_by_stop_token: bool


def compute_asd_prefix(*, target_logits, proposed_tokens, state, gate=0.25,
                       cap=2, value_scheme="suffix_length", stop_token_ids=None,
                       max_emitted_tokens=None):
    """Select, clip to actually emitted tokens, then debit one request ledger.

    Target logits cover draft positions only; row i scores proposed token i.
    The caller chooses the recovery/bonus token from the target distribution
    at the first uncommitted row. There is no p/q acceptance, nucleus test, or
    residual recovery in this prefix decision.
    """
    validate_asd(state.budget, gate, cap, value_scheme)
    if target_logits.ndim != 3 or target_logits.shape[:2] != proposed_tokens.shape:
        raise ValueError("Expected aligned [1,K,V] logits and [1,K] tokens")
    if proposed_tokens.shape[0] != 1:
        raise ValueError("ASD adapter supports batch size one")
    if max_emitted_tokens is not None and int(max_emitted_tokens) < 0:
        raise ValueError("max_emitted_tokens must be nonnegative")
    logits = target_logits
    top_logits, top_ids = logits.max(dim=-1)
    chosen = logits.gather(-1, proposed_tokens.unsqueeze(-1)).squeeze(-1)
    regrets = (top_logits.double() - chosen.double()).clamp_min(0)
    if not torch.isfinite(regrets).all():
        raise ValueError("ASD requires finite target regrets")
    mismatches = proposed_tokens != top_ids
    width = proposed_tokens.shape[1]
    values = torch.arange(width, 0, -1, device=logits.device, dtype=torch.float64).unsqueeze(0)
    if value_scheme == "normalized_suffix" and width:
        values = values / width
    charged = torch.where(mismatches, regrets, 0.0)
    cumulative = charged.cumsum(dim=1)
    counts = mismatches.long().cumsum(dim=1)
    remaining = max(0.0, state.budget - state.spent)
    if state.budget == 0:
        feasible = ~mismatches
    else:
        affordable = cumulative <= remaining
        if value_scheme == "normalized_suffix":

            affordable = affordable & ((cumulative - charged) < remaining)
        feasible = (~mismatches) | ((regrets <= float(gate) * values)
                                    & affordable & (counts <= int(cap)))
    prefix = feasible.long().cumprod(dim=1)
    accepted = int(prefix.sum().item())
    if max_emitted_tokens is not None:
        accepted = min(accepted, int(max_emitted_tokens))
    terminated = False
    if accepted and stop_token_ids:
        stops = torch.tensor(stop_token_ids, device=proposed_tokens.device)
        hits = torch.isin(proposed_tokens[0, :accepted], stops).nonzero()
        if hits.numel():
            accepted = int(hits[0, 0].item()) + 1
            terminated = True
    prefix = torch.zeros_like(prefix)
    prefix[:, :accepted] = 1
    debit = float((charged * prefix).sum().item())
    state.spent = min(state.budget, state.spent + debit)
    state.relaxed_tokens += int((mismatches & prefix.bool()).sum().item())
    state.blocks += 1
    return ASDDecision(accepted, feasible, prefix, regrets, values, mismatches, terminated)
