from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
from transformers import AutoModel, AutoTokenizer, DynamicCache

from deepspec.eval.base_evaluator import (
    BaseEvaluator,
    _target_top_p_accept_mask,
    trim_output_ids,
)
from deepspec.eval.approximate_sd import ASDState, compute_asd_prefix
from deepspec.eval.base_evaluator import DraftProposal, VerificationResult
from deepspec.eval.cactus import (
    cactus_adjusted_distribution,
    cactus_adjusted_selected_probs,
    validate_cactus_delta,
)
from deepspec.eval.mars import (
    compute_mars_acceptance,
    validate_mars_relaxation_threshold,
)
from deepspec.eval.verification_timing import VerificationPhaseTimer
from deepspec.utils.sampling import (
    gather_token_probs,
    logits_to_probs,
    sample_from_probs,
    sample_residual,
)


def run_linear_spec_with_acceptance_capture(
    model,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    block_length: int,
    temperature: float,
    eos_token_id: int | None,
    threshold: float,
    post_verify=None,
    dataset_name: str | None = None,
    sample_index: int | None = None,
) -> tuple[torch.Tensor, int, list[int]]:
    """Run NVIDIA's implementation and capture commits and diagnostics.

    Generation remains delegated to the model's official
    ``linear_spec_generate`` method.  When a diagnostics callback is supplied,
    temporary forward wrappers retain the draft/target logits already computed
    by that method; they do not alter its samples or exact-match verifier.
    """
    if post_verify is not None and float(threshold) != 0.0:
        raise ValueError(
            "Official Linear Self-Speculation token diagnostics currently "
            "require threshold=0.0. Iterative threshold drafting has a "
            "selection-conditioned proposal distribution that cannot be "
            "represented by only the final raw softmax."
        )

    generate_method = model.linear_spec_generate
    function = getattr(generate_method, "__func__", generate_method)
    function = inspect.unwrap(function)
    function_globals = getattr(function, "__globals__", None)
    if not isinstance(function_globals, dict):
        raise RuntimeError("Unable to inspect linear_spec_generate globals")

    crop_name = "_crop_dynamic_cache"
    original_crop = function_globals.get(crop_name)
    if original_crop is None:
        raise RuntimeError(
            "The Nemotron Linear Self-Speculation implementation does not "
            "expose _crop_dynamic_cache; acceptance lengths cannot be measured"
        )

    committed_cache_length = int(prompt_ids.shape[1])
    acceptance_lengths: list[int] = []
    proposal_index = 0

    encoder = getattr(model, "encoder", None)
    diffusion_head = getattr(model, "diffusion_head", None)
    original_encoder_forward = None
    original_head_forward = None
    capture_phase: str | None = None
    captured_verify_input_ids: torch.Tensor | None = None
    captured_draft_logits: torch.Tensor | None = None
    captured_target_logits: torch.Tensor | None = None

    if post_verify is not None:
        if encoder is None or diffusion_head is None:
            raise RuntimeError(
                "Official Linear Self-Speculation diagnostics require the "
                "model encoder and diffusion_head modules"
            )
        original_encoder_forward = encoder.forward
        original_head_forward = diffusion_head.forward

        def capture_encoder_forward(*args, **kwargs):
            nonlocal capture_phase, captured_verify_input_ids
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            output = original_encoder_forward(*args, **kwargs)
            if (
                isinstance(input_ids, torch.Tensor)
                and input_ids.ndim == 2
                and int(input_ids.shape[1]) == int(block_length)
            ):
                if bool(kwargs.get("use_causal_mask", False)):
                    capture_phase = "target"
                    captured_verify_input_ids = input_ids.detach().clone()
                else:
                    capture_phase = "draft"
            else:
                capture_phase = None
            return output

        def capture_head_forward(*args, **kwargs):
            nonlocal captured_draft_logits, captured_target_logits
            logits = original_head_forward(*args, **kwargs)
            if (
                isinstance(logits, torch.Tensor)
                and logits.ndim == 3
                and int(logits.shape[1]) == int(block_length)
            ):
                if capture_phase == "draft":
                    captured_draft_logits = logits.detach()
                elif capture_phase == "target":
                    captured_target_logits = logits.detach()
            return logits

        encoder.forward = capture_encoder_forward
        diffusion_head.forward = capture_head_forward

    def capture_crop(past_key_values, max_length: int):
        nonlocal committed_cache_length, proposal_index
        max_length = int(max_length)
        accepted = max_length - committed_cache_length
        if accepted < 1:
            raise RuntimeError(
                "Nemotron Linear Self-Speculation committed a non-positive "
                f"token count: previous={committed_cache_length}, "
                f"new={max_length}"
            )

        if post_verify is not None:
            if (
                captured_verify_input_ids is None
                or captured_draft_logits is None
                or captured_target_logits is None
            ):
                raise RuntimeError(
                    "Unable to capture official Linear Self-Speculation "
                    "draft/target tensors for token diagnostics"
                )
            draft_probs_full = logits_to_probs(
                captured_draft_logits,
                float(temperature),
            )
            target_probs = logits_to_probs(
                captured_target_logits,
                float(temperature),
                top_p=1.0,
            )
            accepted_draft_tokens = min(
                int(block_length) - 1,
                max(0, accepted - 1),
            )
            accept_prefix_mask = torch.zeros(
                (1, int(block_length) - 1),
                dtype=torch.int64,
                device=captured_target_logits.device,
            )
            accept_prefix_mask[:, :accepted_draft_tokens] = 1
            proposal = SimpleNamespace(
                draft_token_count=int(block_length) - 1,
                verify_input_ids=captured_verify_input_ids,
                draft_probs=draft_probs_full[:, 1:, :],
                confidence_logits=None,
                draft_block_size=int(block_length) - 1,
            )
            verification = SimpleNamespace(
                target_output=SimpleNamespace(logits=captured_target_logits),
                target_probs=target_probs,
                accept_prefix_mask=accept_prefix_mask,
                accept_mask=accept_prefix_mask.bool(),
                accepted_draft_tokens=accepted_draft_tokens,
                next_token=None,
                effective_proposal_length=int(block_length) - 1,
                standard_accept_mask=None,
                standard_uniform_random=None,
                target_top_p_accept_mask=None,
                logical_draft_block_count=1,
                blind_accept_prefix_length=0,
                forced_reject_position=None,
            )
            post_verify(
                proposal,
                verification,
                start=committed_cache_length - int(prompt_ids.shape[1]),
                proposal_index=proposal_index,
                dataset_name=dataset_name,
                sample_index=sample_index,
            )
            proposal_index += 1

        acceptance_lengths.append(accepted)
        committed_cache_length = max_length
        return original_crop(past_key_values, max_length)

    function_globals[crop_name] = capture_crop
    try:
        output_ids, nfe = generate_method(
            prompt_ids,
            max_new_tokens=int(max_new_tokens),
            block_length=int(block_length),
            temperature=float(temperature),
            eos_token_id=eos_token_id,
            threshold=float(threshold),
        )
    finally:
        function_globals[crop_name] = original_crop
        if original_encoder_forward is not None:
            encoder.forward = original_encoder_forward
        if original_head_forward is not None:
            diffusion_head.forward = original_head_forward

    return output_ids, int(nfe), acceptance_lengths


def normalize_acceptance_lengths(
    acceptance_lengths: list[int],
    *,
    num_output_tokens: int,
) -> list[int]:
    """Trim recorded commits to the returned output (EOS or token cap)."""
    remaining = max(0, int(num_output_tokens) - 1)
    normalized: list[int] = []
    for value in acceptance_lengths:
        if remaining == 0:
            break
        committed = min(int(value), remaining)
        if committed > 0:
            normalized.append(committed)
            remaining -= committed
    if remaining != 0:
        raise RuntimeError(
            "Nemotron output-token accounting disagrees with captured cache "
            f"commits: {remaining} tokens are unaccounted for"
        )
    return normalized


def compute_hybrid_acceptance_prefix(
    *,
    target_probs: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    proposed_tokens: torch.Tensor,
    acceptance_top_p: float,
    uniform_random: torch.Tensor | None = None,
    enable_target_top_p_rescue: bool = True,
    clean_nsd: bool = False,
) -> SimpleNamespace:
    """Apply rejection sampling, then rescue failures inside target Top-P."""
    if not 0.0 <= float(acceptance_top_p) <= 1.0:
        raise ValueError("acceptance_top_p must be in [0, 1]")
    if target_probs.shape != target_logits.shape or target_probs.shape != draft_probs.shape:
        raise ValueError("target_probs, target_logits and draft_probs must match")
    if target_probs.shape[:-1] != proposed_tokens.shape:
        raise ValueError("proposed_tokens must match the probability sequence shape")

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
    if clean_nsd:
        from deepspec.eval.clean_nsd import target_top_p_accept_mask as nucleus_membership
    else:
        nucleus_membership = _target_top_p_accept_mask
    target_top_p_accept_mask = (
        nucleus_membership(
            target_probs=target_probs,
            target_logits=target_logits if clean_nsd else target_logits.float(),
            proposed_tokens=proposed_tokens,
            acceptance_top_p=float(acceptance_top_p),
        )
        if enable_target_top_p_rescue
        else torch.zeros_like(standard_accept_mask)
    )
    rescue_mask = ~standard_accept_mask & target_top_p_accept_mask
    accept_mask = standard_accept_mask | rescue_mask
    accept_prefix_mask = accept_mask.to(torch.int64).cumprod(dim=1)
    accepted_draft_tokens = int(accept_prefix_mask[0].sum().item())
    return SimpleNamespace(
        accepted_draft_tokens=accepted_draft_tokens,
        acceptance_probability=acceptance_probability,
        uniform_random=uniform_random,
        standard_accept_mask=standard_accept_mask,
        target_top_p_accept_mask=target_top_p_accept_mask,
        rescue_mask=rescue_mask,
        accept_mask=accept_mask,
        accept_prefix_mask=accept_prefix_mask,
    )


def compute_cactus_acceptance_prefix(
    *,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    proposed_tokens: torch.Tensor,
    cactus_delta: float,
    uniform_random: torch.Tensor | None = None,
) -> SimpleNamespace:
    """Apply Cactus acceptance to a Nemotron diffusion proposal block."""
    cactus_delta = validate_cactus_delta(cactus_delta)
    if target_probs.shape != draft_probs.shape:
        raise ValueError("target_probs and draft_probs must match")
    if target_probs.shape[:-1] != proposed_tokens.shape:
        raise ValueError("proposed_tokens must match the probability sequence shape")

    selected_draft_probs = gather_token_probs(
        draft_probs,
        proposed_tokens,
    ).clamp_min(1e-8)
    adjusted_selected_probs = cactus_adjusted_selected_probs(
        target_probs,
        proposed_tokens,
        cactus_delta,
    )
    acceptance_probability = torch.clamp(
        adjusted_selected_probs / selected_draft_probs,
        max=1.0,
    )
    if uniform_random is None:
        uniform_random = torch.rand_like(acceptance_probability)
    if uniform_random.shape != acceptance_probability.shape:
        raise ValueError("uniform_random must match proposed_tokens")

    accept_mask = uniform_random < acceptance_probability
    accept_prefix_mask = accept_mask.to(torch.int64).cumprod(dim=1)
    accepted_draft_tokens = int(accept_prefix_mask[0].sum().item())
    return SimpleNamespace(
        accepted_draft_tokens=accepted_draft_tokens,
        acceptance_probability=acceptance_probability,
        uniform_random=uniform_random,
        accept_mask=accept_mask,
        accept_prefix_mask=accept_prefix_mask,
        adjusted_selected_probs=adjusted_selected_probs,
    )


@torch.no_grad()
def run_linear_spec_rejection_sampling_then_target_top_p(
    model,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    block_length: int,
    temperature: float,
    mask_token_id: int | None,
    eos_token_id: int | None,
    acceptance_top_p: float,
    target_sampling_top_p: float,
    enable_target_top_p_rescue: bool = True,
    clean_nsd: bool = False,
    cactus_delta: float | None = None,
    mars_relaxation_threshold: float | None = None,
    asd_state: ASDState | None = None,
    asd_gate: float = 0.25,
    asd_max_exceptions: int = 2,
    asd_value_scheme: str = "suffix_length",
    post_verify=None,
    dataset_name: str | None = None,
    sample_index: int | None = None,
    verification_phase_timer: VerificationPhaseTimer | None = None,
) -> tuple[torch.Tensor, int, list[int], int]:
    """Nemotron Linear Self-Speculation with a sampled verifier."""
    if prompt_ids.shape[0] != 1:
        raise ValueError("Linear Self-Speculation requires batch_size == 1")
    if int(block_length) < 2:
        raise ValueError("block_length must be >= 2")
    if int(max_new_tokens) < 1:
        raise ValueError("max_new_tokens must be >= 1")
    use_cactus = cactus_delta is not None
    use_mars = mars_relaxation_threshold is not None
    use_asd = asd_state is not None
    if use_asd and (use_cactus or use_mars or enable_target_top_p_rescue or clean_nsd):
        raise ValueError("ASD cannot be combined with another relaxed verifier")
    if use_cactus and use_mars:
        raise ValueError("Cactus and MARS verification cannot be enabled together")
    if use_cactus:
        cactus_delta = validate_cactus_delta(cactus_delta)
    if use_mars:
        mars_relaxation_threshold = validate_mars_relaxation_threshold(
            mars_relaxation_threshold
        )

    token_mask_id = (
        int(mask_token_id)
        if mask_token_id is not None
        else int(model.config.mask_token_id)
    )
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    device = prompt_ids.device



    clean_attention_modules = (
        [layer.self_attn for layer in model.encoder.layers
         if hasattr(layer.self_attn, "diffusion_lm")] if clean_nsd else None
    )
    clean_adapter_modules = (
        [module for module in model.modules() if hasattr(module, "_disable_adapters")]
        if clean_nsd else None
    )

    def set_diffusion_lm(value: bool) -> None:
        if clean_nsd:
            for module in clean_attention_modules:
                module.diffusion_lm = value
            return
        for layer in model.encoder.layers:
            if hasattr(layer.self_attn, "diffusion_lm"):
                layer.self_attn.diffusion_lm = value

    def toggle_adapters(enable: bool) -> None:
        if clean_nsd:
            for module in clean_adapter_modules:
                module._disable_adapters = not enable
            return
        for module in model.modules():
            if hasattr(module, "_disable_adapters"):
                module._disable_adapters = not enable


    set_diffusion_lm(False)
    toggle_adapters(False)
    past_key_values = DynamicCache()
    encoder_output = model.encoder(
        input_ids=prompt_ids,
        past_key_values=past_key_values,
        use_cache=True,
        use_causal_mask=True,
    )
    past_key_values = encoder_output.past_key_values
    initial_logits = model.diffusion_head(
        encoder_output.last_hidden_state[:, -1:, :]
    )
    initial_probs = logits_to_probs(
        initial_logits,
        float(temperature),
        top_p=float(target_sampling_top_p),
    )
    next_token = sample_from_probs(initial_probs)
    nfe = 1

    generated = [next_token]
    total_generated = 1
    acceptance_lengths: list[int] = []
    rescue_count = 0
    proposal_index = 0
    if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
        return torch.cat([prompt_ids, next_token], dim=1), nfe, [], rescue_count

    if verification_phase_timer is not None:
        verification_phase_timer.start_generation()
    while total_generated < int(max_new_tokens):
        cache_length = int(past_key_values.get_seq_length())
        block = torch.full(
            (1, int(block_length)),
            token_mask_id,
            dtype=torch.long,
            device=device,
        )
        block[:, 0] = next_token[:, 0]




        set_diffusion_lm(True)
        toggle_adapters(True)
        draft_output = model.encoder(
            input_ids=block,
            past_key_values=past_key_values,
            use_cache=False,
        )
        nfe += 1
        draft_logits = model.diffusion_head(draft_output.last_hidden_state)
        draft_probs_full = logits_to_probs(draft_logits, float(temperature))
        draft_tokens = sample_from_probs(draft_probs_full)
        block[:, 1:] = draft_tokens[:, 1:]
        draft_probs = draft_probs_full[:, 1:, :]


        set_diffusion_lm(False)
        toggle_adapters(False)
        if verification_phase_timer is not None:
            verification_phase_timer.start_target_model()
        target_output = model.encoder(
            input_ids=block,
            past_key_values=past_key_values,
            use_cache=True,
            use_causal_mask=True,
        )
        past_key_values = target_output.past_key_values
        nfe += 1
        target_logits = model.diffusion_head(target_output.last_hidden_state)
        if verification_phase_timer is not None:
            verification_phase_timer.stop_target_model()
            verification_phase_timer.start_acceptance_rule()
        target_probs = logits_to_probs(
            target_logits,
            float(temperature),
            top_p=float(target_sampling_top_p),
        )

        proposed_tokens = block[:, 1:]
        if use_asd:
            acceptance = compute_asd_prefix(
                target_logits=target_logits[:, :-1, :], proposed_tokens=proposed_tokens,
                state=asd_state, gate=asd_gate, cap=asd_max_exceptions,
                value_scheme=asd_value_scheme,
                stop_token_ids=[] if eos_token_id is None else [int(eos_token_id)],
                max_emitted_tokens=int(max_new_tokens) - total_generated,
            )
        elif use_cactus:
            acceptance = compute_cactus_acceptance_prefix(
                target_probs=target_probs[:, :-1, :],
                draft_probs=draft_probs,
                proposed_tokens=proposed_tokens,
                cactus_delta=float(cactus_delta),
            )
        elif use_mars:
            acceptance = compute_mars_acceptance(
                target_probs=target_probs[:, :-1, :],
                target_logits=target_logits[:, :-1, :].float(),
                draft_probs=draft_probs,
                proposed_tokens=proposed_tokens,
                relaxation_threshold=float(mars_relaxation_threshold),
            )
        else:
            acceptance = compute_hybrid_acceptance_prefix(
                target_probs=target_probs[:, :-1, :],
                target_logits=target_logits[:, :-1, :],
                draft_probs=draft_probs,
                proposed_tokens=proposed_tokens,
                acceptance_top_p=float(acceptance_top_p),
                enable_target_top_p_rescue=enable_target_top_p_rescue,
                clean_nsd=clean_nsd,
            )
        accepted_draft_tokens = int(acceptance.accepted_draft_tokens)
        if not use_cactus and not use_mars and not use_asd:
            rescue_count += int(
                acceptance.rescue_mask[:, :accepted_draft_tokens].sum().item()
            )

        accepted_tokens = proposed_tokens[:, :accepted_draft_tokens]
        if use_asd:
            next_token = sample_from_probs(
                target_probs[:, accepted_draft_tokens : accepted_draft_tokens + 1, :]
            )
            if verification_phase_timer is not None:
                verification_phase_timer.stop_acceptance_rule()
            if post_verify is not None:
                post_verify(
                    DraftProposal(int(block_length) - 1, block, draft_probs),
                    VerificationResult(
                        target_output=SimpleNamespace(logits=target_logits), target_probs=target_probs,
                        accept_prefix_mask=acceptance.accept_prefix_mask,
                        accepted_draft_tokens=accepted_draft_tokens, next_token=next_token[:, 0],
                        effective_proposal_length=(accepted_draft_tokens if acceptance.terminated_by_stop_token
                                                   else int(block_length) - 1),
                        terminated_by_stop_token=acceptance.terminated_by_stop_token,
                        accept_mask=acceptance.accept_mask, asd_decision=acceptance,
                        asd_statistics=asd_state.statistics(),
                    ), start=total_generated - 1, proposal_index=proposal_index,
                    dataset_name=dataset_name, sample_index=sample_index,
                )
            if acceptance.terminated_by_stop_token:
                acceptance_lengths.append(accepted_draft_tokens)
                generated.append(accepted_tokens)
                past_key_values.crop(cache_length + accepted_draft_tokens)
                break
        elif accepted_draft_tokens < int(block_length) - 1:
            recovery_target_probs = target_probs[:, accepted_draft_tokens, :]
            if use_cactus:
                recovery_target_probs = cactus_adjusted_distribution(
                    recovery_target_probs,
                    proposed_tokens[:, accepted_draft_tokens],
                    float(cactus_delta),
                )
            mars_relaxation_rejection = (
                use_mars
                and bool(
                    acceptance.relaxation_mask[
                        0, accepted_draft_tokens
                    ].item()
                )
            )
            if mars_relaxation_rejection:
                next_token = sample_from_probs(
                    recovery_target_probs.unsqueeze(1)
                )
            else:
                next_token = sample_residual(
                    recovery_target_probs,
                    draft_probs[:, accepted_draft_tokens, :],
                ).unsqueeze(1)
        else:
            next_token = sample_from_probs(target_probs[:, -1:, :])

        if verification_phase_timer is not None and not use_asd:
            verification_phase_timer.stop_acceptance_rule()
        if post_verify is not None and not use_asd:
            proposal = SimpleNamespace(
                draft_token_count=int(block_length) - 1,
                verify_input_ids=block,
                draft_probs=draft_probs,
                confidence_logits=None,
                draft_block_size=int(block_length) - 1,
            )
            verification = SimpleNamespace(
                target_output=SimpleNamespace(logits=target_logits),
                target_probs=target_probs,
                accept_prefix_mask=acceptance.accept_prefix_mask,
                accept_mask=acceptance.accept_mask,
                accepted_draft_tokens=accepted_draft_tokens,
                next_token=next_token,
                effective_proposal_length=int(block_length) - 1,
                standard_accept_mask=(
                    None if use_cactus else acceptance.standard_accept_mask
                ),
                standard_uniform_random=(
                    None if use_cactus else acceptance.uniform_random
                ),
                target_top_p_accept_mask=(
                    acceptance.target_top_p_accept_mask
                    if not use_cactus and enable_target_top_p_rescue
                    and not use_mars
                    else None
                ),
                cactus_adjusted_selected_probs=(
                    acceptance.adjusted_selected_probs if use_cactus else None
                ),
                cactus_accept_prob=(
                    acceptance.acceptance_probability if use_cactus else None
                ),
                cactus_accept_mask=(
                    acceptance.accept_mask if use_cactus else None
                ),
                cactus_delta=(float(cactus_delta) if use_cactus else 1.0),
                mars_top2_logit_ratio=(
                    acceptance.top2_logit_ratio if use_mars else None
                ),
                mars_relaxation_mask=(
                    acceptance.relaxation_mask if use_mars else None
                ),
                mars_inside_target_top2=(
                    acceptance.inside_target_top2 if use_mars else None
                ),
                mars_relaxed_accept_mask=(
                    acceptance.relaxed_accept_mask if use_mars else None
                ),
                mars_relaxed_reject_mask=(
                    acceptance.relaxed_reject_mask if use_mars else None
                ),
                mars_relaxation_threshold=(
                    float(mars_relaxation_threshold) if use_mars else 0.90
                ),
                logical_draft_block_count=1,
                blind_accept_prefix_length=0,
                forced_reject_position=None,
            )
            post_verify(
                proposal,
                verification,
                start=total_generated - 1,
                proposal_index=proposal_index,
                dataset_name=dataset_name,
                sample_index=sample_index,
            )
        proposal_index += 1

        committed_length = accepted_draft_tokens + 1
        acceptance_lengths.append(committed_length)
        generated.append(accepted_tokens)
        generated.append(next_token)
        total_generated += committed_length
        past_key_values.crop(cache_length + committed_length)

        committed_tokens = torch.cat([accepted_tokens, next_token], dim=1)
        if eos_token_id is not None:
            eos_positions = (
                committed_tokens[0] == int(eos_token_id)
            ).nonzero(as_tuple=True)[0]
            if eos_positions.numel() > 0:
                first_eos = int(eos_positions[0].item())
                generated[-2:] = [committed_tokens[:, : first_eos + 1]]
                total_generated -= committed_length - first_eos - 1
                break

    if verification_phase_timer is not None:
        verification_phase_timer.stop_generation()
    output_ids = torch.cat([prompt_ids, *generated], dim=1)
    return output_ids, nfe, acceptance_lengths, rescue_count


class NemotronLabsDiffusionEvaluator(BaseEvaluator):
    """Benchmark NVIDIA's diffusion/AR Linear Self-Speculation mode."""

    @property
    def block_length(self) -> int:
        return int(getattr(self.args, "nemotron_linear_spec_block_length", 32))

    @property
    def threshold(self) -> float:
        return float(getattr(self.args, "nemotron_linear_spec_threshold", 0.0))

    @property
    def use_lora(self) -> bool:
        return bool(int(getattr(self.args, "nemotron_linear_spec_use_lora", 0)))

    @property
    def lora_subfolder(self) -> str:
        return str(
            getattr(
                self.args,
                "nemotron_linear_spec_lora_subfolder",
                "linear_spec_lora",
            )
        )

    @property
    def max_proposal_tokens(self) -> int:
        if getattr(self.args, "acceptance_rule", None) == "nemotron_diffusion":
            return 1
        return self.block_length - 1

    def build_models(self):
        if self.args.target_name_or_path != self.args.draft_name_or_path:
            raise ValueError(
                "Nemotron Linear Self-Speculation is a single-model method; "
                "target_name_or_path and draft_name_or_path must be identical"
            )
        if self.block_length < 2:
            raise ValueError("Nemotron Linear Self-Speculation block length must be >= 2")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Nemotron Linear Self-Speculation threshold must be in [0, 1]")
        acceptance_rule = getattr(self.args, "acceptance_rule", None)
        supported_rules = {
            "asd",
            "cleanNSD",
            "cactus",
            "mars",
            "linear_self_speculation",
            "rejection_sampling",
            "rejection_sampling_then_target_top_p",
            "nemotron_diffusion",
        }
        if acceptance_rule not in supported_rules:
            raise ValueError(
                "NemotronLabsDiffusionModel supports acceptance_rule values "
                f"{sorted(supported_rules)}, got {acceptance_rule!r}"
            )
        if acceptance_rule in {
            "asd",
            "cleanNSD",
            "cactus",
            "mars",
            "rejection_sampling",
            "rejection_sampling_then_target_top_p",
        } and self.threshold != 0.0:
            raise ValueError(
                "Nemotron sampled verification requires "
                "nemotron_linear_spec_threshold=0.0 so the sampled draft "
                "distribution is available to the verifier"
            )

        model = AutoModel.from_pretrained(
            self.args.target_name_or_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(self.device).eval()
        if self.use_lora:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "LoRA-enhanced Linear Self-Speculation requires PEFT. "
                    "Install it with `pip install peft`."
                ) from exc

            peft_model = PeftModel.from_pretrained(
                model,
                self.args.target_name_or_path,
                subfolder=self.lora_subfolder,
            ).to(self.device).eval()
            model = peft_model.model
            self.peft_model = peft_model

            lora_module_count = sum(
                1
                for module in model.modules()
                if hasattr(module, "lora_A") and len(module.lora_A) > 0
            )
            if lora_module_count == 0:
                raise RuntimeError(
                    "PEFT loaded without injecting any LoRA layers from "
                    f"{self.lora_subfolder!r}"
                )
            self.lora_module_count = lora_module_count

        tokenizer = AutoTokenizer.from_pretrained(
            self.args.target_name_or_path,
            trust_remote_code=True,
        )
        return model, model, tokenizer

    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,
        stop_token_ids: list[int] | None,
        dataset_name: str | None = None,
        sample_index: int | None = None,
    ) -> SimpleNamespace:
        if int(self.args.max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.replay_prefix_token_ids_for_sample(sample_index=sample_index) is not None:
            raise ValueError("Prefix replay is not supported by Linear Self-Speculation")
        if any(
            value is not None
            for value in self.generation_intervention_kwargs(
                sample_index=sample_index
            ).values()
        ):
            raise ValueError(
                "Forced-rejection interventions are not supported by "
                "Linear Self-Speculation"
            )

        eos_token_id = None if not stop_token_ids else int(stop_token_ids[0])
        asd_state = (ASDState(float(getattr(self.args, "asd_budget", 8.0)))
                     if self.args.acceptance_rule == "asd" else None)
        verification_phase_timer = (
            VerificationPhaseTimer(self.device)
            if self.should_record_verification_timing() else None
        )
        if verification_phase_timer is not None and self.args.acceptance_rule in {
            "nemotron_diffusion", "linear_self_speculation"
        }:
            raise ValueError(
                "Verification timing currently supports Nemotron SD/MARS/Cactus/"
                "cleanNSD sampled verification, not the official diffusion helpers"
            )
        if self.args.acceptance_rule == "nemotron_diffusion":
            if int(self.args.max_new_tokens) % self.block_length != 0:
                raise ValueError(
                    "Nemotron diffusion max_new_tokens must be divisible by "
                    f"block_length ({self.block_length})"
                )
            output_ids, nfe = self.target_model.generate(
                input_ids,
                max_new_tokens=int(self.args.max_new_tokens),
                block_length=self.block_length,
                threshold=self.threshold,
                temperature=float(self.args.temperature),
                eos_token_id=eos_token_id,
            )
            raw_acceptance_lengths = []
            target_top_p_rescue_count = 0
        elif self.args.acceptance_rule == "linear_self_speculation":
            output_ids, nfe, raw_acceptance_lengths = (
                run_linear_spec_with_acceptance_capture(
                    self.target_model,
                    input_ids,
                    max_new_tokens=int(self.args.max_new_tokens),
                    block_length=self.block_length,
                    temperature=float(self.args.temperature),
                    eos_token_id=eos_token_id,
                    threshold=self.threshold,
                    post_verify=self._post_verify,
                    dataset_name=dataset_name,
                    sample_index=sample_index,
                )
            )
            target_top_p_rescue_count = 0
        else:
            (
                output_ids,
                nfe,
                raw_acceptance_lengths,
                target_top_p_rescue_count,
            ) = run_linear_spec_rejection_sampling_then_target_top_p(
                self.target_model,
                input_ids,
                max_new_tokens=int(self.args.max_new_tokens),
                block_length=self.block_length,
                temperature=float(self.args.temperature),
                mask_token_id=None,
                eos_token_id=eos_token_id,
                acceptance_top_p=float(self.args.acceptance_top_p),
                target_sampling_top_p=float(self.args.target_sampling_top_p),
                enable_target_top_p_rescue=(
                    self.args.acceptance_rule in
                    {"cleanNSD", "rejection_sampling_then_target_top_p"}
                ),
                clean_nsd=self.args.acceptance_rule == "cleanNSD",
                asd_state=asd_state,
                asd_gate=float(getattr(self.args, "asd_gate", 0.25)),
                asd_max_exceptions=int(getattr(self.args, "asd_max_exceptions", 2)),
                asd_value_scheme=str(getattr(self.args, "asd_value_scheme", "suffix_length")),
                cactus_delta=(
                    float(self.args.cactus_delta)
                    if self.args.acceptance_rule == "cactus"
                    else None
                ),
                mars_relaxation_threshold=(
                    float(self.args.mars_relaxation_threshold)
                    if self.args.acceptance_rule == "mars"
                    else None
                ),
                post_verify=(None if self.args.acceptance_rule == "cleanNSD"
                             and self.token_diagnostics_recorder is None else self._post_verify),
                dataset_name=dataset_name,
                sample_index=sample_index,
                verification_phase_timer=verification_phase_timer,
            )

        num_input_tokens = int(input_ids.shape[1])
        output_ids = output_ids[:, : num_input_tokens + int(self.args.max_new_tokens)]
        output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)
        num_output_tokens = int(output_ids.shape[1]) - num_input_tokens
        acceptance_lengths = (
            []
            if self.args.acceptance_rule == "nemotron_diffusion"
            else normalize_acceptance_lengths(
                raw_acceptance_lengths,
                num_output_tokens=num_output_tokens,
            )
        )
        proposal_lengths = [self.max_proposal_tokens] * len(acceptance_lengths)
        accepted_draft_lengths = [
            min(self.max_proposal_tokens, max(0, accepted - 1))
            for accepted in acceptance_lengths
        ]

        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            nfe=nfe,
            target_top_p_rescue_count=target_top_p_rescue_count,
            acceptance_lengths=acceptance_lengths,
            proposal_lengths=proposal_lengths,
            accepted_draft_lengths=accepted_draft_lengths,
            blind_accept_prefix_lengths=[0] * len(acceptance_lengths),
            logical_draft_block_counts=[1] * len(acceptance_lengths),
            blind_accept_event_count=0,
            blind_accepted_token_count=0,
            verify_count=len(acceptance_lengths),
            replay_prefix_tokens=0,
            forced_reject_events=[],
            asd_statistics=asd_state.statistics() if asd_state is not None else None,
            risk_budget_rescue_count=0,
            risk_budget_reject_outside_top_p_count=0,
            risk_budget_reject_budget_count=0,
            verification_timing=(
                verification_phase_timer.summary()
                if verification_phase_timer is not None else None
            ),
        )
