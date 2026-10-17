from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
)

from deepspec.eval.base_evaluator import BaseEvaluator, trim_output_ids
from deepspec.eval.verification_timing import VerificationPhaseTimer
from deepspec.utils.sampling import logits_to_probs, sample_from_probs


NEMOTRON_ARCHITECTURE = "NemotronLabsDiffusionModel"


def _empty_decoding_metrics(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        acceptance_lengths=[],
        proposal_lengths=[],
        accepted_draft_lengths=[],
        blind_accept_prefix_lengths=[],
        logical_draft_block_counts=[],
        blind_accept_event_count=0,
        blind_accepted_token_count=0,
        verify_count=0,
        replay_prefix_tokens=0,
        forced_reject_events=[],
        risk_budget_rescue_count=0,
        risk_budget_reject_outside_top_p_count=0,
        risk_budget_reject_budget_count=0,
        **kwargs,
    )


@torch.no_grad()
def run_nemotron_autoregressive(
    model,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int | None,
    forward_timer: VerificationPhaseTimer | None = None,
) -> tuple[torch.Tensor, int]:
    """NVIDIA's AR loop with optional nucleus sampling added at the sampler."""
    for layer in model.encoder.layers:
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = False

    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)

    device = prompt_ids.device
    batch_size, prompt_length = prompt_ids.shape
    past_key_values = DynamicCache()
    cache_position = torch.arange(prompt_length, device=device)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

    encoder_output = model.encoder(
        input_ids=prompt_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
    )
    past_key_values = encoder_output.past_key_values
    next_logits = model.diffusion_head(
        encoder_output.last_hidden_state[:, -1:, :]
    )

    if forward_timer is not None:
        forward_timer.start_generation()

    generated_tokens: list[torch.Tensor] = []
    for step in range(int(max_new_tokens)):
        next_token = sample_from_probs(
            logits_to_probs(
                next_logits,
                float(temperature),
                top_p=float(top_p),
            )
        )
        generated_tokens.append(next_token)
        if eos_token_id is not None and (next_token == int(eos_token_id)).all():
            break

        if step < int(max_new_tokens) - 1:
            current_position = prompt_length + step
            step_cache_position = torch.tensor([current_position], device=device)
            step_position_ids = step_cache_position.unsqueeze(0).expand(
                batch_size, -1
            )
            if forward_timer is not None:
                forward_timer.start_target_model()
            encoder_output = model.encoder(
                input_ids=next_token,
                position_ids=step_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=step_cache_position,
            )
            past_key_values = encoder_output.past_key_values
            next_logits = model.diffusion_head(
                encoder_output.last_hidden_state[:, -1:, :]
            )
            if forward_timer is not None:
                forward_timer.stop_target_model(count_as_call=True)

    if forward_timer is not None:
        forward_timer.stop_generation()
    if not generated_tokens:
        return prompt_ids, 0
    output_ids = torch.cat([prompt_ids, *generated_tokens], dim=1)
    return output_ids, len(generated_tokens)


class AutoregressiveEvaluator(BaseEvaluator):
    """Target-only autoregressive baseline with matched benchmark artifacts."""

    @property
    def max_proposal_tokens(self) -> int:


        return 1

    def build_models(self):
        if self.args.target_name_or_path != self.args.draft_name_or_path:
            raise ValueError(
                "Autoregressive evaluation is target-only; target and draft "
                "paths must be identical."
            )
        config = AutoConfig.from_pretrained(
            self.args.target_name_or_path,
            trust_remote_code=True,
        )
        architecture = (getattr(config, "architectures", None) or [""])[0]
        self.is_nemotron = architecture == NEMOTRON_ARCHITECTURE
        ar_top_k = getattr(self.args, "autoregressive_top_k", None)
        if ar_top_k is not None and int(ar_top_k) < 0:
            raise ValueError("autoregressive_top_k must be nonnegative")
        if self.is_nemotron and ar_top_k not in (None, 0):
            raise ValueError("Nemotron's custom AR loop supports autoregressive_top_k=0 only")
        loader = AutoModel if self.is_nemotron else AutoModelForCausalLM
        model = loader.from_pretrained(
            self.args.target_name_or_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(self.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.target_name_or_path,
            trust_remote_code=True,
        )
        return model, model, tokenizer

    @staticmethod
    def _generate_with_forward_timing(model, timer: VerificationPhaseTimer, *args, **kwargs):
        """Time decode forwards inside ``generate``, excluding its prefill."""
        first_forward = True

        def before_forward(_module, _args, _kwargs):
            if not first_forward:
                timer.start_target_model()

        def after_forward(_module, _args, _kwargs, _output):
            nonlocal first_forward
            if first_forward:
                first_forward = False
                timer.start_generation()
            else:
                timer.stop_target_model(count_as_call=True)

        pre_hook = model.register_forward_pre_hook(before_forward, with_kwargs=True)
        post_hook = model.register_forward_hook(after_forward, with_kwargs=True)
        try:
            output = model.generate(*args, **kwargs)
            timer.stop_generation()
            return output
        finally:
            pre_hook.remove()
            post_hook.remove()

    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,
        stop_token_ids: list[int] | None,
        dataset_name: str | None = None,
        sample_index: int | None = None,
    ) -> SimpleNamespace:
        if self.replay_prefix_token_ids_for_sample(sample_index=sample_index) is not None:
            raise ValueError("Prefix replay is not supported by AR baselines")
        if any(
            value is not None
            for value in self.generation_intervention_kwargs(
                sample_index=sample_index
            ).values()
        ):
            raise ValueError("Forced-rejection controls do not apply to AR baselines")

        temperature = float(self.args.temperature)
        top_p = float(self.args.target_sampling_top_p)
        eos_token_id = None if not stop_token_ids else int(stop_token_ids[0])
        num_input_tokens = int(input_ids.shape[1])
        forward_timer = (
            VerificationPhaseTimer(self.device)
            if self.should_record_verification_timing() else None
        )

        if self.is_nemotron:
            output_ids, nfe = run_nemotron_autoregressive(
                self.target_model,
                input_ids,
                max_new_tokens=int(self.args.max_new_tokens),
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_token_id,
                forward_timer=forward_timer,
            )
        else:
            generation_kwargs = {
                "max_new_tokens": int(self.args.max_new_tokens),
                "do_sample": temperature > 0.0,
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id,
            }
            if stop_token_ids:
                generation_kwargs["eos_token_id"] = stop_token_ids
            if temperature > 0.0:
                generation_kwargs.update(temperature=temperature, top_p=top_p)
                ar_top_k = getattr(self.args, "autoregressive_top_k", None)
                if ar_top_k is not None:
                    generation_kwargs["top_k"] = int(ar_top_k)
            if forward_timer is None:
                output_ids = self.target_model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    **generation_kwargs,
                )
            else:
                output_ids = self._generate_with_forward_timing(
                    self.target_model,
                    forward_timer,
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    **generation_kwargs,
                )
            nfe = int(output_ids.shape[1]) - num_input_tokens

        output_ids = output_ids[:, : num_input_tokens + int(self.args.max_new_tokens)]
        output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)
        return _empty_decoding_metrics(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=int(output_ids.shape[1]) - num_input_tokens,
            nfe=nfe,
            verification_timing=(
                forward_timer.summary() if forward_timer is not None else None
            ),
        )
