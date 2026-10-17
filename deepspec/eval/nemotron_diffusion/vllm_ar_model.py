"""vLLM adapter for the causal path of Nemotron Labs Diffusion checkpoints.

The published checkpoints contain a standard causal Ministral-3 backbone, but
store it as ``encoder.*`` and store the shared output projection as
``diffusion_head.*``.  vLLM already supports ``Ministral3ForCausalLM`` through
its Mistral implementation.  This adapter only translates checkpoint names;
it does not change vLLM or the checkpoint on disk.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from vllm.model_executor.models.mistral import MistralForCausalLM


def remap_nemotron_diffusion_weight_name(name: str) -> str:
    if name.startswith("encoder."):
        return "model." + name.removeprefix("encoder.")
    if name.startswith("diffusion_head."):
        return "lm_head." + name.removeprefix("diffusion_head.")
    return name


class NemotronLabsDiffusionARForCausalLM(MistralForCausalLM):
    """Load the checkpoint's causal backbone with vLLM's Ministral-3 model."""

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        def remapped() -> Iterator[tuple[str, torch.Tensor]]:
            for name, value in weights:
                yield remap_nemotron_diffusion_weight_name(name), value

        return super().load_weights(remapped())


def register_nemotron_diffusion_ar_model() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "NemotronLabsDiffusionModel",
        (
            "deepspec.eval.nemotron_diffusion.vllm_ar_model:"
            "NemotronLabsDiffusionARForCausalLM"
        ),
    )
