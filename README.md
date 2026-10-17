# Nucleus Speculative Decoding

Official reproduction code for **Nucleus Speculative Decoding (NSD): Plausibility-Aware Verification Beyond Exact Distribution**.

NSD extends standard speculative decoding with a target-nucleus rescue rule. A draft token is accepted when it passes the standard stochastic acceptance test or belongs to the target model's nucleus. The implementation reuses probabilities computed during target verification and does not require another target-model forward pass.

This artifact reproduces the Nemotron-14B, seed-22 experiment with:

- autoregressive decoding (AR);
- official Linear Self-Speculation (SD); and
- Nucleus Speculative Decoding (NSD), with target Top-P threshold 0.95.

## Reproduction configuration

| Item | Value |
|---|---|
| Model | `nvidia/Nemotron-Labs-Diffusion-14B` |
| Model revision | `f8c3e2c078e193599b8882d965b1001c456ba738` |
| Seed | 22 |
| Dataset seed | 20260721 |
| Temperature | 1.0 |
| Target sampling Top-P | 1.0 |
| NSD acceptance Top-P | 0.95 |
| Linear-speculation block length | 32 |
| Maximum new tokens | 32768 |
| Token diagnostics | Disabled |

The included full evaluation sets are:

| Benchmark | Samples |
|---|---:|
| GSM8K | 1,319 |
| MATH500 | 500 |
| AIME24 | 30 |
| AIME25 | 30 |
| AIME26 | 30 |
| HumanEval | 164 |
| MBPP | 257 |

Dataset hashes are recorded in [`eval_datasets/SHA256SUMS`](eval_datasets/SHA256SUMS).

## Installation

Use Linux with an NVIDIA GPU and a CUDA-compatible Python environment. Python 3.11 is recommended.

```bash
git clone https://github.com/EIT-NLP/Nucleus-Speculative-Decoding.git
cd Nucleus-Speculative-Decoding

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for the local CUDA runtime if the default wheel in `requirements.txt` is unsuitable. Functional scoring for HumanEval and MBPP requires Linux `bubblewrap` at `/usr/bin/bwrap`.

The model is downloaded automatically from Hugging Face unless `MODEL` points to a local model directory. A local copy must contain the `linear_spec_lora` subdirectory used by SD and NSD.

## Smoke test

The smoke test evaluates one AIME25 example with AR, SD, and NSD and generates at most 16 tokens per method.

```bash
bash scripts/smoke_test.sh
```

To use a local checkpoint or another result directory:

```bash
MODEL=/path/to/Nemotron-Labs-Diffusion-14B \
RESULT_ROOT=/path/to/smoke-results \
bash scripts/smoke_test.sh
```

## Full reproduction

Run every included sample for AR, SD, and NSD:

```bash
bash scripts/run_nemotron14b_seed22.sh
```

Common overrides are available as environment variables:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL=/path/to/Nemotron-Labs-Diffusion-14B \
RESULT_ROOT=/path/to/results \
SKIP_COMPLETED=1 \
bash scripts/run_nemotron14b_seed22.sh
```

Completed dataset-method cells are skipped by default. Set `SKIP_COMPLETED=0` to rerun them.

## Outputs

Results are written under:

```text
results/nemotron14b_seed22/{AR,SD,NSD}/<dataset>/seed_22/
```

Each cell contains:

- `quality_metrics.json`: accuracy or pass@1;
- `throughput_metrics.json`: generated-token throughput and acceptance length;
- `artifacts/`: rank-local predictions and generation metadata; and
- `run.log`: the complete run log.

The runner also creates:

```text
results/nemotron14b_seed22/summary.json
```

This file collects accuracy, throughput, and mean acceptance length across methods and benchmarks.

## Citation

If this repository is useful in your research, please cite:

```bibtex
@inproceedings{anonymous2027nucleus,
  title     = {Nucleus Speculative Decoding: Plausibility-Aware Verification Beyond Exact Distribution},
  author    = {Anonymous Authors},
  booktitle = {International Conference on Learning Representations},
  year      = {2027}
}
```

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for details.
