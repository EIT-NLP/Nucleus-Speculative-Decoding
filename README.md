# Nucleus Speculative Decoding

Official reproduction code for **Nucleus Speculative Decoding (NSD): Plausibility-Aware Verification Beyond Exact Distribution**.

Paper: [arXiv:2610.07822](https://arxiv.org/abs/2610.07822)

NSD extends standard speculative decoding with a target-nucleus rescue rule. A draft token is accepted when it passes the standard stochastic acceptance test or belongs to the target model's nucleus. The implementation reuses probabilities computed during target verification and does not require another target-model forward pass.

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

## Citation

If this repository is useful in your research, please cite:

```bibtex
@article{li2026nucleus,
  title         = {Nucleus Speculative Decoding: Plausibility-Aware Verification Beyond Exact Distribution},
  author        = {Li, Shuhao and Ye, Fanghua and Lin, Wanyu and Yuan, Tianyu and Shen, Xiaoyu},
  journal       = {arXiv preprint arXiv:2610.07822},
  year          = {2026},
  eprint        = {2610.07822},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.07822}
}
```

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for details.
