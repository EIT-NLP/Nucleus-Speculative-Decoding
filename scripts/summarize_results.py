
from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = ("AR", "SD", "NSD")
DATASETS = ("gsm8k", "math500", "aime24", "aime25", "aime26", "humaneval", "mbpp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for method in METHODS:
        for dataset in DATASETS:
            root = args.result_root / method / dataset / "seed_22"
            quality = json.loads((root / "quality_metrics.json").read_text())["datasets"][dataset]
            timing = json.loads((root / "throughput_metrics.json").read_text())["datasets"][dataset]
            accuracy = quality.get("accuracy_percent", quality.get("pass_at_1_percent"))
            rows.append({
                "method": method,
                "dataset": dataset,
                "samples": int(quality["num_samples"]),
                "accuracy_percent": float(accuracy),
                "throughput_tokens_per_second": float(timing["throughput_tokens_per_second"]),
                "mean_accept_length": None if method == "AR" else timing.get("acceptance_length"),
            })

    payload = {
        "model": "nvidia/Nemotron-Labs-Diffusion-14B",
        "seed": 22,
        "max_new_tokens": 32768,
        "temperature": 1.0,
        "target_sampling_top_p": 1.0,
        "acceptance_top_p": 0.95,
        "block_length": 32,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
