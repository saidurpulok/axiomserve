"""Plot measured cases; never synthesize missing measurements."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", default="results/plots")
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    cases = [case for path in args.reports for case in json.loads(Path(path).read_text())["cases"]]
    quantities = {
        "tokens_per_second": "Output tokens / s",
        "ttft_p50_seconds": "Median TTFT (s)",
        "latency_p95_seconds": "p95 request latency (s)",
        "peak_gpu_mib": "Peak allocated GPU memory (MiB)",
    }
    for case in cases:
        case["peak_gpu_mib"] = case["memory_bytes"]["peak_allocated"] / 2**20
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    for key, label in quantities.items():
        fig, axis = plt.subplots(figsize=(5.5, 3.5))
        for mode in sorted({case["mode"] for case in cases}):
            rows = sorted((c for c in cases if c["mode"] == mode), key=lambda c: c["concurrency"])
            axis.plot(
                [c["concurrency"] for c in rows], [c[key] for c in rows], marker="o", label=mode
            )
        axis.set(xlabel="Client concurrency", ylabel=label)
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
        for extension in ["pdf", "png"]:
            fig.savefig(target / f"{key}.{extension}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
