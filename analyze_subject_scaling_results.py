from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_results_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prepared subject-scaling experiment outputs.")
    parser.add_argument("--experiment_dir", type=str, default="experiments/subject_scaling")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    manifest_path = exp_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = read_manifest(manifest_path)

    run_summaries: list[dict[str, str]] = []
    for row in manifest:
        results_path = Path(row["run_dir"]) / "results.csv"
        if not results_path.exists():
            continue

        subject_rows = read_results_csv(results_path)
        if not subject_rows:
            continue

        test_acc = [float(r["test_accuracy"]) for r in subject_rows]
        run_summaries.append(
            {
                "experiment_id": row["experiment_id"],
                "fraction": row["fraction"],
                "repeat": row["repeat"],
                "num_subjects": row["num_subjects"],
                "mean_test_accuracy": f"{np.mean(test_acc):.6f}",
                "std_test_accuracy": f"{np.std(test_acc):.6f}",
            }
        )

    if not run_summaries:
        print("No completed experiment results found yet.")
        print(f"Expected per-run results at: {exp_dir / 'runs' / '<experiment_id>' / 'results.csv'}")
        return

    summary_path = exp_dir / "scaling_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment_id", "fraction", "repeat", "num_subjects", "mean_test_accuracy", "std_test_accuracy"],
        )
        writer.writeheader()
        writer.writerows(run_summaries)

    grouped: dict[float, list[float]] = {}
    for row in run_summaries:
        frac = float(row["fraction"])
        grouped.setdefault(frac, []).append(float(row["mean_test_accuracy"]))

    xs = sorted(grouped.keys())
    ys = [float(np.mean(grouped[x])) for x in xs]
    yerr = [float(np.std(grouped[x])) for x in xs]

    plt.figure(figsize=(9, 5.5))
    plt.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2, capsize=4)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Subject Fraction")
    plt.ylabel("Mean Test Accuracy")
    plt.title("Scaling Curve: Accuracy vs Subject Fraction")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = exp_dir / "accuracy_vs_subject_fraction.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()

    print("=== Scaling Analysis Complete ===")
    print(f"Run-level summary: {summary_path}")
    print(f"Scaling plot: {plot_path}")


if __name__ == "__main__":
    main()
