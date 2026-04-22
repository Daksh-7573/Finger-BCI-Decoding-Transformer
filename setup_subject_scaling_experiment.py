from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from data_loader import discover_offline_movement_files


def parse_fractions(spec: str) -> list[float]:
    fractions: list[float] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"Invalid fraction {value}. Expected 0 < fraction <= 1.")
        fractions.append(value)

    if not fractions:
        raise ValueError("At least one fraction is required.")

    return sorted(set(fractions))


def discover_subject_ids(root_dir: str | Path) -> list[str]:
    files = discover_offline_movement_files(root_dir)
    if not files:
        raise FileNotFoundError("No OfflineMovement .mat files found.")

    subjects = sorted({fp.stem.split("_", 1)[0] if "_" in fp.stem else "UNKNOWN" for fp in files})
    return subjects


def build_command(
    python_exe: str,
    train_script: str,
    root_dir: str,
    subject_ids: list[str],
    save_dir: Path,
    epochs: int,
    batch_size: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    max_time_steps: int,
    low_memory: bool,
    weighted_loss: bool,
) -> str:
    parts = [
        f'& "{python_exe}"',
        train_script,
        f"--root_dir {root_dir}",
        "--per_subject",
        f'--subject_ids "{",".join(subject_ids)}"',
        f"--epochs {epochs}",
        f"--batch_size {batch_size}",
        f"--d_model {d_model}",
        f"--nhead {nhead}",
        f"--num_layers {num_layers}",
        f"--dim_feedforward {dim_feedforward}",
        f"--max_time_steps {max_time_steps}",
        f'--save_dir "{save_dir.as_posix()}"',
    ]

    if low_memory:
        parts.append("--low_memory")
    if weighted_loss:
        parts.append("--weighted_loss")

    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare subject-scaling experiment commands without running training.")
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="experiments/subject_scaling")
    parser.add_argument("--fractions", type=str, default="0.25,0.5,0.75,1.0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--python_exe", type=str, default="d:/EEG Dataset/.venv/Scripts/python.exe")
    parser.add_argument("--train_script", type=str, default="train.py")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dim_feedforward", type=int, default=64)
    parser.add_argument("--max_time_steps", type=int, default=1024)
    parser.add_argument(
        "--no_low_memory",
        action="store_true",
        help="Disable --low_memory in generated train commands",
    )
    parser.add_argument(
        "--no_weighted_loss",
        action="store_true",
        help="Disable --weighted_loss in generated train commands",
    )

    args = parser.parse_args()

    fractions = parse_fractions(args.fractions)
    low_memory = not args.no_low_memory
    weighted_loss = not args.no_weighted_loss
    subjects = discover_subject_ids(args.root_dir)
    n_subjects = len(subjects)

    if args.repeats < 1:
        raise ValueError("repeats must be >= 1")

    output_dir = Path(args.output_dir)
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    manifest_rows: list[dict[str, str]] = []
    commands: list[str] = []

    for fraction in fractions:
        k = max(1, int(round(n_subjects * fraction)))
        for repeat in range(1, args.repeats + 1):
            chosen = sorted(rng.choice(subjects, size=k, replace=False).tolist())
            exp_id = f"f{int(round(fraction * 100)):03d}_r{repeat:02d}_n{k:02d}"
            exp_dir = runs_dir / exp_id

            cmd = build_command(
                python_exe=args.python_exe,
                train_script=args.train_script,
                root_dir=args.root_dir,
                subject_ids=chosen,
                save_dir=exp_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dim_feedforward=args.dim_feedforward,
                max_time_steps=args.max_time_steps,
                low_memory=low_memory,
                weighted_loss=weighted_loss,
            )

            manifest_rows.append(
                {
                    "experiment_id": exp_id,
                    "fraction": f"{fraction:.4f}",
                    "repeat": str(repeat),
                    "num_subjects": str(k),
                    "subject_ids": ",".join(chosen),
                    "run_dir": exp_dir.as_posix(),
                }
            )
            commands.append(cmd)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment_id", "fraction", "repeat", "num_subjects", "subject_ids", "run_dir"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    ps1_path = output_dir / "run_experiments.ps1"
    with ps1_path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated subject-scaling experiment commands\n")
        f.write("# This script only contains commands; nothing has been executed yet.\n\n")
        for cmd in commands:
            f.write(cmd + "\n")

    print("=== Experiment Setup Complete ===")
    print(f"Detected subjects: {subjects}")
    print(f"Total subjects: {n_subjects}")
    print(f"Manifest: {manifest_path}")
    print(f"PowerShell commands: {ps1_path}")
    print(f"Total experiment runs prepared: {len(commands)}")


if __name__ == "__main__":
    main()
