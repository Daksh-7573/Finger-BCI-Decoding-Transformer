from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import signal
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm as mpl_cm
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

from data_loader import (
    TrialRef,
    build_trial_index_for_run,
    discover_movement_files,
    discover_offline_movement_files,
    detect_format,
    inspect_mat_file,
    load_all_movement_runs,
    load_all_offline_movement_runs,
)
from scipy.io import loadmat
from model import EEGTransformerClassifier
from preprocess import (
    build_dataloaders,
    combine_runs,
    relabel_to_zero,
    split_by_runs,
    split_by_subjects,
    split_random,
    zscore_per_trial_channel,
)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.long)
            logits = model(Xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * yb.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_count += yb.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    preds_np = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    targets_np = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    return avg_loss, acc, preds_np, targets_np


def parse_class_labels(spec: str) -> list[int]:
    if not spec.strip():
        return []
    labels: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        labels.append(int(token))
    return sorted(set(labels))


def parse_subject_ids(spec: str) -> set[str]:
    if not spec.strip():
        return set()
    items = set()
    for token in spec.split(","):
        token = token.strip()
        if token:
            items.add(token)
    return items


def subject_span(start: int, end: int) -> list[str]:
    return [f"S{i:02d}" for i in range(start, end + 1)]


def split_train_val_indices(n_samples: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to split train/val.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    n_val = max(1, int(round(n_samples * val_frac)))
    n_val = min(n_samples - 1, n_val)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def save_training_history_csv(
    path: str | Path,
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
) -> None:
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    n_epochs = max(len(train_losses), len(val_losses), len(train_accs), len(val_accs))
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        writer.writeheader()
        for i in range(n_epochs):
            writer.writerow(
                {
                    "epoch": i + 1,
                    "train_loss": f"{train_losses[i]:.8f}" if i < len(train_losses) else "",
                    "val_loss": f"{val_losses[i]:.8f}" if i < len(val_losses) else "",
                    "train_acc": f"{train_accs[i]:.8f}" if i < len(train_accs) else "",
                    "val_acc": f"{val_accs[i]:.8f}" if i < len(val_accs) else "",
                }
            )


def save_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    loss: float = 0.0,
    history: dict[str, list[float]] | None = None,
    best_val_acc: float | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "loss": float(loss),
    }
    if model is not None:
        payload["model_state_dict"] = model.state_dict()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if history is not None:
        payload["history"] = history
    if best_val_acc is not None:
        payload["best_val_acc"] = float(best_val_acc)

    torch.save(payload, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any] | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_confusion_matrix_plot(cm_array: np.ndarray, labels: list[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_array, interpolation="nearest", cmap=mpl_cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    threshold = cm_array.max() / 2.0 if cm_array.size else 0.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_array[i, j]}",
                ha="center",
                va="center",
                color="white" if cm_array[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def confusion_matrix_markdown_table(cm_array: np.ndarray, labels: list[int]) -> str:
    header = "| True \\ Pred | " + " | ".join(str(lbl) for lbl in labels) + " |"
    sep = "|---|" + "---|" * len(labels)
    rows = [header, sep]
    for i, lbl in enumerate(labels):
        row = "| " + str(lbl) + " | " + " | ".join(str(int(v)) for v in cm_array[i]) + " |"
        rows.append(row)
    return "\n".join(rows)


class LazyRunDataset(torch.utils.data.Dataset):
    """Dataset that reads trials from file-level trial references without building giant arrays."""

    def __init__(self, refs: list[TrialRef], label_map: dict[int, int], eps: float = 1e-6) -> None:
        self.refs = refs
        self.label_map = label_map
        self.eps = eps
        self.index: list[tuple[int, int]] = []
        self.labels: list[int] = []

        self._cache_file: str | None = None
        self._cache_fmt: str | None = None
        self._cache_data: np.ndarray | None = None
        self._cache_X: np.ndarray | None = None
        self.file_to_indices: dict[str, list[int]] = {}

        for ref_idx, ref in enumerate(refs):
            mapped = self.label_map.get(int(ref.raw_label))
            if mapped is None:
                continue
            dataset_idx = len(self.index)
            self.index.append((ref_idx, mapped))
            self.labels.append(mapped)
            if ref.file_path not in self.file_to_indices:
                self.file_to_indices[ref.file_path] = []
            self.file_to_indices[ref.file_path].append(dataset_idx)

        if not self.index:
            raise ValueError("No samples available after applying label mapping.")

    def __len__(self) -> int:
        return len(self.index)

    def _load_file_cache(self, file_path: str) -> None:
        if self._cache_file == file_path:
            return

        mat = loadmat(file_path, simplify_cells=True)
        fmt = detect_format(mat)
        key_map = {k.lower(): k for k in mat.keys()}

        if fmt == "FORMAT_A":
            eeg = mat[key_map["eeg"]]
            self._cache_data = np.asarray(eeg["data"])
            self._cache_X = None
        else:
            self._cache_X = np.asarray(mat[key_map["x"]])
            self._cache_data = None

        self._cache_file = file_path
        self._cache_fmt = fmt

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ref_idx, mapped_label = self.index[idx]
        ref = self.refs[ref_idx]

        self._load_file_cache(ref.file_path)

        if ref.fmt == "FORMAT_A":
            if self._cache_data is None or ref.start is None or ref.end is None:
                raise ValueError("Invalid FORMAT_A cache state or trial reference.")
            x = np.asarray(self._cache_data[:, ref.start : ref.end], dtype=np.float32)
        else:
            if self._cache_X is None or ref.trial_idx is None:
                raise ValueError("Invalid FORMAT_B cache state or trial reference.")
            x = np.asarray(self._cache_X[ref.trial_idx], dtype=np.float32)

        # Keep normalization per trial/channel to match previous preprocessing behavior.
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        x = (x - mean) / (std + self.eps)

        return torch.from_numpy(x.astype(np.float32, copy=False)), int(mapped_label)


class FileGroupedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Yield batches grouped by file to minimize expensive MAT reloads."""

    def __init__(
        self,
        indices_by_file: dict[str, list[int]],
        batch_size: int,
        shuffle_files: bool = True,
        shuffle_within_file: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        self.indices_by_file = {k: list(v) for k, v in indices_by_file.items() if v}
        self.batch_size = int(batch_size)
        self.shuffle_files = shuffle_files
        self.shuffle_within_file = shuffle_within_file
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not self.indices_by_file:
            raise ValueError("indices_by_file is empty")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        file_keys = list(self.indices_by_file.keys())
        if self.shuffle_files:
            rng.shuffle(file_keys)

        for fk in file_keys:
            idxs = list(self.indices_by_file[fk])
            if self.shuffle_within_file:
                rng.shuffle(idxs)

            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for idxs in self.indices_by_file.values():
            if self.drop_last:
                total += len(idxs) // self.batch_size
            else:
                total += math.ceil(len(idxs) / self.batch_size)
        return total


class PrecomputedTrialDataset(torch.utils.data.Dataset):
    """Dataset backed by precomputed per-file normalized numpy caches."""

    def __init__(self, samples: list[tuple[str, int, int, str]]) -> None:
        self.samples = samples
        self.labels = [int(s[2]) for s in samples]
        self.file_to_indices: dict[str, list[int]] = {}
        for i, (cache_path, _, _, _) in enumerate(samples):
            if cache_path not in self.file_to_indices:
                self.file_to_indices[cache_path] = []
            self.file_to_indices[cache_path].append(i)

        self._cache_file: str | None = None
        self._cache_arr: np.ndarray | None = None

        if not self.samples:
            raise ValueError("No precomputed samples available.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_cache(self, cache_path: str) -> None:
        if self._cache_file == cache_path:
            return
        self._cache_arr = np.load(cache_path, mmap_mode="r")
        self._cache_file = cache_path

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        cache_path, trial_idx, mapped_label, _ = self.samples[idx]
        self._load_cache(cache_path)
        if self._cache_arr is None:
            raise ValueError("Cache array is not loaded.")
        x = np.asarray(self._cache_arr[trial_idx], dtype=np.float32)
        return torch.from_numpy(x), int(mapped_label)


def _cache_stem(file_path: str, window_sec: float, dtype_name: str) -> str:
    key = f"{Path(file_path).resolve().as_posix()}|window={window_sec:.6f}|dtype={dtype_name}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{Path(file_path).stem}_{digest}"


def build_precomputed_cache_samples(
    refs: list[TrialRef],
    label_map: dict[int, int],
    cache_dir: Path,
    window_sec: float,
    cache_dtype: np.dtype,
    rebuild_cache: bool,
    split_name: str = "",
) -> tuple[list[tuple[str, int, int, str]], tuple[int, int], int, int]:
    """Create/reuse per-file caches and return dataset-ready sample references."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    refs_by_file: dict[str, list[TrialRef]] = {}
    for ref in refs:
        refs_by_file.setdefault(ref.file_path, []).append(ref)

    samples: list[tuple[str, int, int, str]] = []
    created_files = 0
    reused_files = 0
    sample_shape: tuple[int, int] | None = None
    dtype_name = np.dtype(cache_dtype).name

    for file_path in sorted(refs_by_file):
        refs_for_file = refs_by_file[file_path]
        kept: list[tuple[TrialRef, int]] = []
        for ref in refs_for_file:
            mapped = label_map.get(int(ref.raw_label))
            if mapped is not None:
                kept.append((ref, mapped))
        if not kept:
            continue

        source_fp = Path(file_path)
        stem = _cache_stem(file_path, window_sec, dtype_name)
        cache_fp = cache_dir / f"{stem}.npy"
        meta_fp = cache_dir / f"{stem}.json"

        expected_mtime_ns = source_fp.stat().st_mtime_ns
        expected_size = source_fp.stat().st_size
        can_reuse = False

        if not rebuild_cache and cache_fp.exists() and meta_fp.exists():
            try:
                meta = json.loads(meta_fp.read_text(encoding="utf-8"))
                can_reuse = (
                    meta.get("source_path") == str(source_fp.resolve())
                    and int(meta.get("source_mtime_ns", -1)) == int(expected_mtime_ns)
                    and int(meta.get("source_size", -1)) == int(expected_size)
                    and float(meta.get("window_sec", -1.0)) == float(window_sec)
                    and meta.get("dtype") == dtype_name
                    and int(meta.get("n_trials", -1)) == len(kept)
                )
            except Exception:
                can_reuse = False

        if not can_reuse:
            mat = loadmat(source_fp, simplify_cells=True)
            fmt = detect_format(mat)

            if fmt == "FORMAT_A":
                key_map = {k.lower(): k for k in mat.keys()}
                eeg = mat[key_map["eeg"]]
                data = np.asarray(eeg["data"], dtype=np.float32)
                if kept[0][0].start is None or kept[0][0].end is None:
                    raise ValueError(f"Invalid FORMAT_A reference for cache build: {file_path}")
                window_samples = int(kept[0][0].end - kept[0][0].start)
                X = np.empty((len(kept), data.shape[0], window_samples), dtype=np.float32)
                for i, (ref, _) in enumerate(kept):
                    if ref.start is None or ref.end is None:
                        raise ValueError(f"Missing FORMAT_A boundaries in cache build: {file_path}")
                    X[i] = data[:, ref.start : ref.end]
            else:
                key_map = {k.lower(): k for k in mat.keys()}
                X_src = np.asarray(mat[key_map["x"]], dtype=np.float32)
                if kept[0][0].trial_idx is None:
                    raise ValueError(f"Invalid FORMAT_B trial index for cache build: {file_path}")
                first_trial = X_src[int(kept[0][0].trial_idx)]
                X = np.empty((len(kept), first_trial.shape[0], first_trial.shape[1]), dtype=np.float32)
                for i, (ref, _) in enumerate(kept):
                    if ref.trial_idx is None:
                        raise ValueError(f"Missing FORMAT_B trial index in cache build: {file_path}")
                    X[i] = X_src[int(ref.trial_idx)]

            mean = X.mean(axis=-1, keepdims=True)
            std = X.std(axis=-1, keepdims=True)
            X = ((X - mean) / (std + 1e-6)).astype(cache_dtype, copy=False)

            np.save(cache_fp, X)
            meta_fp.write_text(
                json.dumps(
                    {
                        "source_path": str(source_fp.resolve()),
                        "source_mtime_ns": int(expected_mtime_ns),
                        "source_size": int(expected_size),
                        "window_sec": float(window_sec),
                        "dtype": dtype_name,
                        "n_trials": int(len(kept)),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            created_files += 1
            if split_name.lower() == "train":
                print(
                    f"[Global Train] Completed file: {source_fp.name} | "
                    f"trials={len(kept)} | cache=created"
                )
        else:
            reused_files += 1
            if split_name.lower() == "train":
                print(
                    f"[Global Train] Completed file: {source_fp.name} | "
                    f"trials={len(kept)} | cache=reused"
                )

        arr = np.load(cache_fp, mmap_mode="r")
        if sample_shape is None:
            sample_shape = (int(arr.shape[1]), int(arr.shape[2]))

        cache_path_str = str(cache_fp)
        for local_idx, (ref, mapped_label) in enumerate(kept):
            samples.append((cache_path_str, local_idx, int(mapped_label), ref.subject_id))

    if not samples or sample_shape is None:
        raise ValueError("No cached samples were prepared.")

    return samples, sample_shape, created_files, reused_files


def train_cross_subject_global(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_subjects = set(subject_span(args.train_subject_start, args.train_subject_end))
    test_subjects = set(subject_span(args.test_subject_start, args.test_subject_end))
    overlap = train_subjects.intersection(test_subjects)
    if overlap:
        raise ValueError(f"Subject leakage detected. Overlap: {sorted(overlap)}")

    available_files = discover_movement_files(
        args.root_dir,
        include_offline=True,
        include_online=True,
    )
    available_subjects = sorted({fp.stem.split("_", 1)[0] if "_" in fp.stem else "UNKNOWN" for fp in available_files})
    train_subjects_present = sorted(train_subjects.intersection(available_subjects))
    test_subjects_present = sorted(test_subjects.intersection(available_subjects))

    if not train_subjects_present:
        raise ValueError("No training subjects found for the requested range.")
    if not test_subjects_present:
        raise ValueError("No test subjects found for the requested range.")

    print("\n=== Cross-Subject Split ===")
    print(f"Train subjects requested: {sorted(train_subjects)}")
    print(f"Test subjects requested: {sorted(test_subjects)}")
    print(f"Train subjects present: {train_subjects_present}")
    print(f"Test subjects present: {test_subjects_present}")
    print("===========================\n")

    include_online_movement = not args.cross_subject_offline_only

    train_files = discover_movement_files(
        args.root_dir,
        subject_ids=set(train_subjects_present),
        include_offline=True,
        include_online=include_online_movement,
    )
    test_files = discover_movement_files(
        args.root_dir,
        subject_ids=set(test_subjects_present),
        include_offline=True,
        include_online=include_online_movement,
    )

    if not train_files:
        raise ValueError("No movement files found for train subjects.")
    if not test_files:
        raise ValueError("No movement files found for test subjects.")

    train_refs: list[TrialRef] = []
    test_refs: list[TrialRef] = []
    train_shape: tuple[int, int] | None = None

    for fp in train_files:
        refs, sample_shape = build_trial_index_for_run(fp, window_sec=args.window_sec, print_debug=True)
        train_refs.extend(refs)
        if train_shape is None:
            train_shape = sample_shape

    for fp in test_files:
        refs, _ = build_trial_index_for_run(fp, window_sec=args.window_sec, print_debug=True)
        test_refs.extend(refs)

    if not train_refs:
        raise ValueError("No training trials indexed from train files.")
    if not test_refs:
        raise ValueError("No test trials indexed from test files.")

    def print_trials_per_subject(refs: list[TrialRef], title: str) -> None:
        counts: dict[str, int] = {}
        for ref in refs:
            counts[ref.subject_id] = counts.get(ref.subject_id, 0) + 1
        print(f"\n=== {title} Trials Per Subject ===")
        for sid in sorted(counts):
            print(f"{sid}: {counts[sid]}")
        print("==================================\n")

    print_trials_per_subject(train_refs, "Training")
    print_trials_per_subject(test_refs, "Test")

    train_labels_raw = np.array([int(ref.raw_label) for ref in train_refs], dtype=np.int64)
    unique_train_labels = sorted(np.unique(train_labels_raw).tolist())
    label_map = {int(old): int(new) for new, old in enumerate(unique_train_labels)}
    inv_label_map = {new: old for old, new in label_map.items()}

    train_idx, val_idx = split_train_val_indices(len(train_refs), args.val_frac_global, args.seed)
    train_ref_split = [train_refs[int(i)] for i in train_idx.tolist()]
    val_ref_split = [train_refs[int(i)] for i in val_idx.tolist()]

    cache_dtype = np.float16 if args.low_memory else np.float32
    cache_root = Path(args.cross_subject_cache_dir)
    if not cache_root.is_absolute():
        cache_root = Path(args.root_dir) / cache_root

    train_samples, train_shape, train_created, train_reused = build_precomputed_cache_samples(
        train_ref_split,
        label_map,
        cache_root / "train",
        window_sec=args.window_sec,
        cache_dtype=cache_dtype,
        rebuild_cache=args.rebuild_cross_subject_cache,
        split_name="train",
    )
    val_samples, val_shape, val_created, val_reused = build_precomputed_cache_samples(
        val_ref_split,
        label_map,
        cache_root / "val",
        window_sec=args.window_sec,
        cache_dtype=cache_dtype,
        rebuild_cache=args.rebuild_cross_subject_cache,
        split_name="val",
    )
    test_samples, test_shape, test_created, test_reused = build_precomputed_cache_samples(
        test_refs,
        label_map,
        cache_root / "test",
        window_sec=args.window_sec,
        cache_dtype=cache_dtype,
        rebuild_cache=args.rebuild_cross_subject_cache,
        split_name="test",
    )

    train_dataset = PrecomputedTrialDataset(train_samples)
    val_dataset = PrecomputedTrialDataset(val_samples)
    test_dataset = PrecomputedTrialDataset(test_samples)

    train_batch_sampler = FileGroupedBatchSampler(
        train_dataset.file_to_indices,
        batch_size=args.batch_size,
        shuffle_files=True,
        shuffle_within_file=True,
        drop_last=False,
        seed=args.seed,
    )

    effective_workers = max(0, int(args.num_workers))
    if os.name == "nt" and args.low_memory and effective_workers > 0:
        print(
            "[Info] Windows low-memory mode detected; forcing num_workers=0 "
            "to avoid multiprocessing DLL/page-file errors."
        )
        effective_workers = 0
    pin_memory = device.type == "cuda"
    loader_kwargs: dict[str, Any] = {}
    if effective_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    print("\n=== Dataset Sizes ===")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Sample trial shape (train cache): {train_shape}")
    print(f"Sample trial shape (val cache): {val_shape}")
    print(f"Sample trial shape (test cache): {test_shape}")
    sample_x, _ = train_dataset[0]
    print(f"Unique train labels: {sorted(set(train_dataset.labels))}")
    print(f"Unique test labels: {sorted(set(test_dataset.labels))}")
    print("=====================\n")

    print("=== Cache Summary ===")
    print(f"Cache root: {cache_root}")
    print(f"Train cache files -> created: {train_created}, reused: {train_reused}")
    print(f"Val cache files -> created: {val_created}, reused: {val_reused}")
    print(f"Test cache files -> created: {test_created}, reused: {test_reused}")
    print("=====================")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )

    num_classes = len(label_map)
    in_channels = int(sample_x.shape[0])
    time_len = int(sample_x.shape[1])

    model = EEGTransformerClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_time_steps=max(time_len, args.max_time_steps),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(args.root_dir) / checkpoint_path

    history_path = Path(args.training_history_path)
    if not history_path.is_absolute():
        history_path = Path(args.root_dir) / history_path

    best_model_path = Path(args.best_model_path)
    if not best_model_path.is_absolute():
        best_model_path = Path(args.root_dir) / best_model_path

    best_val_acc = -1.0
    best_state = None
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    start_epoch = 1
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer, map_location=device)
        if checkpoint is not None:
            last_epoch = int(checkpoint.get("epoch", 0))
            start_epoch = last_epoch + 1
            best_val_acc = float(checkpoint.get("best_val_acc", -1.0))
            history = checkpoint.get("history", {})
            train_losses = [float(v) for v in history.get("train_loss", [])]
            val_losses = [float(v) for v in history.get("val_loss", [])]
            train_accs = [float(v) for v in history.get("train_acc", [])]
            val_accs = [float(v) for v in history.get("val_acc", [])]
            print(f"Resuming from epoch {last_epoch}")

    # Optional manual checkpoint save at startup.
    if args.manual_checkpoint_path:
        manual_path = Path(args.manual_checkpoint_path)
        if not manual_path.is_absolute():
            manual_path = Path(args.root_dir) / manual_path
        save_checkpoint(
            manual_path,
            model=model,
            optimizer=optimizer,
            epoch=max(0, start_epoch - 1),
            loss=float(train_losses[-1]) if train_losses else 0.0,
            history={
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_acc": train_accs,
                "val_acc": val_accs,
            },
            best_val_acc=best_val_acc,
        )

    print("\n=== Global Training Start ===")
    interrupted = {"flag": False}

    def _handle_interrupt(sig, frame):
        interrupted["flag"] = True
        print("\nKeyboard interrupt received. Finishing current batch and saving checkpoint...")

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_interrupt)

    last_epoch_train_loss = float(train_losses[-1]) if train_losses else 0.0
    completed_epoch = max(0, start_epoch - 1)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_batch_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_count = 0

            for Xb, yb in train_loader:
                if interrupted["flag"]:
                    raise KeyboardInterrupt

                Xb = Xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)
                logits = model(Xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * yb.size(0)
                running_correct += (torch.argmax(logits, dim=1) == yb).sum().item()
                running_count += yb.size(0)

            epoch_train_loss = running_loss / max(1, running_count)
            epoch_train_acc = running_correct / max(1, running_count)
            epoch_val_loss, epoch_val_acc, _, _ = evaluate(model, val_loader, criterion, device)

            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            train_accs.append(epoch_train_acc)
            val_accs.append(epoch_val_acc)
            completed_epoch = epoch
            last_epoch_train_loss = epoch_train_loss

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if args.save_best_model:
                    best_model_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "epoch": int(epoch),
                            "val_acc": float(epoch_val_acc),
                            "args": vars(args),
                        },
                        best_model_path,
                    )

            history_payload = {
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_acc": train_accs,
                "val_acc": val_accs,
            }
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=epoch_train_loss,
                history=history_payload,
                best_val_acc=best_val_acc,
            )

            if args.save_epoch_checkpoints:
                epoch_ckpt = checkpoint_path.with_name(f"{checkpoint_path.stem}_epoch_{epoch:03d}{checkpoint_path.suffix}")
                save_checkpoint(
                    epoch_ckpt,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=epoch_train_loss,
                    history=history_payload,
                    best_val_acc=best_val_acc,
                )

            save_training_history_csv(history_path, train_losses, val_losses, train_accs, val_accs)

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
            )
    except KeyboardInterrupt:
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=completed_epoch,
            loss=last_epoch_train_loss,
            history={
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_acc": train_accs,
                "val_acc": val_accs,
            },
            best_val_acc=best_val_acc,
        )
        save_training_history_csv(history_path, train_losses, val_losses, train_accs, val_accs)
        print("Training interrupted by user. Progress saved. You can safely resume later.")
        return
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    print("=== Global Training End ===\n")

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    test_cm = confusion_matrix(test_targets, test_preds)

    print("=== Final Global Evaluation ===")
    print(f"Final training accuracy: {train_accs[-1]:.4f}")
    print(f"Final validation accuracy: {val_accs[-1]:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("Confusion matrix:")
    print(test_cm)
    print("===============================")

    models_dir = Path("models")
    plots_dir = Path("plots")
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = models_dir / "final_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_mapping": label_map,
            "args": vars(args),
            "train_subjects": sorted(train_subjects_present),
            "test_subjects": sorted(test_subjects_present),
        },
        final_model_path,
    )

    loss_plot = plots_dir / "loss.png"
    train_acc_plot = plots_dir / "train_accuracy.png"
    val_acc_plot = plots_dir / "val_accuracy.png"
    cm_plot = plots_dir / "confusion_matrix.png"

    epochs_axis = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot, dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, train_accs, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(train_acc_plot, dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, val_accs, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(val_acc_plot, dpi=120)
    plt.close()

    raw_labels = [inv_label_map[i] for i in range(len(inv_label_map))]
    save_confusion_matrix_plot(test_cm, raw_labels, cm_plot)

    report_path = Path("report.md")
    cm_table = confusion_matrix_markdown_table(test_cm, raw_labels)
    report_text = (
        "# EEG Transformer Classification Report\n\n"
        "## Dataset Description\n\n"
        f"- Number of training subjects: {len(train_subjects_present)} ({', '.join(train_subjects_present)})\n"
        f"- Number of test subjects: {len(test_subjects_present)} ({', '.join(test_subjects_present)})\n"
        f"- Total training trials: {len(train_dataset) + len(val_dataset)}\n"
        f"- Total test trials: {len(test_dataset)}\n\n"
        "## Training Results\n\n"
        f"- Final training accuracy: {train_accs[-1]:.4f}\n"
        f"- Final validation accuracy: {val_accs[-1]:.4f}\n\n"
        "## Test Results\n\n"
        f"- Test accuracy: {test_acc:.4f}\n\n"
        "### Confusion Matrix (Table)\n\n"
        f"{cm_table}\n\n"
        "### Visualizations\n\n"
        "![Loss Curve](plots/loss.png)\n"
        "![Training Accuracy](plots/train_accuracy.png)\n"
        "![Validation Accuracy](plots/val_accuracy.png)\n"
        "![Confusion Matrix](plots/confusion_matrix.png)\n"
    )
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Saved final model: {final_model_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved plots: {plots_dir}")

    # Cleanup
    del model
    del optimizer
    del criterion
    del train_loader
    del val_loader
    del test_loader
    del train_refs
    del test_refs
    del train_dataset
    del val_dataset
    del test_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        target_log_probs = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - target_probs).pow(self.gamma)

        if self.weight is not None:
            sample_weights = self.weight.gather(dim=0, index=targets)
            loss = -sample_weights * focal_factor * target_log_probs
        else:
            loss = -focal_factor * target_log_probs

        return loss.mean()


def train_one(
    args: argparse.Namespace,
    subject_filter: set[str] | None = None,
    subject_id: str | None = None,
    model_save_path: Path | None = None,
    metrics_save_path: Path | None = None,
    loss_plot_path: Path | None = None,
) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    files = discover_offline_movement_files(args.root_dir, subject_ids=subject_filter)
    if not files:
        raise FileNotFoundError("No OfflineMovement .mat files found.")

    if subject_filter:
        print(f"Subject filter: {sorted(subject_filter)}")

    # Mandatory first-step inspection.
    inspect_mat_file(files[0])

    # Load all runs from OfflineMovement only.
    run_x_dtype = np.float16 if args.low_memory else np.float32
    runs = load_all_offline_movement_runs(
        args.root_dir,
        window_sec=args.window_sec,
        subject_ids=subject_filter,
        x_dtype=run_x_dtype,
    )

    X_total, y_total, run_ids, subject_ids = combine_runs(runs)

    selected_labels = parse_class_labels(args.class_labels)
    if selected_labels:
        keep_mask = np.isin(y_total, selected_labels)
        X_total = X_total[keep_mask]
        y_total = y_total[keep_mask]
        run_ids = run_ids[keep_mask]
        subject_ids = subject_ids[keep_mask]
        print("\n=== Class Filtering ===")
        print(f"Requested original labels: {selected_labels}")
        print(f"Filtered X_total shape: {X_total.shape}")
        print(f"Filtered y unique labels (raw): {np.unique(y_total)}")
        print(f"Filtered subjects: {np.unique(subject_ids)}")
        print("=======================\n")

    if X_total.shape[0] == 0:
        raise ValueError("No samples left after class filtering.")

    # Preprocessing.
    norm_dtype = np.float16 if args.low_memory else np.float32
    norm_chunk_size = 64 if args.low_memory else None
    X_total = zscore_per_trial_channel(X_total, out_dtype=norm_dtype, chunk_size=norm_chunk_size)
    y_total, label_map = relabel_to_zero(y_total)

    print("\n=== After Preprocessing ===")
    print(f"X_total shape: {X_total.shape}")
    print(f"y_total shape: {y_total.shape}")
    print(f"Unique labels (zero-based): {np.unique(y_total)}")
    print(f"Label mapping old->new: {label_map}")
    print(f"Sample trial shape: {X_total[0].shape}")
    print("===========================\n")

    if args.split_mode == "run":
        split_idx = split_by_runs(
            run_ids,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
    elif args.split_mode == "subject":
        n_unique_subjects = len(np.unique(subject_ids))
        if n_unique_subjects < 3:
            print(
                "Requested split_mode='subject' but found fewer than 3 unique subjects. "
                "Falling back to split_mode='run' for this execution."
            )
            split_idx = split_by_runs(
                run_ids,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                seed=args.seed,
            )
        else:
            split_idx = split_by_subjects(
                subject_ids,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                seed=args.seed,
            )
    else:
        split_idx = split_random(
            n_samples=X_total.shape[0],
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )

    train_labels = y_total[split_idx.train]
    class_counts = np.bincount(train_labels, minlength=len(np.unique(y_total))).astype(np.float32)
    class_inv_freq = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_inv_freq = class_inv_freq / class_inv_freq.mean()

    train_sample_weights = None
    if args.train_sampler == "weighted":
        train_sample_weights = class_inv_freq[train_labels]
        print(f"Using train sampler='weighted' with class inverse frequency: {class_inv_freq}")

    train_loader, val_loader, test_loader = build_dataloaders(
        X_total,
        y_total,
        split_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_sample_weights=train_sample_weights,
    )

    num_classes = len(np.unique(y_total))
    in_channels = X_total.shape[1]
    time_len = X_total.shape[2]

    model = EEGTransformerClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_time_steps=max(time_len, args.max_time_steps),
    ).to(device)

    class_weights = None
    if args.weighted_loss:
        class_weights = torch.tensor(class_inv_freq, dtype=torch.float32, device=device)
        print(f"Using weighted loss with class weights: {class_inv_freq}")

    if args.loss_type == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
        print(f"Using loss_type='focal' with gamma={args.focal_gamma}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using loss_type='ce'")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, args.patience // 2)
    )

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(args.root_dir) / checkpoint_path
    if subject_id:
        checkpoint_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_{subject_id}{checkpoint_path.suffix}"
        )

    history_path = Path(args.training_history_path)
    if not history_path.is_absolute():
        history_path = Path(args.root_dir) / history_path
    if subject_id:
        history_path = history_path.with_name(f"{history_path.stem}_{subject_id}{history_path.suffix}")

    best_model_path = Path(args.best_model_path)
    if not best_model_path.is_absolute():
        best_model_path = Path(args.root_dir) / best_model_path
    if subject_id:
        best_model_path = best_model_path.with_name(f"{best_model_path.stem}_{subject_id}{best_model_path.suffix}")

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    start_epoch = 1
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer, map_location=device)
        if checkpoint is not None:
            last_epoch = int(checkpoint.get("epoch", 0))
            start_epoch = last_epoch + 1
            best_val_acc = float(checkpoint.get("best_val_acc", -1.0))
            history = checkpoint.get("history", {})
            train_losses = [float(v) for v in history.get("train_loss", [])]
            val_losses = [float(v) for v in history.get("val_loss", [])]
            train_accs = [float(v) for v in history.get("train_acc", [])]
            val_accs = [float(v) for v in history.get("val_acc", [])]
            print(f"Resuming from epoch {last_epoch}")

    print("\n=== Training Start ===")
    interrupted = {"flag": False}

    def _handle_interrupt(sig, frame):
        interrupted["flag"] = True
        print("\nKeyboard interrupt received. Saving checkpoint...")

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_interrupt)

    last_train_loss = float(train_losses[-1]) if train_losses else 0.0
    completed_epoch = max(0, start_epoch - 1)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_count = 0

            for Xb, yb in train_loader:
                if interrupted["flag"]:
                    raise KeyboardInterrupt

                Xb = Xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)
                logits = model(Xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * yb.size(0)
                running_correct += (torch.argmax(logits, dim=1) == yb).sum().item()
                running_count += yb.size(0)

            train_loss = running_loss / max(1, running_count)
            train_acc = running_correct / max(1, running_count)

            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_acc)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            completed_epoch = epoch
            last_train_loss = train_loss

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                if args.save_best_model:
                    best_model_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "epoch": int(epoch),
                            "val_acc": float(val_acc),
                            "args": vars(args),
                        },
                        best_model_path,
                    )
            else:
                patience_counter += 1

            history_payload = {
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_acc": train_accs,
                "val_acc": val_accs,
            }
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                history=history_payload,
                best_val_acc=best_val_acc,
            )

            if args.save_epoch_checkpoints:
                epoch_ckpt = checkpoint_path.with_name(f"{checkpoint_path.stem}_epoch_{epoch:03d}{checkpoint_path.suffix}")
                save_checkpoint(
                    epoch_ckpt,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=train_loss,
                    history=history_payload,
                    best_val_acc=best_val_acc,
                )

            save_training_history_csv(history_path, train_losses, val_losses, train_accs, val_accs)

            if args.early_stopping and patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
    except KeyboardInterrupt:
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=completed_epoch,
            loss=last_train_loss,
            history={
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_acc": train_accs,
                "val_acc": val_accs,
            },
            best_val_acc=best_val_acc,
        )
        save_training_history_csv(history_path, train_losses, val_losses, train_accs, val_accs)
        print("Training interrupted by user. Progress saved. You can safely resume later.")
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    print("=== Training End ===\n")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Error (Loss)', fontsize=12)
    plt.title('Training Progress: Epoch vs Error', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = loss_plot_path if loss_plot_path is not None else Path("loss_curve.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=100)
    print(f"Loss curve saved to: {plot_path}")
    plt.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    cm = confusion_matrix(test_targets, test_preds)
    bal_acc = balanced_accuracy_score(test_targets, test_preds)

    print("=== Final Evaluation ===")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Final balanced accuracy: {bal_acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(classification_report(test_targets, test_preds, digits=4, zero_division=0))
    print("========================")

    inferred_subject = subject_id
    if inferred_subject is None and subject_filter and len(subject_filter) == 1:
        inferred_subject = next(iter(subject_filter))

    inv_label_map = {new: old for old, new in label_map.items()}
    test_targets_raw = np.array([inv_label_map[int(lbl)] for lbl in test_targets], dtype=np.int64)
    test_preds_raw = np.array([inv_label_map[int(lbl)] for lbl in test_preds], dtype=np.int64)

    resolved_model_save_path = model_save_path
    if resolved_model_save_path is None and args.save_model_path:
        resolved_model_save_path = Path(args.save_model_path)

    if resolved_model_save_path is not None:
        resolved_model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_mapping": label_map,
                "args": vars(args),
                "subject_id": inferred_subject,
                "metrics": {
                    "train_accuracy": float(train_accs[-1]) if train_accs else 0.0,
                    "val_accuracy": float(val_accs[-1]) if val_accs else 0.0,
                    "test_accuracy": float(test_acc),
                },
            },
            resolved_model_save_path,
        )
        print(f"Saved model to: {resolved_model_save_path}")

    if metrics_save_path is not None:
        metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            "subject_id": inferred_subject,
            "num_trials": int(X_total.shape[0]),
            "train_accuracy": float(train_accs[-1]) if train_accs else 0.0,
            "val_accuracy": float(val_accs[-1]) if val_accs else 0.0,
            "test_accuracy": float(test_acc),
            "balanced_accuracy": float(bal_acc),
            "test_loss": float(test_loss),
            "confusion_matrix": cm.tolist(),
        }
        with metrics_save_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)
        print(f"Saved metrics to: {metrics_save_path}")

    result = {
        "subject_id": inferred_subject,
        "num_trials": int(X_total.shape[0]),
        "train_acc": float(train_accs[-1]) if train_accs else 0.0,
        "val_acc": float(val_accs[-1]) if val_accs else 0.0,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "balanced_acc": float(bal_acc),
        "test_targets_raw": test_targets_raw,
        "test_preds_raw": test_preds_raw,
        "loss_curve_path": str(plot_path),
    }

    # Explicit cleanup to keep memory stable across subjects.
    del model
    del optimizer
    del scheduler
    del criterion
    del train_loader
    del val_loader
    del test_loader
    del runs
    del X_total
    del y_total
    del run_ids
    del subject_ids
    del train_labels
    del class_counts
    del class_inv_freq
    del best_state
    del test_preds
    del test_targets
    del test_preds_raw
    del test_targets_raw
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def train(args: argparse.Namespace) -> None:
    try:
        if args.cross_subject_global:
            train_cross_subject_global(args)
            return

        requested_subjects = parse_subject_ids(args.subject_ids)

        if args.per_subject:
            files = discover_offline_movement_files(args.root_dir)
            if not files:
                raise FileNotFoundError("No OfflineMovement .mat files found.")

            all_subjects = sorted({fp.stem.split("_", 1)[0] if "_" in fp.stem else "UNKNOWN" for fp in files})
            subjects_to_run = [s for s in all_subjects if (not requested_subjects or s in requested_subjects)]
            if not subjects_to_run:
                raise ValueError("No matching subjects to run in per-subject mode.")

            print("\n=== Per-Subject Training Mode ===")
            print(f"Subjects to run: {subjects_to_run}")
            print("===============================\n")

            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            summary: list[dict[str, Any]] = []
            all_targets_raw: list[np.ndarray] = []
            all_preds_raw: list[np.ndarray] = []

            for subj in subjects_to_run:
                print(f"\nTraining Subject {subj}...")
                model_path = save_dir / f"subject_{subj}.pth"
                metrics_path = save_dir / f"subject_{subj}_metrics.json"
                subject_loss_plot_path = save_dir / f"loss_curve_{subj}.png"

                metrics = train_one(
                    args,
                    subject_filter={subj},
                    subject_id=subj,
                    model_save_path=model_path,
                    metrics_save_path=metrics_path,
                    loss_plot_path=subject_loss_plot_path,
                )
                summary.append(metrics)
                all_targets_raw.append(metrics["test_targets_raw"])
                all_preds_raw.append(metrics["test_preds_raw"])

                print(f"Finished {subj} -> Test Accuracy: {metrics['test_acc'] * 100:.2f}%")
                print(f"Model saved at: {model_path}")
                print("--------------------------------------")

            # Ensure no subject state lingers before next subject starts.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results_csv_path = save_dir / "results.csv"
            with results_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["subject_id", "num_trials", "train_accuracy", "val_accuracy", "test_accuracy"])
                writer.writeheader()
                for metrics in summary:
                    writer.writerow(
                        {
                            "subject_id": metrics["subject_id"],
                            "num_trials": metrics["num_trials"],
                            "train_accuracy": f"{metrics['train_acc']:.6f}",
                            "val_accuracy": f"{metrics['val_acc']:.6f}",
                            "test_accuracy": f"{metrics['test_acc']:.6f}",
                        }
                    )
            print(f"Saved summary CSV to: {results_csv_path}")

            subject_acc_plot_path = save_dir / "subject_accuracy_trend.png"
            plt.figure(figsize=(11, 6))
            x = np.arange(len(summary))
            y = [m["test_acc"] for m in summary]
            labels = [m["subject_id"] for m in summary]
            plt.plot(x, y, marker="o", linewidth=2)
            plt.xticks(x, labels, rotation=45, ha="right")
            plt.ylim(0.0, 1.0)
            plt.xlabel("Subject")
            plt.ylabel("Test Accuracy")
            plt.title("Subject-wise Test Accuracy")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(subject_acc_plot_path, dpi=100)
            plt.close()
            print(f"Saved subject accuracy graph to: {subject_acc_plot_path}")

            overall_targets = np.concatenate(all_targets_raw) if all_targets_raw else np.array([], dtype=np.int64)
            overall_preds = np.concatenate(all_preds_raw) if all_preds_raw else np.array([], dtype=np.int64)
            all_labels = np.unique(np.concatenate([overall_targets, overall_preds])) if overall_targets.size else np.array([], dtype=np.int64)
            overall_cm = confusion_matrix(overall_targets, overall_preds, labels=all_labels) if overall_targets.size else np.array([[]])
            overall_acc = float((overall_preds == overall_targets).mean()) if overall_targets.size else 0.0

            print("\n=== Per-Subject Summary ===")
            for metrics in summary:
                print(
                    f"{metrics['subject_id']} | test_loss={metrics['test_loss']:.4f} | "
                    f"test_acc={metrics['test_acc']:.4f} | balanced_acc={metrics['balanced_acc']:.4f}"
                )
            print(f"Overall accuracy across subjects: {overall_acc:.4f}")
            print("Overall confusion matrix:")
            print(overall_cm)
            print("===========================")
            return

        subject_filter = requested_subjects if requested_subjects else None
        train_one(args, subject_filter=subject_filter)
    except KeyboardInterrupt:
        print("Training interrupted by user. Latest checkpoint has been saved.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OfflineMovement EEG Transformer classification")
    p.add_argument("--root_dir", type=str, default=".", help="Workspace root containing subject folders")
    p.add_argument("--window_sec", type=float, default=1.0, help="Trial window length (seconds) for FORMAT_A")
    p.add_argument(
        "--split_mode",
        type=str,
        default="run",
        choices=["run", "subject", "random"],
        help="Split by unseen runs, unseen subjects, or random samples",
    )
    p.add_argument(
        "--class_labels",
        type=str,
        default="",
        help="Optional comma-separated original labels to keep, e.g. '1,2' or '1,2,3'",
    )
    p.add_argument(
        "--subject_ids",
        type=str,
        default="",
        help="Optional comma-separated subject IDs to include, e.g. 'S01,S02'",
    )
    p.add_argument(
        "--per_subject",
        action="store_true",
        help="Train each subject separately to reduce memory usage",
    )
    p.add_argument(
        "--cross_subject_global",
        action="store_true",
        help="Train one global model on S01-S18 and evaluate on S19-S21 using movement-only data",
    )
    p.add_argument("--train_subject_start", type=int, default=1)
    p.add_argument("--train_subject_end", type=int, default=18)
    p.add_argument("--test_subject_start", type=int, default=19)
    p.add_argument("--test_subject_end", type=int, default=21)
    p.add_argument("--val_frac_global", type=float, default=0.15)
    p.add_argument(
        "--cross_subject_offline_only",
        action="store_true",
        help="Use only OfflineMovement runs in cross-subject global mode to reduce data volume and speed up epochs",
    )
    p.add_argument(
        "--cross_subject_cache_dir",
        type=str,
        default=".cache/cross_subject",
        help="Directory to store precomputed normalized trial caches for cross-subject mode",
    )
    p.add_argument(
        "--rebuild_cross_subject_cache",
        action="store_true",
        help="Force rebuilding cross-subject cache files instead of reusing existing caches",
    )

    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dim_feedforward", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max_time_steps", type=int, default=2048)

    p.add_argument("--early_stopping", action="store_true")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--loss_type", type=str, default="ce", choices=["ce", "focal"])
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--weighted_loss", action="store_true", help="Use inverse-frequency class weighting")
    p.add_argument(
        "--train_sampler",
        type=str,
        default="none",
        choices=["none", "weighted"],
        help="Optional training sampler strategy",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--low_memory", action="store_true", help="Use lower-precision arrays to reduce RAM usage")
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/latest_checkpoint.pth",
        help="Path to the latest training checkpoint for auto-resume",
    )
    p.add_argument(
        "--training_history_path",
        type=str,
        default="checkpoints/training_history.csv",
        help="CSV file to persist epoch-wise train/val loss and accuracy",
    )
    p.add_argument(
        "--save_epoch_checkpoints",
        action="store_true",
        help="Also save per-epoch checkpoint files in addition to latest checkpoint",
    )
    p.add_argument(
        "--manual_checkpoint_path",
        type=str,
        default="",
        help="Optional path to save a manual checkpoint snapshot at startup",
    )
    p.add_argument(
        "--save_best_model",
        action="store_true",
        help="Save the best model separately based on validation accuracy",
    )
    p.add_argument(
        "--best_model_path",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path for best-model checkpoint when --save_best_model is enabled",
    )
    p.add_argument("--save_model_path", type=str, default="")
    p.add_argument("--save_dir", type=str, default="models", help="Output directory for per-subject models and summaries")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    train(args)
