from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler

from data_loader import RunData


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def zscore_per_trial_channel(
    X: np.ndarray,
    eps: float = 1e-6,
    out_dtype: np.dtype = np.float32,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Normalize each trial and channel along the time axis."""
    if chunk_size is None or chunk_size <= 0:
        Xf = X.astype(np.float32, copy=False)
        mean = Xf.mean(axis=-1, keepdims=True)
        std = Xf.std(axis=-1, keepdims=True)
        Xn = (Xf - mean) / (std + eps)
        return Xn.astype(out_dtype)

    Xn = np.empty_like(X, dtype=out_dtype)
    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        Xf = X[start:end].astype(np.float32, copy=False)
        mean = Xf.mean(axis=-1, keepdims=True)
        std = Xf.std(axis=-1, keepdims=True)
        Xn[start:end] = ((Xf - mean) / (std + eps)).astype(out_dtype)

    return Xn


def relabel_to_zero(y: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Map labels to contiguous integers starting at 0."""
    unique = np.unique(y)
    mapping = {int(old): int(new) for new, old in enumerate(unique.tolist())}
    y_new = np.array([mapping[int(lbl)] for lbl in y], dtype=np.int64)
    return y_new, mapping


def combine_runs(runs: list[RunData]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_chunks = []
    y_chunks = []
    run_ids = []
    subject_ids = []

    for run in runs:
        X_chunks.append(run.X)
        y_chunks.append(run.y)
        run_ids.extend([run.run_id] * run.X.shape[0])
        subject_ids.extend([run.subject_id] * run.X.shape[0])

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0).astype(np.int64)
    run_ids = np.array(run_ids)
    subject_ids = np.array(subject_ids)

    print("\n=== Combined Dataset ===")
    print(f"X_total shape: {X.shape}")
    print(f"y_total shape: {y.shape}")
    print(f"Unique labels (raw): {np.unique(y)}")
    print(f"Unique subjects: {np.unique(subject_ids)}")
    print(f"Sample trial shape: {X[0].shape}")
    print("========================\n")

    return X, y, run_ids, subject_ids


def split_by_runs(
    run_ids: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    rng = np.random.default_rng(seed)
    unique_runs = np.unique(run_ids)
    perm = rng.permutation(len(unique_runs))
    unique_runs = unique_runs[perm]

    n_runs = len(unique_runs)
    n_train = max(1, int(round(n_runs * train_frac)))
    n_val = max(1, int(round(n_runs * val_frac))) if n_runs >= 3 else 1
    n_test = n_runs - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_runs = set(unique_runs[:n_train])
    val_runs = set(unique_runs[n_train : n_train + n_val])
    test_runs = set(unique_runs[n_train + n_val :])

    train_idx = np.where(np.isin(run_ids, list(train_runs)))[0]
    val_idx = np.where(np.isin(run_ids, list(val_runs)))[0]
    test_idx = np.where(np.isin(run_ids, list(test_runs)))[0]

    print("\n=== Run-wise Split ===")
    print(f"Total unique runs: {n_runs}")
    print(f"Train runs: {len(train_runs)}, Val runs: {len(val_runs)}, Test runs: {len(test_runs)}")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")
    print("======================\n")

    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def split_random(
    n_samples: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.0")
    if n_samples < 3:
        raise ValueError("Need at least 3 samples for random split.")

    rng = np.random.default_rng(seed)
    all_idx = rng.permutation(n_samples)

    n_train = max(1, int(round(n_samples * train_frac)))
    n_val = max(1, int(round(n_samples * val_frac)))
    n_test = n_samples - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train : n_train + n_val]
    test_idx = all_idx[n_train + n_val :]

    print("\n=== Random Split ===")
    print(f"Total samples: {n_samples}")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")
    print("====================\n")

    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def split_by_subjects(
    subject_ids: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) < 3:
        raise ValueError("Need at least 3 unique subjects for subject-wise split.")

    perm = rng.permutation(len(unique_subjects))
    unique_subjects = unique_subjects[perm]

    n_subjects = len(unique_subjects)
    n_train = max(1, int(round(n_subjects * train_frac)))
    n_val = max(1, int(round(n_subjects * val_frac)))
    n_test = n_subjects - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train : n_train + n_val])
    test_subjects = set(unique_subjects[n_train + n_val :])

    train_idx = np.where(np.isin(subject_ids, list(train_subjects)))[0]
    val_idx = np.where(np.isin(subject_ids, list(val_subjects)))[0]
    test_idx = np.where(np.isin(subject_ids, list(test_subjects)))[0]

    print("\n=== Subject-wise Split ===")
    print(f"Total unique subjects: {n_subjects}")
    print(f"Train subjects: {sorted(train_subjects)}")
    print(f"Val subjects: {sorted(val_subjects)}")
    print(f"Test subjects: {sorted(test_subjects)}")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")
    print("==========================\n")

    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def build_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    split_idx: SplitIndices,
    batch_size: int = 32,
    num_workers: int = 0,
    train_sample_weights: np.ndarray | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    y_train = y[split_idx.train]
    y_val = y[split_idx.val]
    y_test = y[split_idx.test]

    print("\n=== Tensor Split Shapes ===")
    print(f"Train X/y: ({len(split_idx.train)}, {X.shape[1]}, {X.shape[2]}) / {y_train.shape}")
    print(f"Val   X/y: ({len(split_idx.val)}, {X.shape[1]}, {X.shape[2]}) / {y_val.shape}")
    print(f"Test  X/y: ({len(split_idx.test)}, {X.shape[1]}, {X.shape[2]}) / {y_test.shape}")
    print(f"Train unique labels: {np.unique(y_train)}")
    print(f"Val unique labels: {np.unique(y_val)}")
    print(f"Test unique labels: {np.unique(y_test)}")
    print("===========================\n")

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    all_ds = TensorDataset(X_tensor, y_tensor)
    train_ds = Subset(all_ds, split_idx.train.tolist())
    val_ds = Subset(all_ds, split_idx.val.tolist())
    test_ds = Subset(all_ds, split_idx.test.tolist())

    if train_sample_weights is not None:
        if len(train_sample_weights) != len(train_ds):
            raise ValueError(
                f"train_sample_weights length ({len(train_sample_weights)}) must match train samples ({len(train_ds)})."
            )
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(train_sample_weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
        )
        print("Using weighted random sampler for training batches.")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
