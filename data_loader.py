from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


@dataclass
class RunData:
    run_id: str
    file_path: str
    subject_id: str
    X: np.ndarray  # (trials, channels, time)
    y: np.ndarray  # (trials,)


@dataclass
class TrialRef:
    file_path: str
    run_id: str
    subject_id: str
    fmt: str
    raw_label: int
    start: int | None = None
    end: int | None = None
    trial_idx: int | None = None


def _shape_of(value: Any) -> str:
    shape = getattr(value, "shape", None)
    return str(shape) if shape is not None else "N/A"


def inspect_mat_file(file_path: str | Path) -> tuple[dict[str, Any], str]:
    """Load one MAT file and print key/type/shape diagnostics (mandatory step)."""
    file_path = Path(file_path)
    mat = loadmat(file_path, simplify_cells=True)

    print("\n=== MAT FILE INSPECTION ===")
    print(f"File: {file_path}")
    print("All keys:", list(mat.keys()))
    for k, v in mat.items():
        print(f"- Key='{k}' | type={type(v)} | shape={_shape_of(v)}")

    fmt = detect_format(mat)
    print(f"Detected format: {fmt}")
    print("===========================\n")
    return mat, fmt


def detect_format(mat: dict[str, Any]) -> str:
    """Detect supported MAT structure.

    FORMAT_A: continuous EEG + events
    FORMAT_B: pre-epoched arrays X, y
    """
    key_map = {k.lower(): k for k in mat.keys()}

    x_key = key_map.get("x")
    y_key = key_map.get("y")
    if x_key and y_key:
        x = mat[x_key]
        y = mat[y_key]
        if isinstance(x, np.ndarray) and x.ndim == 3 and isinstance(y, np.ndarray) and y.ndim == 1:
            return "FORMAT_B"

    eeg_key = key_map.get("eeg")
    event_key = key_map.get("event")
    if eeg_key and event_key and isinstance(mat[eeg_key], dict):
        eeg = mat[eeg_key]
        if "data" in eeg and isinstance(eeg["data"], np.ndarray) and eeg["data"].ndim == 2:
            return "FORMAT_A"

    raise ValueError("Unsupported MAT structure. Could not detect FORMAT_A or FORMAT_B.")


def _ensure_int_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    if not np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.int64)
    return y


def _extract_event_label(ev: dict[str, Any]) -> int | None:
    # Prefer explicit numeric value, fallback to type if numeric string.
    if "value" in ev:
        value = ev["value"]
        try:
            return int(value)
        except Exception:
            pass

    if "type" in ev:
        event_type = ev["type"]
        if isinstance(event_type, (int, np.integer)):
            return int(event_type)
        if isinstance(event_type, str) and event_type.isdigit():
            return int(event_type)

    return None


def _extract_event_sample(ev: dict[str, Any]) -> int | None:
    for key in ("sample", "latency"):
        if key in ev:
            try:
                return int(ev[key])
            except Exception:
                return None
    return None


def extract_trials_from_format_a(
    mat: dict[str, Any],
    window_sec: float = 1.0,
    x_dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    key_map = {k.lower(): k for k in mat.keys()}
    eeg = mat[key_map["eeg"]]
    events = mat[key_map["event"]]

    data = np.asarray(eeg["data"])  # (channels, time)

    # Sampling rate compatibility across styles.
    fs = eeg.get("srate", eeg.get("fsample", None))
    if fs is None:
        raise ValueError("FORMAT_A detected, but no sampling rate key found (expected 'srate' or 'fsample').")
    fs = float(fs)
    window_samples = int(round(window_sec * fs))

    if isinstance(events, dict):
        events = [events]
    if not isinstance(events, (list, tuple)):
        raise ValueError("Expected events to be a list/dict in FORMAT_A.")

    valid_windows: list[tuple[int, int, int]] = []

    for ev in events:
        if not isinstance(ev, dict):
            continue

        label = _extract_event_label(ev)
        sample_1_based = _extract_event_sample(ev)
        if label is None or sample_1_based is None:
            continue

        # MATLAB uses 1-based indexing; convert to 0-based for Python.
        start = sample_1_based - 1
        if start < 0:
            continue
        end = start + window_samples
        if end > data.shape[1]:
            continue

        valid_windows.append((start, end, label))

    if not valid_windows:
        raise ValueError("No valid trials extracted from FORMAT_A events.")

    n_trials = len(valid_windows)
    n_channels = int(data.shape[0])
    X = np.empty((n_trials, n_channels, window_samples), dtype=x_dtype)
    y = np.empty((n_trials,), dtype=np.int64)

    for i, (start, end, label) in enumerate(valid_windows):
        # Copy one window at a time to avoid large temporary allocations.
        X[i] = data[:, start:end].astype(x_dtype, copy=False)
        y[i] = int(label)

    y = _ensure_int_labels(y)
    return X, y


def extract_trials_from_format_b(
    mat: dict[str, Any],
    x_dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    key_map = {k.lower(): k for k in mat.keys()}
    X = np.asarray(mat[key_map["x"]], dtype=x_dtype)
    y = _ensure_int_labels(np.asarray(mat[key_map["y"]]))

    if X.ndim != 3:
        raise ValueError(f"FORMAT_B expected X to be 3D (trials, channels, time), got shape {X.shape}.")
    if y.ndim != 1:
        y = y.reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"FORMAT_B mismatch: X has {X.shape[0]} trials but y has {y.shape[0]} labels.")

    return X, y


def load_single_run(
    file_path: str | Path,
    window_sec: float = 1.0,
    print_debug: bool = True,
    x_dtype: np.dtype = np.float32,
) -> RunData:
    file_path = Path(file_path)
    mat = loadmat(file_path, simplify_cells=True)
    fmt = detect_format(mat)

    if fmt == "FORMAT_A":
        X, y = extract_trials_from_format_a(mat, window_sec=window_sec, x_dtype=x_dtype)
    else:
        X, y = extract_trials_from_format_b(mat, x_dtype=x_dtype)
    X = X.astype(x_dtype, copy=False)

    run_id = file_path.stem
    subject_id = "UNKNOWN"
    if run_id.startswith("S") and "_" in run_id:
        subject_id = run_id.split("_", 1)[0]
    else:
        # Fallback to nearest folder name that looks like Sxx.
        for part in file_path.parts:
            if part.startswith("S") and part[1:].isdigit():
                subject_id = part
                break

    if print_debug:
        print(
            f"Run {run_id} (subject={subject_id}) -> "
            f"X shape: {X.shape}, y shape: {y.shape}, unique labels: {np.unique(y)}"
        )
        if X.shape[0] > 0:
            print(f"Sample trial shape: {X[0].shape}")

    return RunData(run_id=run_id, file_path=str(file_path), subject_id=subject_id, X=X, y=y)


def build_trial_index_for_run(
    file_path: str | Path,
    window_sec: float = 1.0,
    print_debug: bool = True,
) -> tuple[list[TrialRef], tuple[int, int]]:
    """Build trial-level references for a run without constructing a full trial tensor."""
    file_path = Path(file_path)
    run_id = file_path.stem
    subject_id = "UNKNOWN"
    if run_id.startswith("S") and "_" in run_id:
        subject_id = run_id.split("_", 1)[0]
    else:
        for part in file_path.parts:
            if part.startswith("S") and part[1:].isdigit():
                subject_id = part
                break

    mat = loadmat(file_path, simplify_cells=True)
    fmt = detect_format(mat)
    refs: list[TrialRef] = []

    if fmt == "FORMAT_A":
        key_map = {k.lower(): k for k in mat.keys()}
        eeg = mat[key_map["eeg"]]
        events = mat[key_map["event"]]
        data = np.asarray(eeg["data"])  # (channels, time)

        fs = eeg.get("srate", eeg.get("fsample", None))
        if fs is None:
            raise ValueError("FORMAT_A detected, but no sampling rate key found (expected 'srate' or 'fsample').")
        fs = float(fs)
        window_samples = int(round(window_sec * fs))

        if isinstance(events, dict):
            events = [events]
        if not isinstance(events, (list, tuple)):
            raise ValueError("Expected events to be a list/dict in FORMAT_A.")

        for ev in events:
            if not isinstance(ev, dict):
                continue
            label = _extract_event_label(ev)
            sample_1_based = _extract_event_sample(ev)
            if label is None or sample_1_based is None:
                continue

            start = sample_1_based - 1
            if start < 0:
                continue
            end = start + window_samples
            if end > data.shape[1]:
                continue

            refs.append(
                TrialRef(
                    file_path=str(file_path),
                    run_id=run_id,
                    subject_id=subject_id,
                    fmt=fmt,
                    raw_label=int(label),
                    start=int(start),
                    end=int(end),
                    trial_idx=None,
                )
            )

        sample_shape = (int(data.shape[0]), int(window_samples))
    else:
        key_map = {k.lower(): k for k in mat.keys()}
        X = np.asarray(mat[key_map["x"]])
        y = _ensure_int_labels(np.asarray(mat[key_map["y"]]))
        if X.ndim != 3:
            raise ValueError(f"FORMAT_B expected X to be 3D (trials, channels, time), got shape {X.shape}.")
        if y.ndim != 1:
            y = y.reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"FORMAT_B mismatch: X has {X.shape[0]} trials but y has {y.shape[0]} labels.")

        for i, label in enumerate(y.tolist()):
            refs.append(
                TrialRef(
                    file_path=str(file_path),
                    run_id=run_id,
                    subject_id=subject_id,
                    fmt=fmt,
                    raw_label=int(label),
                    start=None,
                    end=None,
                    trial_idx=int(i),
                )
            )

        sample_shape = (int(X.shape[1]), int(X.shape[2]))

    if not refs:
        raise ValueError(f"No valid trials found in run: {run_id}")

    if print_debug:
        labels = sorted({ref.raw_label for ref in refs})
        print(
            f"Run {run_id} (subject={subject_id}) -> "
            f"Trials: {len(refs)}, unique labels: {labels}"
        )
        print(f"Sample trial shape: {sample_shape}")

    return refs, sample_shape


def discover_offline_movement_files(root_dir: str | Path, subject_ids: set[str] | None = None) -> list[Path]:
    """Find all OfflineMovement MAT files under root directory."""
    root_dir = Path(root_dir)
    files = sorted(root_dir.glob("S*/S*/OfflineMovement/*.mat"))
    if not files:
        # Fallback to a wider search if directory depth differs.
        files = sorted(root_dir.rglob("*OfflineMovement*.mat"))

    if subject_ids:
        filtered = []
        for fp in files:
            stem = fp.stem
            subject = stem.split("_", 1)[0] if "_" in stem else "UNKNOWN"
            if subject in subject_ids:
                filtered.append(fp)
        files = filtered

    return files


def discover_movement_files(
    root_dir: str | Path,
    subject_ids: set[str] | None = None,
    include_offline: bool = True,
    include_online: bool = True,
) -> list[Path]:
    """Discover movement MAT files while explicitly excluding imagery and smooth-movement files."""
    if not include_offline and not include_online:
        return []

    root_dir = Path(root_dir)
    all_mat = sorted(root_dir.rglob("*.mat"))
    filtered: list[Path] = []

    for fp in all_mat:
        path_str = str(fp).lower()
        if "imagery" in path_str:
            continue
        if "smoothmovement" in path_str:
            continue
        if "movement" not in path_str:
            continue

        is_offline = "offlinemovement" in path_str
        is_online = "onlinemovement" in path_str
        if is_offline and not include_offline:
            continue
        if is_online and not include_online:
            continue
        if not is_offline and not is_online:
            # Keep movement files under custom movement folders as long as they are not imagery.
            pass

        if subject_ids:
            stem = fp.stem
            subject = stem.split("_", 1)[0] if "_" in stem else "UNKNOWN"
            if subject not in subject_ids:
                continue

        filtered.append(fp)

    return filtered


def load_all_offline_movement_runs(
    root_dir: str | Path,
    window_sec: float = 1.0,
    subject_ids: set[str] | None = None,
    x_dtype: np.dtype = np.float32,
) -> list[RunData]:
    files = discover_offline_movement_files(root_dir, subject_ids=subject_ids)
    if not files:
        raise FileNotFoundError("No OfflineMovement MAT files found.")

    print(f"Discovered {len(files)} OfflineMovement runs.")
    runs: list[RunData] = []
    for fp in files:
        run = load_single_run(fp, window_sec=window_sec, print_debug=True, x_dtype=x_dtype)
        runs.append(run)
    return runs


def load_all_movement_runs(
    root_dir: str | Path,
    window_sec: float = 1.0,
    subject_ids: set[str] | None = None,
    x_dtype: np.dtype = np.float32,
    include_offline: bool = True,
    include_online: bool = True,
) -> list[RunData]:
    files = discover_movement_files(
        root_dir,
        subject_ids=subject_ids,
        include_offline=include_offline,
        include_online=include_online,
    )
    if not files:
        raise FileNotFoundError("No movement MAT files found for the requested filters.")

    print(f"Discovered {len(files)} movement runs.")
    runs: list[RunData] = []
    for fp in files:
        run = load_single_run(fp, window_sec=window_sec, print_debug=True, x_dtype=x_dtype)
        runs.append(run)
    return runs
