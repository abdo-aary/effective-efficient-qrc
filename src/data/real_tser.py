"""Utilities for Monash/UEA/UCR TSER real-world regression datasets."""

from __future__ import annotations

import json
import zipfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.data.generate.base import WindowsDataset
from src.settings import PROJECT_ROOT_PATH


PROJECT_ROOT = Path(PROJECT_ROOT_PATH)
DEFAULT_TSER_RAW_ROOT = PROJECT_ROOT / "storage/raw/tser"
DEFAULT_TSER_DATA_ROOT = PROJECT_ROOT / "storage/data/real/tser"
DEFAULT_HYDRAULIC_RAW_ROOT = PROJECT_ROOT / "storage/raw/hydraulic_systems"
HYDRAULIC_UCI_URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
HYDRAULIC_DATASET_KEY = "hydraulic_systems"
HYDRAULIC_SPLIT_SOURCE = "deterministic_random_80_20"

HYDRAULIC_SENSOR_SPECS: tuple[tuple[str, int], ...] = (
    ("PS1", 100),
    ("PS2", 100),
    ("PS3", 100),
    ("PS4", 100),
    ("PS5", 100),
    ("PS6", 100),
    ("EPS1", 100),
    ("FS1", 10),
    ("FS2", 10),
    ("TS1", 1),
    ("TS2", 1),
    ("TS3", 1),
    ("TS4", 1),
    ("VS1", 1),
    ("CE", 1),
    ("CP", 1),
    ("SE", 1),
)

HYDRAULIC_TARGET_COLUMNS: dict[str, int] = {
    "cooler_condition": 0,
    "valve_condition": 1,
    "internal_pump_leakage": 2,
    "hydraulic_accumulator_pressure": 3,
    "stable_flag": 4,
}


@dataclass(frozen=True)
class TSERDatasetSpec:
    key: str
    record_id: str
    dataset_name: str
    train_file: str
    test_file: str
    description: str
    expected_dimensions: int | None = None

    @property
    def dataset_id(self) -> str:
        return self.key


TSER_DATASETS: dict[str, TSERDatasetSpec] = {
    "benzene_concentration": TSERDatasetSpec(
        key="benzene_concentration",
        record_id="3902673",
        dataset_name="BenzeneConcentration",
        train_file="BenzeneConcentration_TRAIN.ts",
        test_file="BenzeneConcentration_TEST.ts",
        expected_dimensions=8,
        description="Chemical sensor trajectories -> scalar benzene concentration.",
    ),
    "beijing_pm25": TSERDatasetSpec(
        key="beijing_pm25",
        record_id="3902671",
        dataset_name="BeijingPM25Quality",
        train_file="BeijingPM25Quality_TRAIN.ts",
        test_file="BeijingPM25Quality_TEST.ts",
        expected_dimensions=9,
        description="Hourly pollutant/meteorological trajectories -> scalar PM2.5 target.",
    ),
    "live_fuel_moisture": TSERDatasetSpec(
        key="live_fuel_moisture",
        record_id="4632439",
        dataset_name="LiveFuelMoistureContent",
        train_file="LiveFuelMoistureContent_TRAIN.ts",
        test_file="LiveFuelMoistureContent_TEST.ts",
        expected_dimensions=7,
        description="Satellite reflectance trajectories -> scalar live fuel moisture content.",
    ),
}


REAL_DATASET_KEYS: tuple[str, ...] = tuple(sorted([*TSER_DATASETS.keys(), HYDRAULIC_DATASET_KEY]))


def prepare_real_dataset(
    dataset: str,
    *,
    raw_root: str | Path | None = None,
    out_root: str | Path = DEFAULT_TSER_DATA_ROOT,
    download: bool = True,
    force_download: bool = False,
    overwrite: bool = False,
    max_train: int | None = None,
    max_test: int | None = None,
    seed: int = 0,
    clip_value: float = 5.0,
    hydraulic_target: str = "hydraulic_accumulator_pressure",
    hydraulic_test_ratio: float = 0.2,
    hydraulic_split_strategy: str = "random",
    hydraulic_stable_only: bool = True,
    hydraulic_length_seconds: int = 60,
) -> Path:
    """Prepare one supported real-world scalar-label trajectory dataset."""

    key = str(dataset)
    if key == HYDRAULIC_DATASET_KEY:
        return prepare_hydraulic_dataset(
            raw_root=DEFAULT_HYDRAULIC_RAW_ROOT if raw_root is None else raw_root,
            out_root=out_root,
            download=download,
            force_download=force_download,
            overwrite=overwrite,
            max_train=max_train,
            max_test=max_test,
            seed=seed,
            clip_value=clip_value,
            target=hydraulic_target,
            test_ratio=hydraulic_test_ratio,
            split_strategy=hydraulic_split_strategy,
            stable_only=hydraulic_stable_only,
            length_seconds=hydraulic_length_seconds,
        )
    return prepare_tser_dataset(
        key,
        raw_root=DEFAULT_TSER_RAW_ROOT if raw_root is None else raw_root,
        out_root=out_root,
        download=download,
        force_download=force_download,
        overwrite=overwrite,
        max_train=max_train,
        max_test=max_test,
        seed=seed,
        clip_value=clip_value,
    )


def prepare_tser_dataset(
    dataset: str,
    *,
    raw_root: str | Path = DEFAULT_TSER_RAW_ROOT,
    out_root: str | Path = DEFAULT_TSER_DATA_ROOT,
    download: bool = True,
    force_download: bool = False,
    overwrite: bool = False,
    max_train: int | None = None,
    max_test: int | None = None,
    seed: int = 0,
    clip_value: float = 5.0,
) -> Path:
    """Download/parse/preprocess one TSER dataset into this repo's dataset format."""

    spec = get_tser_spec(dataset)
    raw_dir = Path(raw_root) / spec.key
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_path = raw_dir / spec.train_file
    test_path = raw_dir / spec.test_file

    if download:
        download_zenodo_file(spec, spec.train_file, train_path, force=force_download)
        download_zenodo_file(spec, spec.test_file, test_path, force=force_download)
    elif not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing raw TSER files for {spec.key}. Expected {train_path} and {test_path}."
        )

    X_train, y_train, train_meta = parse_tser_ts_file(train_path)
    X_test, y_test, test_meta = parse_tser_ts_file(test_path)
    validate_train_test_compatibility(spec, X_train, X_test, train_meta, test_meta)

    X_train, y_train = deterministic_limit(X_train, y_train, max_count=max_train, seed=seed)
    X_test, y_test = deterministic_limit(X_test, y_test, max_count=max_test, seed=seed + 1)

    X_train, X_test, preprocess = preprocess_tser_arrays(X_train, X_test, clip_value=clip_value)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0).reshape(1, -1)
    train_idx = np.arange(X_train.shape[0], dtype=int)
    test_idx = np.arange(X_train.shape[0], X.shape[0], dtype=int)

    out_dir = Path(out_root) / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"{spec.key}__official_tser__N={X.shape[0]}__w={X.shape[1]}__d={X.shape[2]}"
    data_path = out_dir / f"{artifact_name}.npz"
    if data_path.exists() and not overwrite:
        raise FileExistsError(f"{data_path} exists. Use overwrite=True/--overwrite to regenerate.")

    meta = {
        "dataset_id": spec.key,
        "dataset_name": spec.dataset_name,
        "dataset_family": "tser",
        "task_names": [spec.key],
        "target_name": spec.key,
        "source": f"https://zenodo.org/records/{spec.record_id}",
        "record_id": spec.record_id,
        "description": spec.description,
        "split_source": "official_tser",
        "N": int(X.shape[0]),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "w": int(X.shape[1]),
        "d": int(X.shape[2]),
        "L": 1,
        "raw_files": {"train": str(train_path), "test": str(test_path)},
        "train_meta": train_meta,
        "test_meta": test_meta,
        "preprocess": preprocess,
        "limits": {"max_train": max_train, "max_test": max_test, "seed": int(seed)},
    }
    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta=meta)
    np.savez_compressed(data_path, X=ds.X, y=ds.y)
    (out_dir / f"{artifact_name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "split.npz", train_idx=train_idx, test_idx=test_idx)
    (out_dir / "preprocess.json").write_text(json.dumps(preprocess, indent=2), encoding="utf-8")
    (out_dir / "latest_dataset.txt").write_text(data_path.name + "\n", encoding="utf-8")
    return out_dir


def prepare_hydraulic_dataset(
    *,
    raw_root: str | Path = DEFAULT_HYDRAULIC_RAW_ROOT,
    out_root: str | Path = DEFAULT_TSER_DATA_ROOT,
    download: bool = True,
    force_download: bool = False,
    overwrite: bool = False,
    max_train: int | None = None,
    max_test: int | None = None,
    seed: int = 0,
    clip_value: float = 5.0,
    target: str = "hydraulic_accumulator_pressure",
    test_ratio: float = 0.2,
    split_strategy: str = "random",
    stable_only: bool = True,
    length_seconds: int = 60,
) -> Path:
    """Download/parse/preprocess UCI Hydraulic Systems into window format.

    The raw files store one 60-second operating cycle per row, with sensors
    sampled at 100 Hz, 10 Hz, or 1 Hz. We reshape each cycle into
    ``(length_seconds, 728)`` by keeping within-second samples as channels.
    This makes the high-dimensional JL setting explicit: every time step has
    hundreds of sensor-derived coordinates, then QuaRK projects them to zeta.
    """

    if target not in HYDRAULIC_TARGET_COLUMNS:
        raise ValueError(f"Unknown hydraulic target {target!r}. Choices: {sorted(HYDRAULIC_TARGET_COLUMNS)}")
    test_ratio = float(test_ratio)
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0,1), got {test_ratio}.")
    length_seconds = int(length_seconds)
    if length_seconds <= 0:
        raise ValueError(f"length_seconds must be positive, got {length_seconds}.")

    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    if download:
        download_and_extract_hydraulic(raw_root, force=force_download)
    ensure_hydraulic_files(raw_root)

    X_all = load_hydraulic_sensor_tensor(raw_root, length_seconds=length_seconds)
    profile = load_hydraulic_profile(raw_root)
    if profile.shape[0] != X_all.shape[0]:
        raise ValueError(f"Hydraulic profile rows {profile.shape[0]} do not match X rows {X_all.shape[0]}.")

    source_idx = np.arange(X_all.shape[0], dtype=int)
    if stable_only:
        stable_mask = profile[:, HYDRAULIC_TARGET_COLUMNS["stable_flag"]] == 0
        if not np.any(stable_mask):
            raise ValueError("stable_only=True removed every hydraulic row.")
        X_all = X_all[stable_mask]
        profile = profile[stable_mask]
        source_idx = source_idx[stable_mask]

    y_all = profile[:, HYDRAULIC_TARGET_COLUMNS[target]].astype(float)
    train_source, test_source = split_hydraulic_indices(
        X_all.shape[0],
        test_ratio=test_ratio,
        split_strategy=split_strategy,
        seed=seed,
    )
    X_train, y_train = X_all[train_source], y_all[train_source]
    X_test, y_test = X_all[test_source], y_all[test_source]
    train_orig_idx = source_idx[train_source]
    test_orig_idx = source_idx[test_source]

    train_keep = deterministic_limit_indices(X_train.shape[0], max_count=max_train, seed=seed)
    test_keep = deterministic_limit_indices(X_test.shape[0], max_count=max_test, seed=seed + 1)
    X_train, y_train, train_orig_idx = X_train[train_keep], y_train[train_keep], train_orig_idx[train_keep]
    X_test, y_test, test_orig_idx = X_test[test_keep], y_test[test_keep], test_orig_idx[test_keep]

    X_train, X_test, preprocess = preprocess_tser_arrays(X_train, X_test, clip_value=clip_value)
    X = np.concatenate([X_train, X_test], axis=0).astype(np.float32, copy=False)
    y = np.concatenate([y_train, y_test], axis=0).reshape(1, -1)
    train_idx = np.arange(X_train.shape[0], dtype=int)
    test_idx = np.arange(X_train.shape[0], X.shape[0], dtype=int)

    split_source = f"deterministic_{split_strategy}_{int(round(100 * (1.0 - test_ratio)))}_{int(round(100 * test_ratio))}"
    out_dir = Path(out_root) / HYDRAULIC_DATASET_KEY
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = (
        f"{HYDRAULIC_DATASET_KEY}__target={target}__split={split_source}"
        f"__N={X.shape[0]}__w={X.shape[1]}__d={X.shape[2]}"
    )
    data_path = out_dir / f"{artifact_name}.npz"
    if data_path.exists() and not overwrite:
        raise FileExistsError(f"{data_path} exists. Use overwrite=True/--overwrite to regenerate.")

    meta = {
        "dataset_id": HYDRAULIC_DATASET_KEY,
        "dataset_name": "Condition monitoring of hydraulic systems",
        "dataset_family": "uci_hydraulic_systems",
        "task_names": [target],
        "target_name": target,
        "source": "https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems",
        "download_url": HYDRAULIC_UCI_URL,
        "description": (
            "UCI hydraulic test-rig sensor cycles -> scalar component condition. "
            "Each row is reshaped into seconds with high-dimensional within-second sensor samples."
        ),
        "split_source": split_source,
        "N": int(X.shape[0]),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "w": int(X.shape[1]),
        "d": int(X.shape[2]),
        "L": 1,
        "raw_root": str(raw_root),
        "sensor_specs": [{"name": name, "sampling_rate_hz": rate} for name, rate in HYDRAULIC_SENSOR_SPECS],
        "profile_columns": list(HYDRAULIC_TARGET_COLUMNS),
        "target_column": int(HYDRAULIC_TARGET_COLUMNS[target]),
        "stable_only": bool(stable_only),
        "source_train_indices": train_orig_idx.astype(int).tolist(),
        "source_test_indices": test_orig_idx.astype(int).tolist(),
        "preprocess": preprocess,
        "limits": {"max_train": max_train, "max_test": max_test, "seed": int(seed)},
    }
    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta=meta)
    np.savez_compressed(data_path, X=ds.X, y=ds.y)
    (out_dir / f"{artifact_name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "split.npz", train_idx=train_idx, test_idx=test_idx)
    (out_dir / "preprocess.json").write_text(json.dumps(preprocess, indent=2), encoding="utf-8")
    (out_dir / "latest_dataset.txt").write_text(data_path.name + "\n", encoding="utf-8")
    return out_dir


def download_and_extract_hydraulic(raw_root: Path, *, force: bool = False) -> None:
    zip_path = raw_root / "condition_monitoring_of_hydraulic_systems.zip"
    if force or not zip_path.exists():
        tmp_path = zip_path.with_suffix(zip_path.suffix + ".tmp")
        with urllib.request.urlopen(HYDRAULIC_UCI_URL, timeout=120) as response:
            with tmp_path.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        tmp_path.replace(zip_path)
    if force or not all(find_hydraulic_file(raw_root, f"{name}.txt", required=False) for name, _ in HYDRAULIC_SENSOR_SPECS):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_root)


def ensure_hydraulic_files(raw_root: Path) -> None:
    missing = [
        filename
        for filename in [*(f"{name}.txt" for name, _ in HYDRAULIC_SENSOR_SPECS), "profile.txt"]
        if find_hydraulic_file(raw_root, filename, required=False) is None
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing hydraulic raw files under {raw_root}: {missing}. "
            "Use download=True, or place the extracted UCI files there."
        )


def find_hydraulic_file(raw_root: Path, filename: str, *, required: bool = True) -> Path | None:
    matches = sorted(raw_root.rglob(filename))
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"Missing hydraulic raw file {filename!r} under {raw_root}.")
    return None


def load_hydraulic_sensor_tensor(raw_root: Path, *, length_seconds: int = 60) -> np.ndarray:
    n_rows: int | None = None
    per_second_dim = sum(rate for _, rate in HYDRAULIC_SENSOR_SPECS)
    X: np.ndarray | None = None
    offset = 0
    for sensor_name, rate in HYDRAULIC_SENSOR_SPECS:
        path = find_hydraulic_file(raw_root, f"{sensor_name}.txt")
        values = np.loadtxt(path, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        expected_cols = int(rate) * int(length_seconds)
        if values.shape[1] != expected_cols:
            raise ValueError(
                f"{path} has {values.shape[1]} columns; expected {expected_cols} for "
                f"{length_seconds}s at {rate}Hz."
            )
        if n_rows is None:
            n_rows = int(values.shape[0])
            X = np.empty((n_rows, int(length_seconds), per_second_dim), dtype=np.float32)
        elif values.shape[0] != n_rows:
            raise ValueError(f"{path} has {values.shape[0]} rows, expected {n_rows}.")
        assert X is not None
        for second in range(int(length_seconds)):
            src = slice(second * int(rate), (second + 1) * int(rate))
            dst = slice(offset, offset + int(rate))
            X[:, second, dst] = values[:, src]
        offset += int(rate)
    if X is None:  # pragma: no cover - protected by non-empty sensor specs
        raise RuntimeError("No hydraulic sensors loaded.")
    return X


def load_hydraulic_profile(raw_root: Path) -> np.ndarray:
    path = find_hydraulic_file(raw_root, "profile.txt")
    values = np.loadtxt(path, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] < len(HYDRAULIC_TARGET_COLUMNS):
        raise ValueError(f"{path} has {values.shape[1]} columns, expected at least {len(HYDRAULIC_TARGET_COLUMNS)}.")
    return values


def split_hydraulic_indices(
    n_samples: int,
    *,
    test_ratio: float,
    split_strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 3:
        raise ValueError(f"Need at least 3 hydraulic samples, got {n_samples}.")
    n_test = max(1, int(round(float(test_ratio) * int(n_samples))))
    n_test = min(n_test, int(n_samples) - 1)
    idx = np.arange(int(n_samples), dtype=int)
    if split_strategy == "chronological":
        train_idx = idx[:-n_test]
        test_idx = idx[-n_test:]
    elif split_strategy == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(idx)
        test_idx = np.sort(perm[:n_test])
        train_idx = np.sort(perm[n_test:])
    else:
        raise ValueError(f"Unknown hydraulic split_strategy={split_strategy!r}; expected 'random' or 'chronological'.")
    return train_idx, test_idx


def get_tser_spec(dataset: str) -> TSERDatasetSpec:
    key = str(dataset)
    if key not in TSER_DATASETS:
        raise ValueError(f"Unknown TSER dataset {dataset!r}. Choices: {sorted(TSER_DATASETS)}")
    return TSER_DATASETS[key]


def download_zenodo_file(spec: TSERDatasetSpec, filename: str, out_path: Path, *, force: bool = False) -> None:
    if out_path.exists() and not force:
        return
    url = (
        f"https://zenodo.org/records/{spec.record_id}/files/"
        f"{urllib.parse.quote(filename)}?download=1"
    )
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        with tmp_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    tmp_path.replace(out_path)


def parse_tser_ts_file(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Parse a TSER `.ts` file with scalar target labels."""

    path = Path(path)
    metadata: dict[str, Any] = {}
    data_rows: list[str] = []
    in_data = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if not in_data and line.startswith("@"):
                parts = line.split(maxsplit=1)
                key = parts[0].lower().lstrip("@")
                value = parts[1].strip() if len(parts) > 1 else ""
                metadata[key] = parse_metadata_value(value)
                if key == "data":
                    in_data = True
                continue
            if in_data:
                data_rows.append(line)

    if not data_rows:
        raise ValueError(f"No @data rows found in {path}")

    series: list[list[list[float]]] = []
    labels: list[float] = []
    for row in data_rows:
        dims, label = parse_tser_row(row)
        series.append(dims)
        labels.append(label)

    lengths = {len(dim) for dims in series for dim in dims}
    dims_counts = {len(dims) for dims in series}
    if len(lengths) != 1 or len(dims_counts) != 1:
        raise ValueError(
            f"Expected equal-length multivariate TSER rows in {path}; got lengths={sorted(lengths)} dims={sorted(dims_counts)}."
        )
    n_dims = dims_counts.pop()
    length = lengths.pop()
    X = np.empty((len(series), length, n_dims), dtype=float)
    for i, dims in enumerate(series):
        for j, values in enumerate(dims):
            X[i, :, j] = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=float)
    metadata.update({"n_rows": len(series), "series_length": int(length), "dimensions": int(n_dims)})
    return X, y, metadata


def parse_tser_row(row: str) -> tuple[list[list[float]], float]:
    parts = split_tser_row_top_level(row)
    if len(parts) < 2:
        raise ValueError(f"Expected TSER row with at least one dimension and one target label: {row[:80]!r}")
    dim_parts = parts[:-1]
    label_part = parts[-1]
    dims = [parse_float_sequence(part) for part in dim_parts]
    label = parse_float_token(label_part.strip())
    return dims, label


def parse_float_sequence(text: str) -> list[float]:
    text = text.strip()
    if "(" in text and ")" in text:
        return parse_timestamped_float_sequence(text)
    return [parse_float_token(tok.strip()) for tok in text.split(",") if tok.strip()]


def split_tser_row_top_level(row: str) -> list[str]:
    """Split a TSER data row on top-level ':' separators.

    Timestamped TSER files can contain values like ``(2004-03-10 18:00:00,1.23)``;
    those timestamps contain colons that must not be treated as dimension
    separators.
    """

    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(row):
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        elif ch == ":" and depth == 0:
            parts.append(row[start:i])
            start = i + 1
    parts.append(row[start:])
    return parts


def parse_timestamped_float_sequence(text: str) -> list[float]:
    values: list[float] = []
    depth = 0
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                tuple_text = text[start:i]
                values.append(parse_timestamp_tuple_value(tuple_text))
                start = None
    if values:
        return values
    return [parse_float_token(tok.strip()) for tok in text.split(",") if tok.strip()]


def parse_timestamp_tuple_value(tuple_text: str) -> float:
    if "," not in tuple_text:
        raise ValueError(f"Expected timestamp tuple with comma-separated value, got {tuple_text!r}")
    value_text = tuple_text.rsplit(",", 1)[1].strip()
    return parse_float_token(value_text)


def parse_float_token(token: str) -> float:
    if token in {"?", "NaN", "nan", ""}:
        return float("nan")
    return float(token)


def parse_metadata_value(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def validate_train_test_compatibility(
    spec: TSERDatasetSpec,
    X_train: np.ndarray,
    X_test: np.ndarray,
    train_meta: dict[str, Any],
    test_meta: dict[str, Any],
) -> None:
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(f"Expected train/test X to be rank-3; got {X_train.shape}/{X_test.shape}.")
    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(f"Train/test shape mismatch: {X_train.shape[1:]} vs {X_test.shape[1:]}.")
    if spec.expected_dimensions is not None and X_train.shape[2] != spec.expected_dimensions:
        raise ValueError(
            f"{spec.key} expected d={spec.expected_dimensions}, got d={X_train.shape[2]}."
        )
    if train_meta.get("problemname") and test_meta.get("problemname"):
        if str(train_meta["problemname"]).lower() != str(test_meta["problemname"]).lower():
            raise ValueError(f"Train/test problem names differ: {train_meta['problemname']} vs {test_meta['problemname']}")


def deterministic_limit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_count: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_count is None or int(max_count) <= 0 or X.shape[0] <= int(max_count):
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], size=int(max_count), replace=False))
    return X[idx], y[idx]


def deterministic_limit_indices(n_samples: int, *, max_count: int | None, seed: int) -> np.ndarray:
    n_samples = int(n_samples)
    if max_count is None or int(max_count) <= 0 or n_samples <= int(max_count):
        return np.arange(n_samples, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=int(max_count), replace=False)).astype(int)


def preprocess_tser_arrays(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    clip_value: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    train_flat = X_train.reshape(-1, X_train.shape[-1])
    mean = np.nanmean(train_flat, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    train_imputed = impute_nan_by_channel_mean(X_train, mean)
    test_imputed = impute_nan_by_channel_mean(X_test, mean)
    std = train_imputed.reshape(-1, X_train.shape[-1]).std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    train_std = (train_imputed - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    test_std = (test_imputed - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    clip_value = float(clip_value)
    train_std = np.clip(train_std, -clip_value, clip_value)
    test_std = np.clip(test_std, -clip_value, clip_value)
    preprocess = {
        "impute": "train_channel_nanmean",
        "standardize": "train_channel_mean_std",
        "clip_value": clip_value,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "nan_count_train": int(np.isnan(X_train).sum()),
        "nan_count_test": int(np.isnan(X_test).sum()),
    }
    return train_std.astype(float), test_std.astype(float), preprocess


def impute_nan_by_channel_mean(X: np.ndarray, mean: np.ndarray) -> np.ndarray:
    out = np.asarray(X, dtype=float).copy()
    if not np.isnan(out).any():
        return out
    channel = np.broadcast_to(mean.reshape(1, 1, -1), out.shape)
    return np.where(np.isnan(out), channel, out)


def all_tser_dataset_keys() -> Iterable[str]:
    return TSER_DATASETS.keys()


def all_real_dataset_keys() -> Iterable[str]:
    return REAL_DATASET_KEYS
