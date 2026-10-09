"""Load data and separate train, validation and test before preprocessing."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_config(path):
    config_path = Path(path).resolve()
    # Config files live directly under configs/; data paths are project-relative.
    root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("data_path", "artifact_dir", "report_dir"):
        config[key] = root / config[key]
    validation = config["validation_fraction"]
    test = config["test_fraction"]
    if not (0 < validation < 1 and 0 < test < 1 and validation + test < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than 1.")
    if not 0 < config["contact_fraction"] <= 1:
        raise ValueError("contact_fraction must be in (0, 1].")
    return config


def load_data(config):
    path = config["data_path"]
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Run python tools/download_data.py first.")
    frame = pd.read_csv(path, sep=config["separator"])
    target = config["target"]
    if target not in frame:
        raise ValueError(f"Missing target column: {target}; check the CSV separator.")
    labels = frame[target].astype("string").str.strip().str.lower()
    mapping = {config["negative_label"].lower(): 0, config["positive_label"].lower(): 1}
    y = labels.map(mapping)
    if y.isna().any():
        raise ValueError("Target contains missing or unexpected labels.")
    X = frame.drop(columns=[target, *config["drop_columns"]], errors="ignore")
    if X.empty or len(frame) < 30:
        raise ValueError("Need feature columns and at least 30 rows.")
    return X, y.astype(int)


def split_data(X, y, config):
    indices = np.arange(len(X))
    validation_fraction = config["validation_fraction"]
    test_fraction = config["test_fraction"]
    strategy = config["split_strategy"]
    if strategy == "temporal":
        # Assumes CSV rows are already ordered chronologically.
        train_end = int(len(X) * (1 - validation_fraction - test_fraction))
        validation_end = int(len(X) * (1 - test_fraction))
        groups = (indices[:train_end], indices[train_end:validation_end], indices[validation_end:])
    elif strategy == "stratified":
        train_validation, test = train_test_split(
            indices, test_size=test_fraction, stratify=y, random_state=config["random_state"]
        )
        train, validation = train_test_split(
            train_validation,
            test_size=validation_fraction / (1 - test_fraction),
            stratify=y.iloc[train_validation], random_state=config["random_state"],
        )
        groups = (train, validation, test)
    else:
        raise ValueError("split_strategy must be temporal or stratified.")
    result = {}
    for name, positions in zip(("train", "validation", "test"), groups):
        subset = (X.iloc[positions].copy(), y.iloc[positions].copy())
        if subset[1].nunique() != 2:
            raise ValueError(f"{name} must contain both labels; inspect the dataset/split.")
        result[name] = subset
    return result
