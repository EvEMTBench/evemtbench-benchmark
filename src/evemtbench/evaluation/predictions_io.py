"""Canonical predictions format + alignment to a task's test set.

A predictions file is a table with one row per test window, keyed by ``(sample_id, window_idx)``
with a ``prediction`` column (and optional ``score``). Alignment is by key, never by row order,
so a third party can submit a predictions file and have it audited by ``evemtbench-evaluate``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLS = ["sample_id", "window_idx"]
PRED_COL = "prediction"
SCORE_COL = "score"


def predictions_frame(sample_ids, window_idx, predictions, scores=None) -> pd.DataFrame:
    """Build a canonical predictions frame."""
    data = {
        "sample_id": np.asarray(sample_ids),
        "window_idx": np.asarray(window_idx),
        PRED_COL: np.asarray(predictions),
    }
    if scores is not None:
        data[SCORE_COL] = np.asarray(scores)
    return pd.DataFrame(data)


def save_predictions(df: pd.DataFrame, path: str | os.PathLike) -> None:
    _validate_columns(df)
    path = Path(path)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def load_predictions(path: str | os.PathLike) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    _validate_columns(df)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in KEY_COLS + [PRED_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"predictions missing columns {missing}; have {list(df.columns)}")


def align_to_view(metadata: pd.DataFrame, predictions: pd.DataFrame) -> np.ndarray:
    """Return predictions aligned to ``metadata`` rows (by key). Raises on any mismatch.

    Guarantees exactly-one prediction per test window: errors on duplicates, missing, or extra.
    """
    if predictions.duplicated(subset=KEY_COLS).any():
        raise ValueError("predictions contain duplicate (sample_id, window_idx) keys")
    left = metadata[KEY_COLS].reset_index(drop=True)
    merged = left.merge(predictions, on=KEY_COLS, how="left", indicator=True)
    missing = int((merged["_merge"] == "left_only").sum())
    if missing:
        raise ValueError(f"{missing} test windows have no prediction")
    if len(merged) != len(left):
        raise ValueError("prediction alignment changed row count (non-unique keys?)")
    return merged[PRED_COL].to_numpy()
