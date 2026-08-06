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


def align_to_view(
    metadata: pd.DataFrame, predictions: pd.DataFrame, *, allow_extra: bool = False
) -> np.ndarray:
    """Return predictions aligned to ``metadata`` rows (by key). Raises on any mismatch.

    Guarantees exactly-one prediction per test window: errors on duplicates, missing, or extra.

    ``allow_extra=True`` downgrades the extra-key error to a silent drop. It exists for internal
    tooling that legitimately scores a superset (e.g. one prediction file covering several views);
    a submission must never set it, because an extra key means the file was built against a
    different view than the one being scored, and quietly ignoring that hides the mismatch.
    """
    if predictions.duplicated(subset=KEY_COLS).any():
        raise ValueError("predictions contain duplicate (sample_id, window_idx) keys")
    left = metadata[KEY_COLS].reset_index(drop=True)
    if not allow_extra:
        # Checked as a key-set difference rather than via an outer merge, so the left merge below
        # keeps its exact row order/count semantics.
        extra = pd.MultiIndex.from_frame(predictions[KEY_COLS]).difference(
            pd.MultiIndex.from_frame(left)
        )
        if len(extra):
            raise ValueError(
                f"{len(extra)} predictions are for windows not in this test set "
                f"(first: {list(extra[:3])}); pass allow_extra=True only for internal supersets"
            )
    merged = left.merge(predictions, on=KEY_COLS, how="left", indicator=True)
    missing = int((merged["_merge"] == "left_only").sum())
    if missing:
        raise ValueError(f"{missing} test windows have no prediction")
    if len(merged) != len(left):
        raise ValueError("prediction alignment changed row count (non-unique keys?)")
    return merged[PRED_COL].to_numpy()


# A float class id must be integral to within this tolerance to be accepted and rounded. It exists
# so a third-party float output like 2.9999999 scores as class 3 (as documented), while a genuinely
# non-integral value like 2.4 — which almost always means the file holds scores/probabilities, not
# class ids — is rejected instead of being silently rounded into a different class.
INT_TOL = 1e-6


def check_prediction_values(
    values: np.ndarray, *, regression: bool, n_classes: int | None = None
) -> None:
    """Validate submitted prediction VALUES; raise ``ValueError`` on anything ill-formed.

    Rejected for every task: non-numeric dtypes, NaN and ±Inf. A NaN class id casts to INT64_MIN
    and an out-of-vocabulary id is silently dropped by the confusion matrix, both of which turn a
    broken submission into a plausible-looking score rather than an error.

    Classification additionally requires values that are integral within ``INT_TOL`` and inside the
    task's frozen vocabulary ``range(n_classes)``.

    Regression deliberately accepts out-of-range magnitudes: a fault location outside [0, 100] % is
    a bad prediction that MAE should penalise, not an invalid submission. NaN/Inf are still refused,
    since they poison the aggregate rather than scoring badly.
    """
    arr = np.asarray(values)
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        hint = " (object dtype usually means mixed types or strings)" if arr.dtype == object else ""
        raise ValueError(f"predictions must be numeric, got dtype {arr.dtype!r}{hint}")
    if np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f"predictions must be real-valued, got dtype {arr.dtype!r}")

    asf = arr.astype(float, copy=False)
    bad = ~np.isfinite(asf)
    if bad.any():
        idx = np.flatnonzero(bad)
        n_nan = int(np.isnan(asf).sum())
        raise ValueError(
            f"{idx.size} predictions are not finite ({n_nan} NaN, {idx.size - n_nan} ±Inf); "
            f"first at row(s) {idx[:5].tolist()}"
        )
    if regression:
        return

    off = np.abs(asf - np.round(asf))
    if (off > INT_TOL).any():
        idx = np.flatnonzero(off > INT_TOL)
        raise ValueError(
            f"{idx.size} classification predictions are not integral class ids "
            f"(tolerance {INT_TOL:g}); first at row(s) {idx[:5].tolist()} "
            f"= {asf[idx[:5]].tolist()} — submit class ids, not scores/probabilities"
        )
    if n_classes is not None:
        ids = np.round(asf).astype(int)
        oob = (ids < 0) | (ids >= n_classes)
        if oob.any():
            idx = np.flatnonzero(oob)
            raise ValueError(
                f"{idx.size} predictions are outside the task vocabulary "
                f"[0, {n_classes - 1}]; first at row(s) {idx[:5].tolist()} "
                f"= {ids[idx[:5]].tolist()}"
            )
