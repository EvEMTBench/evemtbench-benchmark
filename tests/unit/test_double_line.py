"""Portable double_line view (multi_grid corpus): projection, bus-canonicalization, per-window
z-score, and transform composition. Pure numpy — no data, no torch."""

import numpy as np
import pytest

from evemtbench.tasks import line_view as lv
from evemtbench.tasks.task import build_input_transform, normalize_per_window

_DL_NAMES = [
    "EndA_Isec_L1_A", "EndA_Isec_L2_A", "EndA_Isec_L3_A",
    "EndA_Usec_L1_V", "EndA_Usec_L2_V", "EndA_Usec_L3_V",
    "EndB_Isec_L1_A", "EndB_Isec_L2_A", "EndB_Isec_L3_A",
    "EndB_Usec_L1_V", "EndB_Usec_L2_V", "EndB_Usec_L3_V",
]  # fmt: skip


def test_is_double_line_view_by_tag_and_names():
    assert lv.is_double_line_view(_DL_NAMES, "double_line")
    assert lv.is_double_line_view(_DL_NAMES, None)  # fallback: EndA_/EndB_ names
    assert not lv.is_double_line_view(["Bus_1_Line_0_Isec_L1_A"], None)  # reference-grid names


def test_double_line_channels_local_line_global():
    assert lv.double_line_channels("local") == 6
    assert lv.double_line_channels("line") == 12
    with pytest.raises(ValueError):
        lv.double_line_channels("global")  # portable 2-terminal view has no GLOBAL


def test_project_local_line_no_swap():
    x = np.arange(3 * 8 * 12, dtype=np.float32).reshape(3, 8, 12)
    loc = lv.project_double_line(x, "local", near_end=None)
    assert loc.shape == (3, 8, 6) and np.array_equal(loc, x[:, :, :6])  # EndA
    ln = lv.project_double_line(x, "line", near_end=None)
    assert ln.shape == (3, 8, 12) and np.array_equal(ln, x)


def test_project_canonicalizes_terminals_by_bus():
    # near_end=1 => EndB is the smaller-bus terminal, so it must land first.
    x = np.arange(2 * 8 * 12, dtype=np.float32).reshape(2, 8, 12)
    near_end = np.array([0, 1])
    loc = lv.project_double_line(x, "local", near_end=near_end)
    assert np.array_equal(loc[0], x[0, :, :6])  # row 0: EndA near
    assert np.array_equal(loc[1], x[1, :, 6:12])  # row 1: EndB near (swapped)
    ln = lv.project_double_line(x, "line", near_end=near_end)
    assert np.array_equal(ln[1, :, :6], x[1, :, 6:12]) and np.array_equal(ln[1, :, 6:], x[1, :, :6])


def test_project_rejects_wrong_width_and_global():
    with pytest.raises(ValueError):
        lv.project_double_line(np.zeros((2, 8, 10), np.float32), "line")
    with pytest.raises(ValueError):
        lv.project_double_line(np.zeros((2, 8, 12), np.float32), "global")


def test_normalize_per_window_is_instance_zscore():
    rng = np.random.default_rng(0)
    x = rng.normal(50.0, 5.0, size=(4, 16, 6)).astype(np.float32)  # a "20 kV"-scale window
    z = normalize_per_window(x)
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-4)  # each window/channel centred
    assert np.allclose(z.std(axis=1), 1.0, atol=1e-2)  # and unit-scaled over time
    # scale invariance: a 100x larger window normalizes to (nearly) the same tensor
    assert np.allclose(z, normalize_per_window(x * 100.0), atol=1e-3)


def test_build_input_transform_composition_and_names():
    assert build_input_transform(False, "none") is None
    assert build_input_transform(False, "per_window").__name__ == "normalize_per_window"
    t = build_input_transform(True, "per_window")
    assert t.__name__ == "add_zero_sequence+normalize_per_window"
    out = t(np.zeros((2, 8, 12), np.float32))
    assert out.shape[-1] == 16  # 12 + 2 zero-seq channels per 6-ch cubicle
    with pytest.raises(ValueError):
        build_input_transform(False, "bogus")
