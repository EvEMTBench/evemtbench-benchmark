"""The designed-not-built seams stay importable, honest (raise loudly), and layer-clean."""

import numpy as np
import pytest

from evemtbench.evaluation import robustness
from evemtbench.tasks import vantage


def test_vantage_seam_raises_with_design_pointer():
    with pytest.raises(NotImplementedError, match="docs/design/selectivity.md"):
        vantage.iter_vantages("double_line")
    with pytest.raises(NotImplementedError, match="docs/design/selectivity.md"):
        vantage.vantage_view(None, 0)


def test_robustness_seam_raises_with_design_pointer():
    rng = np.random.default_rng(0)
    x = np.zeros((2, 4, 6), np.float32)
    with pytest.raises(ValueError, match="unknown perturbation"):
        robustness.perturb(x, "not_a_kind", 1.0, rng)
    with pytest.raises(NotImplementedError, match="docs/design/robustness.md"):
        robustness.perturb(x, "awgn", 30.0, rng)
    with pytest.raises(NotImplementedError, match="docs/design/robustness.md"):
        robustness.curve(None, None, "awgn", [30.0], seeds=[0])
    assert len(set(robustness.PERTURBATION_KINDS)) == 6  # frozen taxonomy, no dupes


def test_gnn_seam_is_a_baseline_and_raises():
    from evemtbench.baselines.gnn import GNNBaseline
    from evemtbench.baselines.interface import Baseline

    model = GNNBaseline(seed=0)
    assert isinstance(model, Baseline)  # satisfies the protocol shape today
    with pytest.raises(NotImplementedError, match="docs/design/cross_topology.md"):
        model.fit(None)
    with pytest.raises(NotImplementedError, match="docs/design/cross_topology.md"):
        model.predict(None)
