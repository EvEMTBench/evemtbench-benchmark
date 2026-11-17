"""Time Series Transformer module (baselines/tst.py): token geometry, output shapes for narrow and
wide inputs, parameter accounting, the three head kinds, and CPU determinism. Needs torch."""

import pytest

pytest.importorskip("torch")

import torch

from evemtbench.baselines.tst import _TST, TSTBaseline, patch_geometry

CFG = dict(d_model=128, n_heads=8, n_layers=3, dim_ff=256, dropout=0.1, patch_len=16, stride=8)
LENGTH = 480


def _n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _expected_params(in_ch: int, out_dim: int, n_tokens: int = 59, kernel: int = 16) -> int:
    d, ff, layers = CFG["d_model"], CFG["dim_ff"], CFG["n_layers"]
    embed = kernel * in_ch * d + d
    embed_norm = 2 * d  # LayerNorm on the patch tokens, weight + bias
    pos = n_tokens * d
    attn = 4 * d * d + 4 * d  # in-projection (3 d^2 + 3 d) + out-projection (d^2 + d)
    feed_forward = d * ff + ff + ff * d + d
    norms = 2 * 2 * d  # two BatchNorm1d, weight + bias each
    head = n_tokens * d * out_dim + out_dim
    return embed + embed_norm + pos + layers * (attn + feed_forward + norms) + head


def test_patch_geometry_benchmark_window_and_short_window():
    assert patch_geometry(480, 16, 8) == (16, 8, 59)
    assert patch_geometry(16, 16, 8) == (16, 8, 1)
    assert patch_geometry(8, 16, 8) == (8, 8, 1)  # shorter than a patch: one full-length patch
    with pytest.raises(ValueError):
        patch_geometry(0, 16, 8)


@pytest.mark.parametrize("in_ch", [6, 12, 96])
def test_forward_shape_for_narrow_and_wide_inputs(in_ch):
    torch.manual_seed(0)
    model = _TST(in_ch, 2, LENGTH, **CFG).eval()
    assert model.n_tokens == 59
    with torch.no_grad():
        out = model(torch.randn(4, LENGTH, in_ch))
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("out_dim", [2, 9, 1])  # binary, multiclass, regression
def test_head_output_size(out_dim):
    model = _TST(6, out_dim, LENGTH, **CFG).eval()
    with torch.no_grad():
        assert model(torch.randn(3, LENGTH, 6)).shape == (3, out_dim)


@pytest.mark.parametrize("in_ch,out_dim", [(6, 2), (12, 1), (96, 9)])
def test_parameter_count_matches_the_architecture(in_ch, out_dim):
    assert _n_params(_TST(in_ch, out_dim, LENGTH, **CFG)) == _expected_params(in_ch, out_dim)


def test_only_the_patch_embedding_depends_on_the_channel_count():
    narrow, wide = _TST(6, 2, LENGTH, **CFG), _TST(96, 2, LENGTH, **CFG)
    d, k = CFG["d_model"], CFG["patch_len"]
    assert _n_params(wide) - _n_params(narrow) == k * (96 - 6) * d
    shapes_n = {n: tuple(p.shape) for n, p in narrow.named_parameters()}
    shapes_w = {n: tuple(p.shape) for n, p in wide.named_parameters()}
    assert {n for n in shapes_n if shapes_n[n] != shapes_w[n]} == {"embed.weight"}


def test_same_seed_gives_identical_weights_and_outputs_on_cpu():
    def build():
        torch.manual_seed(7)
        return _TST(12, 3, LENGTH, **CFG).eval()

    a, b = build(), build()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert na == nb and torch.equal(pa, pb), na
    x = torch.randn(5, LENGTH, 12)
    with torch.no_grad():
        assert torch.equal(a(x), b(x))
        assert torch.equal(a(x), a(x))  # eval mode: no dropout, running BatchNorm statistics


def test_train_mode_backward_reaches_every_parameter():
    torch.manual_seed(0)
    model = _TST(6, 2, LENGTH, **CFG).train()
    loss = model(torch.randn(8, LENGTH, 6)).pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_patch_tokens_are_normalised_whatever_the_input_amplitude():
    # Regression (issue #26): the scaled patch tokens fed the first attention block at a scale that
    # grew with the input amplitude, and training diverged on heavy-tailed fault transients. After
    # the token LayerNorm each token has zero mean and unit variance over d_model, at any amplitude.
    torch.manual_seed(0)
    model = _TST(12, 2, LENGTH, **CFG).eval()
    seen = {}
    model.embed_norm.register_forward_hook(lambda m, i, o: seen.__setitem__("tok", o.detach()))
    x = torch.randn(4, LENGTH, 12)
    for amplitude in (1.0, 1e3):
        with torch.no_grad():
            out = model(x * amplitude)
        tok = seen["tok"]
        assert torch.isfinite(out).all()
        assert tok.mean(-1).abs().max() < 1e-4
        assert (tok.var(-1, unbiased=False) - 1).abs().max() < 1e-2


def test_baseline_defaults_are_the_declared_configuration():
    b = TSTBaseline(device="cpu")
    got = {k: getattr(b, k) for k in CFG}
    assert got == CFG
    # the shared deep-baseline training regime (same defaults as mlp/gru/cnn/resnet)
    assert (b.epochs, b.batch_size, b.lr, b.weight_decay, b.patience) == (60, 256, 1e-3, 1e-4, 12)
    inner = b._build_inner(6, 2, LENGTH)
    assert inner.n_tokens == 59 and _n_params(inner) == _expected_params(6, 2)
