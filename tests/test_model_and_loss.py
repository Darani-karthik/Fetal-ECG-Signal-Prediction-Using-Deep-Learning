import numpy as np
import torch

from fecg.data.dataset import build_features, build_target, window_starts
from fecg.losses import FetalScalogramLoss, correlation_loss, synthesize
from fecg.models import UNet
from fecg.transforms import CWTConfig, cwt, icwt, synthesis_weights


def test_unet_preserves_spatial_size():
    model = UNet(in_channels=8, out_channels=2, base_channels=8, depth=4)
    assert model(torch.randn(2, 8, 64, 512)).shape == (2, 2, 64, 512)


def test_unet_handles_sizes_not_divisible_by_the_depth():
    model = UNet(in_channels=8, out_channels=2, base_channels=4, depth=3)
    assert model(torch.randn(1, 8, 50, 70)).shape == (1, 2, 50, 70)


def test_unet_uses_skip_connections():
    """Every encoder stage must feed the decoder: gradients reach the first conv."""
    model = UNet(base_channels=4, depth=2)
    model(torch.randn(1, 8, 16, 16)).sum().backward()
    assert model.inc[0].weight.grad.abs().sum() > 0


def test_feature_and_target_layout():
    rng = np.random.default_rng(0)
    abd = (rng.normal(size=(4, 8, 16)) + 1j * rng.normal(size=(4, 8, 16))).astype(np.complex64)
    x = build_features(abd)
    assert x.shape == (8, 8, 16)
    assert x[:4].min() >= 0  # log1p magnitude
    assert np.abs(x[4:]).max() <= 1  # phase / pi
    y = build_target(abd[0], target_scale=2.0)
    np.testing.assert_allclose(y[0], abd[0].real / 2.0, rtol=1e-6)
    np.testing.assert_allclose(y[1], abd[0].imag / 2.0, rtol=1e-6)


def test_window_starts_cover_the_whole_record():
    s = window_starts(1000, 512, 128)
    assert s[0] == 0 and s[-1] == 1000 - 512 and all(b > a for a, b in zip(s, s[1:]))


def test_torch_synthesis_matches_numpy_icwt():
    cfg = CWTConfig()
    w = cwt(np.random.default_rng(0).standard_normal(300), cfg)
    scale = 1.7
    ri = torch.from_numpy(np.stack([w.real, w.imag])[None] / scale).float()
    weights = torch.as_tensor(synthesis_weights(cfg) * scale, dtype=torch.float32)
    np.testing.assert_allclose(synthesize(ri, weights)[0].numpy(), icwt(w, cfg), rtol=1e-3, atol=1e-4)


def test_loss_is_zero_for_a_perfect_prediction_and_differentiable():
    cfg = CWTConfig()
    loss_fn = FetalScalogramLoss(cfg, target_scale=1.0, mag_weight=0.5, signal_weight=1.0)
    w = cwt(np.random.default_rng(0).standard_normal(64), cfg)
    target = torch.from_numpy(np.stack([w.real, w.imag])[None]).float()
    signal = synthesize(target, loss_fn.weights)
    total, _ = loss_fn(target.clone(), target, signal)
    assert float(total) < 1e-5

    pred = (target + 0.3).requires_grad_(True)
    total, parts = loss_fn(pred, target, signal)
    total.backward()
    assert float(total) > 0 and pred.grad.abs().sum() > 0 and set(parts) == {"coef", "mag", "signal", "corr"}


def test_correlation_loss_is_scale_invariant_and_penalises_anticorrelation():
    x = torch.randn(4, 200)
    assert float(correlation_loss(3.0 * x + 1.0, x)) < 1e-4
    assert float(correlation_loss(-x, x)) > 1.99
