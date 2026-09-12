import numpy as np

from eeg_lr.features import BandPowerFeatures


def test_bandpower_shape_and_finite_values() -> None:
    sfreq = 160.0
    times = np.arange(561) / sfreq
    alpha = np.sin(2 * np.pi * 10 * times)
    beta = np.sin(2 * np.pi * 20 * times)
    x = np.stack([np.stack([alpha, beta]), np.stack([2 * alpha, beta])])
    features = BandPowerFeatures(sfreq=sfreq).fit_transform(x)
    assert features.shape == (2, 4)
    assert np.isfinite(features).all()
    # Doubling channel-zero amplitude should increase its alpha log power.
    assert features[1, 0] > features[0, 0]
