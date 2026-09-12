import numpy as np

from eeg_lr.csp import BinaryCSP


def test_binary_csp_returns_finite_features() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 6, 100))
    y = np.repeat([0, 1], 20)
    # Inject opposite class-specific variance into two channels.
    x[y == 0, 0] *= 3
    x[y == 1, 1] *= 3
    csp = BinaryCSP(n_components=4)
    features = csp.fit_transform(x, y)
    assert features.shape == (40, 4)
    assert np.isfinite(features).all()
    assert csp.filters_.shape == (4, 6)
