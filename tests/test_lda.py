import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits

from eeg_lr.compare_lda import lda, predictions
from eeg_lr.evaluate import bandpower_logreg


def test_lda_configuration_and_finite_predictions():
    assert lda().solver == 'lsqr'
    assert lda().shrinkage == 'auto'
    rng = np.random.default_rng(19)
    x = rng.normal(size=(24, 3, 561))
    y = np.arange(24) % 2
    with threadpool_limits(limits=1):
        output = predictions(x[:20], y[:20], x[20:], bandpower_logreg)
        expected = bandpower_logreg().fit(x[:20], y[:20]).predict(x[20:])
    np.testing.assert_array_equal(output['logreg'], expected)
    assert output['lda'].shape == (4,)
    assert np.isin(output['lda'], [0, 1]).all()


def test_explicit_lda_score_matches_sklearn():
    rng = np.random.default_rng(27)
    x = rng.normal(size=(60, 8))
    y = np.arange(60) % 2
    with threadpool_limits(limits=1):
        reference = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(x, y)
        actual = lda().fit(x, y)
        np.testing.assert_allclose(actual.decision_function(x), reference.decision_function(x), atol=1e-12)
        np.testing.assert_array_equal(actual.predict(x), reference.predict(x))
