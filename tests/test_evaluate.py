import numpy as np
import pytest

from eeg_lr.evaluate import StableBinaryLogisticRegression, split_groups


def test_subject_grouping() -> None:
    subjects = np.array([1, 1, 2, 2])
    runs = np.array([4, 8, 4, 8])
    assert np.array_equal(split_groups("subject-grouped", subjects, runs), subjects)


def test_run_grouping() -> None:
    subjects = np.array([1, 1, 2, 2])
    runs = np.array([4, 8, 4, 8])
    assert np.array_equal(split_groups("run-grouped", subjects, runs), runs)


def test_unknown_grouping_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        split_groups("random", np.array([1]), np.array([4]))


def test_stable_logistic_prediction() -> None:
    x = np.array([[-2.0, 0.0], [-1.0, 0.2], [1.0, -0.2], [2.0, 0.0]])
    y = np.array([0, 0, 1, 1])
    model = StableBinaryLogisticRegression(solver="liblinear", random_state=7).fit(x, y)
    assert np.array_equal(model.predict(x), y)
