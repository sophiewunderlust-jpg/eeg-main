"""Leakage-aware CSP + logistic-regression evaluation for prepared EEG epochs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .csp import BinaryCSP
from .features import BandPowerFeatures
from .dataset import CHANNELS


@dataclass(frozen=True)
class FoldResult:
    model: str
    split: str
    held_out: str
    n_train: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    left_recall: float
    right_recall: float
    true_left: int
    left_as_right: int
    right_as_left: int
    true_right: int


class StableBinaryLogisticRegression(LogisticRegression):
    """Binary logistic regression with a runtime-stable decision product."""

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("coef_", "intercept_"))
        x = np.asarray(x, dtype=np.float64)
        return np.einsum("ij,j->i", x, self.coef_[0]) + self.intercept_[0]


def load_prepared(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all S###.npz files while retaining subject and run groups."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    runs: list[np.ndarray] = []
    files = sorted(data_dir.glob("S[0-9][0-9][0-9].npz"))
    if not files:
        raise ValueError(f"No prepared S###.npz archives found in {data_dir}")
    expected_shape: tuple[int, int] | None = None
    for path in files:
        with np.load(path) as archive:
            if list(archive['ch_names']) != list(CHANNELS) or float(archive['sfreq']) != 160.0:
                raise ValueError(f'Channel metadata or sampling rate mismatch in {path}')
            if float(archive['tmin']) != 0.5 or float(archive['tmax']) != 4.0:
                raise ValueError(f'Evaluator requires the fixed 0.5-4.0 second window: {path}')
            # Archives preserve MNE's SI unit (volts). Work in microvolts for
            # stable covariance calculations. Average reference makes one of
            # the 64 channels exactly redundant, so omit the final channel for
            # CSP; it is reconstructible from the other 63 channels.
            x = archive["X"].astype(np.float64, copy=False)[:, :-1, :] * 1e6
            y = archive["y"].astype(np.int8, copy=False)
            run = archive["runs"].astype(np.int16, copy=False)
        if x.ndim != 3 or len(x) != len(y) or len(y) != len(run):
            raise ValueError(f"Inconsistent array shapes in {path}")
        if not np.isfinite(x).all() or not np.isin(y, [0,1]).all():
            raise ValueError(f'Invalid EEG values or labels in {path}')
        if expected_shape is None:
            expected_shape = x.shape[1:]
        elif x.shape[1:] != expected_shape:
            raise ValueError(f"Channel/time shape differs in {path}")
        subject = int(path.stem[1:])
        xs.append(x)
        ys.append(y)
        subjects.append(np.full(len(y), subject, dtype=np.int16))
        runs.append(run)
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(subjects),
        np.concatenate(runs),
    )


def split_groups(split: str, subjects: np.ndarray, runs: np.ndarray) -> np.ndarray:
    if split == "subject-grouped":
        return subjects
    if split == "run-grouped":
        # Holding out R04, R08, or R12 tests a recording run not used for training.
        return runs
    raise ValueError(f"Unknown split: {split}")


def csp_logreg() -> Pipeline:
    """Return a fresh trainable pipeline for one cross-validation fold."""
    return Pipeline(
        [
            (
                "csp",
                BinaryCSP(n_components=8),
            ),
            ("standardize", StandardScaler()),
            (
                "clip_extremes",
                FunctionTransformer(
                    np.clip,
                    kw_args={"a_min": -20.0, "a_max": 20.0},
                    validate=False,
                ),
            ),
            (
                "logistic_regression",
                StableBinaryLogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=7,
                ),
            ),
        ]
    )


def bandpower_logreg() -> Pipeline:
    return Pipeline(
        [
            ("bandpower", BandPowerFeatures(sfreq=160.0)),
            ("standardize", StandardScaler()),
            (
                "clip_extremes",
                FunctionTransformer(
                    np.clip,
                    kw_args={"a_min": -20.0, "a_max": 20.0},
                    validate=False,
                ),
            ),
            (
                "logistic_regression",
                StableBinaryLogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=7,
                ),
            ),
        ]
    )


def score_predictions(
    *,
    model: str,
    split: str,
    held_out: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
) -> FoldResult:
    true_left, left_as_right, right_as_left, true_right = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()
    n_left = true_left + left_as_right
    n_right = true_right + right_as_left
    return FoldResult(
        model=model,
        split=split,
        held_out=held_out,
        n_train=len(y_train),
        n_test=len(y_test),
        accuracy=float(accuracy_score(y_test, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        left_recall=float(true_left / n_left) if n_left else 0.0,
        right_recall=float(true_right / n_right) if n_right else 0.0,
        true_left=int(true_left),
        left_as_right=int(left_as_right),
        right_as_left=int(right_as_left),
        true_right=int(true_right),
    )


def evaluate(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
    *,
    split: str,
) -> list[FoldResult]:
    groups = split_groups(split, subjects, runs)
    logo = LeaveOneGroupOut()
    results: list[FoldResult] = []
    for train_idx, test_idx in logo.split(x, y, groups):
        held_value = int(np.unique(groups[test_idx]).item())
        held_out = f"S{held_value:03d}" if split == "subject-grouped" else f"R{held_value:02d}"
        y_train, y_test = y[train_idx], y[test_idx]

        for model_name, model in (
            ("csp_logreg", csp_logreg()),
            ("bandpower_logreg", bandpower_logreg()),
        ):
            model.fit(x[train_idx], y_train)
            results.append(
                score_predictions(
                    model=model_name,
                    split=split,
                    held_out=held_out,
                    y_train=y_train,
                    y_test=y_test,
                    predictions=model.predict(x[test_idx]),
                )
            )

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((len(train_idx), 1)), y_train)
        results.append(
            score_predictions(
                model="majority_baseline",
                split=split,
                held_out=held_out,
                y_train=y_train,
                y_test=y_test,
                predictions=dummy.predict(np.zeros((len(test_idx), 1))),
            )
        )
    return results


def write_results(results: list[FoldResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with (output_dir / "fold_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict[str, float | int]] = {}
    for model in sorted({result.model for result in results}):
        selected = [result for result in results if result.model == model]
        balanced = np.asarray([result.balanced_accuracy for result in selected])
        summary[model] = {
            "n_folds": len(selected),
            "mean_balanced_accuracy": float(balanced.mean()),
            "std_balanced_accuracy": float(balanced.std(ddof=1)) if len(balanced) > 1 else 0.0,
            "min_balanced_accuracy": float(balanced.min()),
            "max_balanced_accuracy": float(balanced.max()),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    if results[0].split == "subject-grouped":
        lines = [
            "# Held-out subject failure analysis",
            "",
            "Balanced accuracy is the average of left and right recall.",
            "",
            "| Model | Subject | Balanced accuracy | Left recall | Right recall |",
            "|---|---:|---:|---:|---:|",
        ]
        for result in sorted(
            (r for r in results if r.model != "majority_baseline"),
            key=lambda result: (result.model, result.balanced_accuracy),
        ):
            lines.append(
                f"| {result.model} | {result.held_out} | "
                f"{result.balanced_accuracy:.3f} | {result.left_recall:.3f} | "
                f"{result.right_recall:.3f} |"
            )
        (output_dir / "failure_analysis.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="eeg-evaluate")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("subject-grouped", "run-grouped"),
        default="subject-grouped",
    )
    args = parser.parse_args()
    x, y, subjects, runs = load_prepared(args.data_dir.expanduser().resolve())
    print(
        f"Loaded {len(y)} epochs from {len(np.unique(subjects))} subjects; "
        f"left={int((y == 0).sum())}, right={int((y == 1).sum())}"
    )
    results = evaluate(x, y, subjects, runs, split=args.split)
    write_results(results, args.output_dir.expanduser().resolve())
    for model in ("csp_logreg", "bandpower_logreg", "majority_baseline"):
        values = [r.balanced_accuracy for r in results if r.model == model]
        print(f"{model}: balanced accuracy {np.mean(values):.3f} +/- {np.std(values, ddof=1):.3f}")


if __name__ == "__main__":
    main()
