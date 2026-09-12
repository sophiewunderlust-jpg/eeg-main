# Logistic regression versus shrinkage LDA

Follow-up experiment requested after the original reserved-test results were known. No changes to the exported model or original experiment. This is not a fresh confirmatory test.

Only the classifier changes. Identical train-fitted CSP/band-power features, scaling, clipping, epochs and folds are shared. LDA uses scikit-learn LSQR with automatic Ledoit-Wolf shrinkage and training-estimated class priors. There is no hyperparameter search.

The sklearn fit is unchanged. Binary decision scores use explicit NumPy einsum for both classifiers; standard matrix multiplication raised a runtime warning at prediction on this Mac. Equivalence is unit tested. Warnings are not suppressed and non-finite coefficients/scores fail the experiment. The underlying runtime cause is not established.

| Features + classifier | Development: unseen person | Development: unseen run | Follow-up: S011–030 |
|---|---:|---:|---:|
| bandpower_logreg | 56.0% ± 11.0% | 60.8% ± 4.9% | 52.1% ± 5.2% |
| bandpower_lda | 56.6% ± 11.6% | 60.2% ± 5.0% | 53.7% ± 5.1% |
| csp_logreg | 54.4% ± 10.6% | 54.7% ± 4.6% | 53.1% ± 6.0% |
| csp_lda | 54.2% ± 9.5% | 55.9% ± 3.9% | 53.1% ± 6.4% |

Balanced accuracy averages left and right recall; 50% is the majority baseline. ± denotes sample SD across folds/people, not confidence intervals.

## Paired differences on the previously inspected test people

- bandpower: LDA minus LR = +1.60 percentage points; 95% paired subject-bootstrap interval [+0.60, +2.93]; LDA better for 10/20 people, 10 ties.
- csp: LDA minus LR = +0.01 percentage points; 95% paired subject-bootstrap interval [-1.35, +1.25]; LDA better for 5/20 people, 9 ties.

Bootstrap: resample people with replacement, 10,000 replicates, seed 7. An interval crossing zero does not establish a consistent improvement. Development folds overlap in training data; their bootstrap intervals are descriptive only. No multiplicity adjustment or new-subject confirmation is claimed.

Reproduce from the repository root:

```bash
python -m eeg_lr.compare_lda
```

Requires the prepared folders created by `eeg-finalize`. All per-fold metrics and confusion counts are stored alongside this report.

LDA reference: https://scikit-learn.org/1.5/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
