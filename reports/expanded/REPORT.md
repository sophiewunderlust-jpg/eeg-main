# Expanded experiment: up to 80 training participants

S001–S080 are development/training; S081–S109 were kept untouched until development selection. Subjects previously used for evaluation, S011–S030, are explicitly reassigned to development. Preprocessing and the six candidates were fixed before this experiment. No test-driven model changes.

## Development selection

Five-fold subject-separated validation. Each usable person has equal weight in the mean. These scores select a model; they are not unbiased estimates for the selected model.

| Model | Balanced accuracy |
|---|---:|
| logreg | 59.19% |
| lda | 59.05% |
| pca_8 | 55.48% |
| pca_16 | 59.77% |
| pca_32 | 60.21% |
| pca_64 | 59.39% |

Frozen choice: **pca_32**. Training trials: **3417**; test trials: **1084**. Scored test people: **25**. Metadata exclusions are preserved in the recording audits.

## One final test on S081–S109

| Training / model | Balanced accuracy |
|---|---:|
| selected_80 | 59.08% |
| same_model_10 | 55.35% |
| majority | 50.00% |
| timing_only | 49.44% |

Selected-model 95% subject-bootstrap interval: **55.86–62.29%**.

Same-model comparison on the same test people: training on up to 80 versus 10 gives **+3.73 percentage points**, paired 95% bootstrap interval **[+1.58, +5.96]**.

The ten-person comparator uses the configuration selected using 80 development people, so it isolates the final fitting-data difference conditional on that configuration, not the entire model-selection process.

## Scope and limitations

Only band-power candidates are included: logistic regression, shrinkage LDA, and PCA (8/16/32/64) + shrinkage LDA. Preprocessing, standardization, clipping, PCA, and LDA settings match the previous comparisons. Only learned transforms fit inside training folds; deterministic filtering/feature extraction is per recording. CSP is not retrained in this focused expansion. Subject IDs form a non-random development/test split. Trial rejection and unsupported-recording exclusions limit the population represented. No clinical, online, or neural-specific performance guarantee is made.

The original exported model is not changed. The prediction CLI still uses the original logistic-regression artifact. See protocol.json, selection.json, folds.json, the per-person score files and event audits.

```bash
python -m eeg_lr.expand_experiment --data-dir /path/to/data
```
