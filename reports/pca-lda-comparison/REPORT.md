# PCA before shrinkage LDA: exploratory follow-up

Pipeline: EDF preprocessing → 126 mu/beta log-band-power features → training-only standardization and fixed ±20 clipping → PCA (full SVD, no whitening) → automatic-shrinkage LDA. The no-PCA baseline uses identical preprocessing and LDA. PCA centers its input and retains directions with the most variance, not necessarily the most left/right information.

PCA and scaling are fitted separately inside each development training fold. Fixed candidate counts are 8, 16, 32 and 64. The best count is selected using S001–010 leave-one-person-out balanced accuracy; ties favor fewer components. Selected development performance is tuning performance, not an unbiased generalization estimate.

| Representation | Development mean balanced accuracy |
|---|---:|
| no_pca | 56.63% |
| pca_8 | 52.65% |
| pca_16 | 53.63% |
| pca_32 | 55.58% |
| pca_64 | 52.99% |

Selected PCA setting: **32 components**, retaining **97.9%** of variance when fitted on all development data.

| Model | Follow-up S011–030 balanced accuracy |
|---|---:|
| No PCA + LDA | 53.75% |
| PCA (32) + LDA | 55.69% |

Mean paired change: **+1.95 percentage points**; 95% paired subject-bootstrap interval **[-1.01, +5.15]**. Better for 11, tied for 0, worse for 9 people.

This is exploratory: S011–030 results were already seen before requesting PCA. Only the development-selected PCA setting is evaluated here; test results do not select the component count. No new untouched subjects, multiplicity-adjusted significance, or production-model replacement is claimed. 50% is chance balanced accuracy; a small improvement is not reliable neural decoding.

The PCA projection uses the algebraically equivalent explicit NumPy einsum product to avoid the previously observed matrix-product warning. Its equivalence to sklearn PCA.transform is tested.

Reproduce (after eeg-finalize creates prepared folders):

```bash
python -m eeg_lr.compare_pca
```

Reference: https://scikit-learn.org/1.5/modules/generated/sklearn.decomposition.PCA.html
