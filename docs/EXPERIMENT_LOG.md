# Experiment log

This is a retrospective account, not a preregistration.

1. Initial development used S001–S010, imagery runs 4/8/12, common-average reference, a 7–35 Hz filter, fixed 0.5–4.0 s epochs and 500 µV rejection.
2. Development explored CSP implementation/numerical handling and compared CSP plus logistic regression against mu/beta band power plus logistic regression. These choices were informed by the development subjects. LDA and epoch-random comparisons discussed earlier were not retained as completed experiments.
3. Band power was selected using development subject-grouped performance. Its settings and the S011–S030 reserved split were fixed before inspecting reserved scores. Both pre-existing EEG models were scored for transparency; the final model was not switched after the reserved comparison.
4. On September 11, 2026, the final script retained 444 training and 821 reserved trials. It exported the band-power artifact and saved all per-person metrics and event-level rejection audits. S031–S109 were unused.
5. A development permutation diagnostic used 99 within-subject/run label shuffles. It is exploratory after model selection, not a confirmatory test of neural decoding.
6. A run/cue-order-only comparator was included in the final analysis. Its 52.5% reserved balanced accuracy shows that a small above-chance score need not imply an EEG-specific mechanism. It does not prove the absence of motor-related information.
7. The selected model's reserved score is 52.1%, with a 95% bootstrap interval over 20 people of approximately 50.0–54.4%. No test-driven retraining or hyperparameter optimization followed this result.

## Subsequent user-requested experiments

After the original test, shrinkage LDA and PCA–LDA were compared on the same development data. Their S011–S030 follow-up results are explicitly exploratory because those people had already been inspected. Band-power LDA scored 53.75%; development-selected 32-component PCA–LDA scored 55.69% on that reused group.

Next, development expanded to S001–S080. Five-fold person-separated validation selected 32-component PCA–LDA from six fixed candidates before S081–S109 were read. The final test included 25 usable people and yielded 59.08%; the same configuration fitted on S001–S010 yielded 55.35% on the same test people. Three unsupported-rate participants and one with all trials rejected did not contribute to that score. Full reports, folds, selection and rejection audits accompany the code.

These follow-ups do not replace the original exported prediction artifact. The optional `--show-targets` prediction flag adds recorded targets and correctness only after EEG-only prediction.

Next work should reserve new data before further tuning; inspect failure recordings and possible ocular/muscle interference; evaluate calibration and task-order controls; and test subject-specific calibration separately from unseen-person generalization.
