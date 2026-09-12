"""Post-submission-protocol comparison: change only LR versus shrinkage LDA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .evaluate import (bandpower_logreg, csp_logreg, load_prepared,
                       score_predictions, split_groups, write_results)


class StableBinaryLDA(LinearDiscriminantAnalysis):
    """Unchanged sklearn LDA fitting; explicit binary linear decision product."""

    def decision_function(self, x):
        check_is_fitted(self, ('coef_', 'intercept_'))
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.coef_.shape[1] or len(self.classes_) != 2:
            raise ValueError('Expected a matching feature matrix and binary LDA fit')
        return np.einsum('ij,j->i', x, self.coef_[0]) + self.intercept_[0]


def lda():
    """Fixed Ledoit-Wolf automatic shrinkage; priors learned on training labels."""
    return StableBinaryLDA(solver='lsqr', shrinkage='auto')


def predictions(x_train, y_train, x_test, factory):
    """Both classifiers see exactly the same training-fitted representation."""
    original = factory()
    transform = Pipeline(original.steps[:-1])
    train = transform.fit_transform(x_train, y_train)
    test = transform.transform(x_test)
    output = {}
    for name, classifier in [('logreg', clone(original.steps[-1][1])), ('lda', lda())]:
        classifier.fit(train, y_train)
        scores = classifier.decision_function(test)
        if not np.isfinite(scores).all() or not np.isfinite(classifier.coef_).all():
            raise ValueError(f'Non-finite {name} fit or predictions')
        output[name] = classifier.predict(test)
    return output


def compare_development(x, y, subjects, runs, split):
    groups = split_groups(split, subjects, runs)
    results = []
    for tr, te in LeaveOneGroupOut().split(x, y, groups):
        value = int(groups[te][0])
        name = f'S{value:03d}' if split == 'subject-grouped' else f'R{value:02d}'
        print(f'{split}: {name}', flush=True)
        for feature, factory in [('bandpower', bandpower_logreg), ('csp', csp_logreg)]:
            for classifier, predicted in predictions(x[tr], y[tr], x[te], factory).items():
                results.append(score_predictions(model=f'{feature}_{classifier}', split=split,
                    held_out=name, y_train=y[tr], y_test=y[te], predictions=predicted))
    return results


def paired_summary(results):
    summary = {}
    for feature in ('bandpower', 'csp'):
        lr = {r.held_out: r.balanced_accuracy for r in results if r.model == feature+'_logreg'}
        ld = {r.held_out: r.balanced_accuracy for r in results if r.model == feature+'_lda'}
        differences = np.array([ld[k]-lr[k] for k in sorted(lr)])
        rng = np.random.default_rng(7)
        boot = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
        summary[feature] = {'mean_difference_lda_minus_lr': float(differences.mean()),
            'paired_bootstrap_95_ci': np.quantile(boot, [.025,.975]).tolist(),
            'lda_wins': int((differences > 0).sum()), 'ties': int((differences == 0).sum()),
            'folds': len(differences)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, default=Path('data/final-development'))
    parser.add_argument('--test', type=Path, default=Path('data/final-heldout'))
    parser.add_argument('--output', type=Path, default=Path('reports/lda-comparison'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('error', category=RuntimeWarning)
    # Limit BLAS concurrency for deterministic resource use; do not suppress warnings.
    with threadpool_limits(limits=1):
        x, y, subjects, runs = load_prepared(args.development)
        xt, yt, st, _ = load_prepared(args.test)
        if set(subjects) != set(range(1,11)) or set(st) != set(range(11,31)):
            raise ValueError('This comparison requires development S001–010 and test S011–030')
        all_results = {}
        for split in ('subject-grouped', 'run-grouped'):
            all_results[split] = compare_development(x, y, subjects, runs, split)
            write_results(all_results[split], args.output/split)
        heldout = []
        print('Follow-up on previously inspected S011–030', flush=True)
        for feature, factory in [('bandpower', bandpower_logreg), ('csp', csp_logreg)]:
            for classifier, predicted in predictions(x, y, xt, factory).items():
                for s in np.unique(st):
                    mask = st == s
                    heldout.append(score_predictions(model=f'{feature}_{classifier}',
                        split='followup-heldout', held_out=f'S{s:03d}', y_train=y,
                        y_test=yt[mask], predictions=predicted[mask]))
        all_results['followup-heldout'] = heldout
        write_results(heldout, args.output/'followup-heldout')
    paired = {split: paired_summary(results) for split, results in all_results.items()}
    (args.output/'paired_differences.json').write_text(json.dumps(paired, indent=2)+'\n')
    lines = ['# Logistic regression versus shrinkage LDA', '',
        'Follow-up experiment requested after the original reserved-test results were known. '
        'No changes to the exported model or original experiment. This is not a fresh confirmatory test.', '',
        'Only the classifier changes. Identical train-fitted CSP/band-power features, scaling, clipping, '
        'epochs and folds are shared. LDA uses scikit-learn LSQR with automatic Ledoit-Wolf shrinkage '
        'and training-estimated class priors. There is no hyperparameter search.', '',
        'The sklearn fit is unchanged. Binary decision scores use explicit NumPy einsum '
        'for both classifiers; standard matrix multiplication raised a runtime warning '
        'at prediction on this Mac. Equivalence is unit tested. Warnings are not suppressed '
        'and non-finite coefficients/scores fail the experiment. The underlying runtime cause '
        'is not established.', '',
        '| Features + classifier | Development: unseen person | Development: unseen run | Follow-up: S011–030 |',
        '|---|---:|---:|---:|']
    for model in ('bandpower_logreg','bandpower_lda','csp_logreg','csp_lda'):
        cells=[]
        for results in all_results.values():
            scores=np.array([r.balanced_accuracy for r in results if r.model==model])*100
            cells.append(f'{scores.mean():.1f}% ± {scores.std(ddof=1):.1f}%')
        lines.append('| '+model+' | '+' | '.join(cells)+' |')
    lines += ['', 'Balanced accuracy averages left and right recall; 50% is the majority baseline. '
              '± denotes sample SD across folds/people, not confidence intervals.', '',
              '## Paired differences on the previously inspected test people', '']
    for feature, item in paired['followup-heldout'].items():
        lo,hi=item['paired_bootstrap_95_ci']
        lines.append(f"- {feature}: LDA minus LR = {item['mean_difference_lda_minus_lr']*100:+.2f} percentage points; "
            f"95% paired subject-bootstrap interval [{lo*100:+.2f}, {hi*100:+.2f}]; "
            f"LDA better for {item['lda_wins']}/{item['folds']} people, {item['ties']} ties.")
    lines += ['', 'Bootstrap: resample people with replacement, 10,000 replicates, seed 7. '
              'An interval crossing zero does not establish a consistent improvement. '
              'Development folds overlap in training data; their bootstrap intervals are descriptive only. '
              'No multiplicity adjustment or new-subject confirmation is claimed.', '',
              'Reproduce from the repository root:', '', '```bash',
              'python -m eeg_lr.compare_lda', '```', '',
              'Requires the prepared folders created by `eeg-finalize`. All per-fold metrics '
              'and confusion counts are stored alongside this report.', '',
              'LDA reference: https://scikit-learn.org/1.5/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
