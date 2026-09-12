"""Run the frozen submission experiment. Training S001-010; test S011-030."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .dataset import CHANNELS
from .evaluate import (bandpower_logreg, csp_logreg, evaluate, load_prepared,
                       score_predictions, write_results, StableBinaryLogisticRegression)
from .predict import export_model, predict_arrays, PARAMETERS
from .prepare import build_dataset


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def macro_score(y, pred, subjects):
    return float(np.mean([balanced_accuracy_score(y[subjects == s], pred[subjects == s])
                          for s in np.unique(subjects)]))


def permutation_test(x, y, subjects, runs, repetitions=99):
    # Welch has no learned parameters; cache only this deterministic transform.
    base = bandpower_logreg()
    features = base.named_steps['bandpower'].fit_transform(x)
    classifier = Pipeline(base.steps[1:])
    folds = list(LeaveOneGroupOut().split(features, y, subjects))

    def score(labels):
        predicted = np.empty_like(labels)
        for tr, te in folds:
            estimator = clone(classifier).fit(features[tr], labels[tr])
            predicted[te] = estimator.predict(features[te])
        return macro_score(labels, predicted, subjects)

    observed = score(y)
    rng = np.random.default_rng(20260911)
    null = []
    for repetition in range(repetitions):
        shuffled = y.copy()
        # Preserve each subject/run's class counts; labels are exchangeable
        # only under the stated within-run trial-exchangeability assumption.
        for s in np.unique(subjects):
            for r in np.unique(runs):
                idx = np.flatnonzero((subjects == s) & (runs == r))
                shuffled[idx] = rng.permutation(y[idx])
        null.append(score(shuffled))
        if (repetition+1) % 20 == 0:
            print(f'Permutation {repetition+1}/{repetitions}', flush=True)
    return dict(observed=observed, null_scores=null,
                p_value=(1+sum(v >= observed for v in null))/(repetitions+1),
                permutations=repetitions, seed=20260911,
                caveat='Exploratory: model selected on these development subjects; within-run exchangeability assumed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--permutations', type=int, default=99)
    parser.add_argument('--out', type=Path, default=Path('reports'))
    parser.add_argument('--model', type=Path, default=Path('models/bandpower.npz'))
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error('Use at least one permutation')
    print('Frozen model: bandpower + logistic regression; no test-based tuning.', flush=True)
    folders = []
    for name, subjects in [('development', list(range(1,11))), ('heldout', list(range(11,31)))]:
        output = Path('data') / ('final-' + name)
        build_dataset(argparse.Namespace(data_dir=args.data_dir, output_dir=output,
            subjects=subjects, runs=[4,8,12], **PARAMETERS))
        folders.append(output)
    x, y, s, r = load_prepared(folders[0])
    xt, yt, st, rt = load_prepared(folders[1])
    for split in ('subject-grouped', 'run-grouped'):
        print(f'Development {split}', flush=True)
        write_results(evaluate(x,y,s,r,split=split), args.out / ('development-'+split))
    models = {'bandpower_logreg': bandpower_logreg(), 'csp_logreg': csp_logreg()}
    results = []
    predictions = {}
    training_metrics = {}
    for name, model in models.items():
        model.fit(x,y)
        pred = model.predict(xt)
        predictions[name] = pred
        training_metrics[name] = macro_score(y,model.predict(x),s)
        for subject in np.unique(st):
            mask = st == subject
            results.append(score_predictions(model=name, split='heldout-subjects',
                held_out=f'S{subject:03d}', y_train=y, y_test=yt[mask], predictions=pred[mask]))
    majority = int(np.bincount(y).argmax())
    for subject in np.unique(st):
        mask = st == subject
        results.append(score_predictions(model='majority_baseline', split='heldout-subjects',
            held_out=f'S{subject:03d}', y_train=y, y_test=yt[mask],
            predictions=np.full(mask.sum(), majority)))
    write_results(results,args.out/'heldout')
    export_model(models['bandpower_logreg'],args.model,list(CHANNELS))
    # Exact export parity, checked against the full development array.
    reconstructed = np.concatenate([np.load(p)['X'] for p in sorted(folders[0].glob('S???.npz'))])
    exported, _ = predict_arrays(reconstructed,args.model)
    assert np.array_equal(exported,models['bandpower_logreg'].predict(x)), 'Export parity failed'
    write_json(args.out/'training_metrics.json',training_metrics)
    write_json(args.out/'permutation.json',permutation_test(x,y,s,r,args.permutations))

    # Nuisance baseline: predict from run and cue ordinal, without EEG.
    def order_features(folder):
        output=[]
        for path in sorted(folder.glob('S???.npz')):
            with np.load(path) as archive:
                output.extend(zip(archive['runs'].astype(int).tolist(),
                    np.rint((archive['onsets']-4.2)/8.3).astype(int).tolist()))
        return np.array([f'{run}:{ordinal}' for run,ordinal in output]).reshape(-1,1)
    nuisance = Pipeline([('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
                         ('clf',StableBinaryLogisticRegression(solver='liblinear',random_state=7))])
    nuisance.fit(order_features(folders[0]),y)
    nuisance_pred = nuisance.predict(order_features(folders[1]))
    nuisance_scores = [asdict(score_predictions(model='run_cue_order_only',split='heldout-subjects',
        held_out=f'S{subject:03d}', y_train=y, y_test=yt[st==subject], predictions=nuisance_pred[st==subject]))
        for subject in np.unique(st)]
    write_json(args.out/'timing_baseline.json',nuisance_scores)
    selected = [a.balanced_accuracy for a in results if a.model=='bandpower_logreg']
    rng=np.random.default_rng(7)
    means=rng.choice(selected,size=(10000,len(selected)),replace=True).mean(axis=1)
    metadata = dict(training_subjects=list(range(1,11)),test_subjects=list(range(11,31)),
        unused_subjects=list(range(31,110)),runs=[4,8,12],preprocessing=PARAMETERS,
        train_epochs=len(y),test_epochs=len(yt),model='bandpower_logreg',
        test_mean_balanced_accuracy=float(np.mean(selected)),
        subject_bootstrap_95_ci=np.quantile(means,[.025,.975]).tolist(),
        timing_baseline_mean=float(np.mean([a['balanced_accuracy'] for a in nuisance_scores])),
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        versions={name:importlib.metadata.version(name) for name in ['mne','numpy','scipy','scikit-learn']},
        python=platform.python_version(),
        scope='Offline cue-locked imagined left/right fists. Fixed artifact exclusion. No deployment calibration.')
    write_json(args.model.with_suffix('.json'),metadata)
    write_json(args.out/'experiment.json',metadata)
    # Retain a portable QC audit without local absolute paths.
    audits={}
    for folder in folders:
        for p in sorted(folder.glob('*-events.json')):
            audits[p.name]=json.loads(p.read_text())
    write_json(args.out/'event_audit.json',audits)
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    main()
