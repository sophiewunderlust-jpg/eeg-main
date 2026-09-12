"""Predict cue-locked motor imagery from EDF using portable numeric weights."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.special import expit

from .features import BandPowerFeatures
from .prepare import prepare_recording

PARAMETERS = dict(l_freq=7.0, h_freq=35.0, tmin=0.5, tmax=4.0, reject_uv=500.0)


def export_model(pipeline, path: Path, channels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scaler = pipeline.named_steps['standardize']
    classifier = pipeline.named_steps['logistic_regression']
    np.savez_compressed(path, mean=scaler.mean_, scale=scaler.scale_,
                        coef=classifier.coef_[0], intercept=classifier.intercept_,
                        channels=np.asarray(channels), format_version=np.array(1))


def predict_arrays(x_volts: np.ndarray, model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The only prediction inputs are EEG samples and learned numeric weights."""
    with np.load(model_path, allow_pickle=False) as model:
        if x_volts.ndim != 3 or x_volts.shape[1:] != (64, 561):
            raise ValueError('Expected EEG epochs with shape (trials, 64, 561)')
        if not np.isfinite(x_volts).all():
            raise ValueError('EEG contains non-finite values')
        if len(x_volts) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        features = BandPowerFeatures().fit_transform(x_volts[:, :-1, :] * 1e6)
        standardized = np.clip((features-model['mean']) / model['scale'], -20, 20)
        scores = np.einsum('ij,j->i', standardized, model['coef']) + model['intercept'][0]
        return (scores > 0).astype(int), expit(scores)


def predict_edf(edf: Path, model_path: Path, run: int | None = None) -> dict:
    if run is None:
        match = re.search(r'R(\d{2})\.edf$', edf.name, re.IGNORECASE)
        if not match:
            raise ValueError('Cannot infer run from filename; supply --run 4, 8, or 12')
        run = int(match.group(1))
    if run not in (4, 8, 12):
        raise ValueError('This trained artifact supports imagined left/right runs 4, 8, 12 only')
    x, _unused_ground_truth, metadata = prepare_recording(edf, run, **PARAMETERS)
    with np.load(model_path, allow_pickle=False) as model:
        if list(model['channels']) != metadata['ch_names']:
            raise ValueError('Model channel order does not match recording')
    labels, probabilities = predict_arrays(x, model_path)
    accepted = iter(zip(labels.tolist(), probabilities.tolist()))
    rows = []
    for cue in metadata['events']:
        row = {k: cue[k] for k in ('trial_index', 'onset_s', 'duration_s', 'accepted', 'reasons')}
        if cue['accepted']:
            label, probability = next(accepted)
            row.update(predicted_label='left_fist' if label == 0 else 'right_fist',
                       probability_right=round(probability, 6))
        else:
            row.update(predicted_label=None, probability_right=None)
        rows.append(row)
    return dict(recording=edf.name, run=run, model=model_path.name,
                accepted_trials=len(labels), rejected_trials=len(rows)-len(labels),
                note='Offline cue-locked predictions; probabilities are not calibrated confidence.',
                predictions=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('edf', type=Path)
    parser.add_argument('--model', type=Path, default=Path('models/bandpower.npz'))
    parser.add_argument('--run', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = predict_edf(args.edf.expanduser(), args.model, args.run)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
