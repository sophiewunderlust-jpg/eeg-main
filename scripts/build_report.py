"""Generate the submission report and scientific figure."""
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'


def read(name):
    return json.loads((REPORTS / name).read_text())


def main():
    summary = read('heldout/summary.json')
    rows = list(csv.DictReader((REPORTS / 'heldout/fold_results.csv').open()))
    selected = [r for r in rows if r['model'] == 'bandpower_logreg']
    scores = [float(r['balanced_accuracy']) * 100 for r in selected]
    counts = {k: sum(int(r[k]) for r in selected)
              for k in ('true_left', 'left_as_right', 'right_as_left', 'true_right')}
    audit = read('event_audit.json')
    events = [e for run in audit.values() for e in run]
    rejected = [e for e in events if not e['accepted']]
    reasons = Counter(reason for e in rejected for reason in e['reasons'])
    (REPORTS / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(11, 4.4), layout='constrained')
    ax.bar([r['held_out'] for r in selected], scores, color='#167e89')
    ax.axhline(50, color='#a94532', linestyle='--', label='Majority baseline: 50%')
    ax.set(ylim=(0, 100), ylabel='Balanced accuracy (%)', title='Frozen band-power model: 20 unseen people')
    ax.tick_params(axis='x', rotation=55)
    ax.legend(loc='upper right')
    fig.savefig(REPORTS / 'figures/heldout_subjects.png', dpi=160)
    plt.close(fig)
    matrix = np.array([[counts['true_left'], counts['left_as_right']],
                       [counts['right_as_left'], counts['true_right']]])
    table = '\n'.join(f"| {r['held_out']} | {r['n_test']} | {float(r['balanced_accuracy'])*100:.1f}% | {float(r['left_recall'])*100:.1f}% | {float(r['right_recall'])*100:.1f}% |" for r in selected)
    report = f'''# Final experiment report

## Main finding

The model selected on S001–S010 achieved **52.1% mean per-person balanced accuracy** on S011–S030. The 95% percentile bootstrap interval, resampling 20 people 10,000 times with seed 7, is **50.0–54.4%**. This interval measures sampling uncertainty over these people, not every source of uncertainty or proof of a neural effect.

The training resubstitution score is 76.1%, versus 56.0% development leave-one-person-out and 52.1% reserved performance. The gap is consistent with poor generalization; training accuracy is not a deployment estimate. These are averages of per-person balanced accuracy, not pooled accuracy.

![Per-person balanced accuracy](figures/heldout_subjects.png)

## Frozen comparison

| Model | Reserved balanced accuracy (mean ± sample SD) |
|---|---:|
| Band power + logistic regression, selected | 52.1% ± 5.2% |
| CSP + logistic regression | 53.1% ± 6.0% |
| Majority class | 50.0% |
| Run + approximate cue ordinal, no EEG | 52.5% |

The selected model is not changed after seeing the reserved comparison. The timing-only control was trained on the same development people and applied to reserved people. It uses one-hot run/approximate trial-position categories, not EEG. Task ordering can carry information, so a small score above chance alone is insufficient evidence of neural decoding.

A 99-permutation development diagnostic gave p=0.02. Labels were shuffled within subject/run, preserving class counts. Model selection used these same development people; within-run exchangeability is an assumption potentially challenged by temporal structure. Treat this diagnostic as exploratory, not confirmatory.

## Trial accounting

There were {len(events)} candidate left/right cues across 30 people and three runs each: {len(events)-len(rejected)} accepted (444 development + 821 reserved) and {len(rejected)} rejected. A trial can have multiple channel reasons. Most frequent rejection-channel tags: {', '.join(f'{k}: {v}' for k,v in reasons.most_common(8))}.

Channel-name reasons indicate an excessive post-filter peak-to-peak amplitude on that channel; they do not diagnose a specific artifact. All candidates and reasons are in `event_audit.json`. Rejection is fixed and label-independent, but changing class/subject retention can still affect evaluation. Raw recordings were not manually certified artifact-free.

## Pooled confusion counts (821 reserved trials)

| Actual / predicted | Left | Right |
|---|---:|---:|
| Left | {matrix[0,0]} | {matrix[0,1]} |
| Right | {matrix[1,0]} | {matrix[1,1]} |

These are pooled counts; the headline metric weights people equally instead of pooling their trials. Correct predictions are on the diagonal.

## Individual reserved subjects

| Subject | Accepted trials | Balanced accuracy | Left recall | Right recall |
|---|---:|---:|---:|---:|
{table}

## Limitations and next work

- Only 30 of 109 people were used; the remaining 79 were not touched in this experiment. The first 10 are a small, non-random development cohort.
- No reliable generalization is established: the confidence interval is near chance and the timing control is comparable. More sophisticated models would not automatically solve this.
- Fixed filtering/rejection does not remove every eye/muscle artifact. Visual review and artifact sensitivity analyses should precede stronger claims.
- Cue annotations locate task intervals. This is not rest detection, asynchronous BCI, or real-time inference; zero-phase filtering is non-causal.
- Next, reserve new people before tuning; compare artifact handling and subject calibration; challenge cue-order shortcuts; add a causal-window evaluation if online use is required.

## Reproducibility

Run the README's `eeg-finalize` command, then `python scripts/build_report.py`. The model metadata records parameters, subject split, library versions and SHA-256. Numeric weights use a non-pickle NPZ archive. Software tests include exported-model prediction parity and invariance to swapping T1/T2 cue values at inference.

See `../docs/AI_USE.md` for the substantial AI-assistance disclosure. Source data: PhysioNet EEGMMIDB v1.0.0, https://physionet.org/content/eegmmidb/1.0.0/.
'''
    (REPORTS / 'REPORT.md').write_text(report)
    print(f'Wrote {REPORTS / "REPORT.md"}')


if __name__ == '__main__':
    main()
