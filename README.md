# Left or right? EEG decoding under subject shift

An end-to-end, offline motor-imagery classifier for the Neurotech@Berkeley FA26 software recruitment project. It opens EDF recordings, audits trials, compares two models, evaluates unseen people, and predicts from one raw EDF.

**Main result:** the preselected band-power model achieved **52.1% mean per-person balanced accuracy** on 20 held-out subjects (95% subject-bootstrap interval **50.0–54.4%**). A timing-only comparator scored **52.5%**. This is a reproducible baseline and failure analysis, **not evidence of reliable neural decoding**.

## Quick start: predict without training

Use Python **3.12** (3.11 is also allowed). From a terminal:

```bash
git clone https://github.com/sophiewunderlust-jpg/eeg-main.git
cd eeg-main
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m eeg_lr.download --data-dir data/raw --subjects 11
eeg-predict data/raw/S011/S011R04.edf --model models/bandpower.npz --output prediction.json
```

On Windows, activate with `.venv\Scripts\activate` instead. The download is optional if you already have the data: replace the EDF path with your own. Run commands from the repository root.

The committed model is already trained. Output contains a left/right label and an **uncalibrated** right-class probability for each accepted cue; rejected cues have a reason and no prediction. This command supports this dataset's **imagery runs R04, R08, R12**, 64 expected channels and 160 Hz recordings. A renamed file needs `--run 4` (or 8/12). Other run types are deliberately rejected.

Cue timestamps and durations locate trials, but the cue's **left/right value is not a classifier input**. This is offline, cue-locked classification—not continuous rest detection or real-time control. EDF's embedded annotations are read directly; a companion event file is not needed for prediction.

## Open an EDF

EDF is a binary signal format, not a text or Excel file. MNE reads it in Python:

```bash
eeg-prepare inspect data/raw/S011/S011R04.edf
eeg-prepare inspect data/raw/S011/S011R04.edf --plot
```

The first prints metadata/events; the second opens a scrollable waveform viewer in a graphical desktop session. Without a desktop, use the metadata command.

## Pipeline

| Stage | Operation | Code / output |
|---|---|---|
| 1. Read and validate | Discover EDFs, verify channel identities and sample rate, decode run-specific cues | `dataset.py`, `prepare.py` |
| 2. Preprocess | Common-average reference; 7–35 Hz FIR; cue-locked epochs; rejection audit | Per-subject arrays and event logs |
| 3. Extract features | Compare mu/beta log band power with eight CSP log-power features | `features.py`, `csp.py` |
| 4. Develop and compare | Leave one person out; separately leave one global run number out | `evaluate.py`, development reports |
| 5. Frozen test | Fit on S001–S010, test once on S011–S030; majority/timing controls | `finalize.py`, held-out scores |
| 6. Predict | Export numeric weights and classify raw EDF cues | `predict.py`, `models/` |
| 7. Communicate | Reproducible report, limitations, public source | `reports/REPORT.md` |

### Event meanings

| Runs | T0 | T1 | T2 | Used in final model? |
|---|---|---|---|---|
| 3, 7, 11 | Rest | Execute left fist | Execute right fist | No |
| 4, 8, 12 | Rest | Imagine left fist | Imagine right fist | **Yes** |
| 5, 9, 13 | Rest | Execute both fists | Execute both feet | No |
| 6, 10, 14 | Rest | Imagine both fists | Imagine both feet | No |

R01/R02 are eyes-open/closed baselines. T1/T2 are not globally synonymous with left/right: their meaning depends on the run. Rest is excluded because this project asks a binary left/right question.

### Exact preprocessing and models

Channels are normalized and placed in a canonical 64-channel order. Each run is average-referenced and filtered continuously using MNE's zero-phase, symmetric FIR band-pass at 7–35 Hz. This uses future samples and is **not a causal online filter**.

Each cue contributes a **fixed 0.5–4.0 second epoch** (561 samples, endpoints included). This is not a sliding window. The offset avoids the earliest cue response; the endpoint stays near the intended task period. These are fixed design choices, not claimed optimal parameters. Reject an epoch if any filtered channel exceeds **500 µV peak-to-peak**, data are incomplete, or the cue duration cannot contain the epoch. No baseline subtraction, ICA, or automated bad-channel interpolation is performed. Every candidate's status is retained in `reports/event_audit.json`.

The archives retain 64 channels in volts. Both model pipelines convert to microvolts and use 63 channels, omitting referenced IZ. Although IZ is linearly reconstructible after average reference, dropping it can still change feature representations and regularization; it is a documented development choice, not a performance guarantee.

- **Band power:** Welch PSD with up to 256 samples per segment; integrate 8–13 Hz and 13–30 Hz per channel; natural log gives 126 features.
- **CSP:** estimate training-class covariances, shrink each 10% toward scaled identity, solve `C_left w = lambda (C_left + C_right) w`, retain eight eigenvectors furthest from lambda=0.5, and take `log(mean((wᵀX)²))`.
- **Classifier:** training-only standardization, fixed clipping to ±20, L2 logistic regression (`C=1`, `liblinear`, seed 7). CSP and every learned transformation are fitted inside training folds.

## Evaluation and interpretation

S001–S010 supplied **444 accepted trials** for development. The band-power model was selected using development results, then frozen. S011–S030 supplied **821 accepted trials** for the reserved test. S031–S109 were **not used**.

| Model | Development: unseen person | Development: unseen run | Reserved: unseen person |
|---|---:|---:|---:|
| Band power + logistic regression (selected) | 56.0% ± 11.0% | 60.8% ± 4.9% | **52.1% ± 5.2%** |
| CSP + logistic regression | 54.4% ± 10.6% | 54.7% ± 4.6% | 53.1% ± 6.0% |
| Majority baseline | 50.0% | 50.0% | 50.0% |

Balanced accuracy is the average of left recall and right recall. Each person has equal weight in the unseen-person means. ± is the **population standard deviation across folds/people**, not a confidence interval. Run-grouped evaluation holds out R04, R08 or R12 globally, but the same people remain in training; it is a different, easier generalization question.

The slightly higher reserved CSP result does **not** trigger a post-test model switch. A 99-shuffle development permutation diagnostic gives p=0.02, but is exploratory after model selection and assumes within-run trial exchangeability. It does not establish a neural mechanism.

A no-EEG comparator using run number and approximate cue ordinal reaches 52.5% on reserved subjects. Alongside the near-chance result and its bootstrap interval, this cautions against attributing performance to motor imagery. It does not prove neural information is absent.

See [the report](reports/REPORT.md) for per-person plots, aggregate confusion counts, uncertainty, rejection counts and next steps. Raw scores and audit JSON are committed.

## Reproduce

The download helper fetches public PhysioNet files; it does not download all 109 subjects by default.

```bash
python -m eeg_lr.download --data-dir data/raw --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
eeg-finalize --data-dir data/raw --permutations 99
python scripts/build_report.py
pytest -q
```

The full experiment takes longer than running prediction alone. It rebuilds prepared data, development folds, reserved scores, permutation control and exported model. Dependencies are pinned in `pyproject.toml`; `requirements-lock.txt` records the tested environment. To use the full lock, install it before the editable package. Numerical results may vary slightly across numerical libraries/platforms.

Raw EDFs, prepared arrays, virtual environments and personal paths are excluded from Git. The small numeric model archive is intentionally included; loading uses `allow_pickle=False`.

## Files to understand first

| File | Purpose |
|---|---|
| `src/eeg_lr/dataset.py` | Run definitions and EDF discovery |
| `src/eeg_lr/prepare.py` | EDF reading, filtering, epoching, rejection |
| `src/eeg_lr/features.py` / `csp.py` | Two feature representations |
| `src/eeg_lr/evaluate.py` | Leakage-aware development comparisons |
| `src/eeg_lr/finalize.py` | Frozen experiment and controls |
| `src/eeg_lr/predict.py` | Saved model and raw-EDF prediction |
| `tests/` | Synthetic and regression checks, including label-swap invariance |

## Attribution and disclosure

Data: [PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/), DOI [10.13026/C28G6P](https://doi.org/10.13026/C28G6P). Please follow the dataset's citation guidance, including Schalk et al. (2004), BCI2000, and Goldberger et al. (2000), PhysioNet.

EDF reading/filtering uses [MNE](https://mne.tools/1.9/generated/mne.io.read_raw_edf.html); modeling uses NumPy, SciPy and scikit-learn. Code is MIT licensed; upstream data retain their own terms.

Substantial AI assistance was used for implementation, analysis and documentation. See [AI disclosure](docs/AI_USE.md) and [experiment log](docs/EXPERIMENT_LOG.md). The applicant should review the code and explain these limitations in their own words.
