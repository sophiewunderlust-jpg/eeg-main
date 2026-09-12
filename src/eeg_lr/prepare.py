"""CLI for inspecting EDF files and building subject-separated epoch archives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import mne
import numpy as np

from .dataset import (
    EXPECTED_SFREQ,
    CHANNELS,
    IMAGINED_LEFT_RIGHT_RUNS,
    N_EXPECTED_EEG_CHANNELS,
    condition_for_run,
    discover_subjects,
    recording_paths,
    semantic_label,
    validate_left_right_runs,
)


def normalize_channel_name(name: str) -> str:
    return name.rstrip(".").upper()


def load_raw(edf_path: Path, *, preload: bool) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose="error")
    raw.rename_channels({name: normalize_channel_name(name) for name in raw.ch_names})
    if set(raw.ch_names) != set(CHANNELS):
        raise ValueError(f"Expected the EEGMMIDB channel set; missing={sorted(set(CHANNELS)-set(raw.ch_names))}, extra={sorted(set(raw.ch_names)-set(CHANNELS))}")
    raw.reorder_channels(list(CHANNELS))
    raw.set_channel_types({name: "eeg" for name in raw.ch_names}, verbose="error")
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, match_case=False, on_missing="warn", verbose="error")
    return raw


def inspect_edf(path: Path, plot: bool) -> None:
    raw = load_raw(path, preload=False)
    print(f"File: {path}")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']:.1f} Hz")
    print(f"Duration: {raw.times[-1]:.2f} seconds")
    print("Channel names: " + ", ".join(raw.ch_names))
    print("Annotations:")
    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
        strict=True,
    ):
        print(f"  {onset:8.3f}s  {duration:6.3f}s  {description}")
    if plot:
        raw.plot(duration=10, n_channels=20, scalings="auto", block=True)


def prepare_recording(
    edf_path: Path,
    run: int,
    *,
    l_freq: float,
    h_freq: float,
    tmin: float,
    tmax: float,
    reject_uv: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    validate_left_right_runs((run,))
    if not 0 < l_freq < h_freq < EXPECTED_SFREQ / 2:
        raise ValueError("Require 0 < l_freq < h_freq < 80 Hz")
    if not 0 <= tmin < tmax or reject_uv <= 0:
        raise ValueError("Require 0 <= tmin < tmax and positive rejection threshold")
    raw = load_raw(edf_path, preload=True)
    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, EXPECTED_SFREQ):
        raise ValueError(f"Expected {EXPECTED_SFREQ} Hz, got {sfreq} Hz in {edf_path}")
    if len(raw.ch_names) != N_EXPECTED_EEG_CHANNELS:
        raise ValueError(
            f"Expected {N_EXPECTED_EEG_CHANNELS} EEG channels, got "
            f"{len(raw.ch_names)} in {edf_path}"
        )

    raw.set_eeg_reference("average", projection=False, verbose="error")
    raw.filter(l_freq, h_freq, method="fir", verbose="error")
    events, annotation_ids = mne.events_from_annotations(
        raw, event_id={"T1": 1, "T2": 2}, verbose="error"
    )
    if not len(events):
        raise ValueError(f"No T1/T2 cues in {edf_path}")
    cues = [a for a in raw.annotations if a["description"] in ("T1", "T2")]
    if len(cues) != len(events):
        raise ValueError("Annotation/event count mismatch")

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"T1": 1, "T2": 2},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject={"eeg": reject_uv * 1e-6},
        reject_by_annotation=True,
        on_missing="ignore",
        verbose="error",
    )
    short = [i for i, original in enumerate(epochs.selection) if float(cues[original]["duration"]) < tmax + 1 / sfreq]
    if short:
        epochs.drop(short, reason="CUE_TOO_SHORT", verbose="error")
    inverse_event_id = {value: key for key, value in epochs.event_id.items()}
    y = np.asarray(
        [semantic_label(run, inverse_event_id[event_code]) for event_code in epochs.events[:, 2]],
        dtype=np.int8,
    )
    metadata = {
        "sfreq": sfreq,
        "ch_names": epochs.ch_names,
        "n_events_before_rejection": int(len(events)),
        "n_epochs_kept": int(len(epochs)),
        "n_epochs_rejected": int(len(events) - len(epochs)),
        "events": [
            {"trial_index": i, "onset_s": float(cue["onset"]),
             "duration_s": float(cue["duration"]),
             "label": semantic_label(run, cue["description"]),
             "accepted": i in set(epochs.selection.tolist()),
             "reasons": list(epochs.drop_log[i])}
            for i, cue in enumerate(cues)
        ],
        "kept_onsets": [float(cues[i]["onset"]) for i in epochs.selection],
    }
    data = epochs.get_data(copy=True) if len(epochs) else np.empty((0, len(CHANNELS), round((tmax-tmin)*sfreq)+1))
    return data, y, metadata


def build_dataset(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    runs = tuple(args.runs)
    validate_left_right_runs(runs)
    subjects = args.subjects or discover_subjects(data_dir)
    if not subjects:
        raise ValueError(f"No S### subject directories found under {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    total_epochs = 0
    for subject in subjects:
        subject_x: list[np.ndarray] = []
        subject_y: list[np.ndarray] = []
        subject_runs: list[np.ndarray] = []
        subject_onsets: list[np.ndarray] = []
        shared_metadata: dict[str, Any] | None = None
        for run in runs:
            recording = recording_paths(data_dir, subject, run)
            if not recording.edf_path.is_file():
                raise FileNotFoundError(recording.edf_path)
            if not recording.event_path.is_file():
                raise FileNotFoundError(recording.event_path)
            x, y, metadata = prepare_recording(
                recording.edf_path,
                run,
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                tmin=args.tmin,
                tmax=args.tmax,
                reject_uv=args.reject_uv,
            )
            shared_metadata = metadata
            subject_onsets.append(np.asarray(metadata["kept_onsets"]))
            (output_dir / f"S{subject:03d}R{run:02d}-events.json").write_text(json.dumps(metadata["events"], indent=2) + "\n")
            subject_x.append(x)
            subject_y.append(y)
            subject_runs.append(np.full(len(y), run, dtype=np.int8))
            manifest_rows.append(
                {
                    "subject": recording.subject_id,
                    "run": recording.run_id,
                    "condition": condition_for_run(run),
                    "edf_path": str(recording.edf_path),
                    "event_path": str(recording.event_path),
                    **{key: value for key, value in metadata.items() if key not in ("ch_names", "events", "kept_onsets")},
                }
            )

        assert shared_metadata is not None
        x_all = np.concatenate(subject_x)
        y_all = np.concatenate(subject_y)
        runs_all = np.concatenate(subject_runs)
        np.savez_compressed(
            output_dir / f"S{subject:03d}.npz",
            X=x_all,
            y=y_all,
            runs=runs_all,
            onsets=np.concatenate(subject_onsets),
            sfreq=np.float64(shared_metadata["sfreq"]),
            ch_names=np.asarray(shared_metadata["ch_names"]),
            tmin=np.float64(args.tmin),
            tmax=np.float64(args.tmax),
        )
        total_epochs += len(y_all)
        print(f"S{subject:03d}: saved {len(y_all)} epochs")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "data_dir": str(data_dir),
        "subjects": subjects,
        "runs": list(runs),
        "condition": sorted({condition_for_run(run) for run in runs}),
        "label_mapping": {"left_fist": 0, "right_fist": 1},
        "filter_hz": [args.l_freq, args.h_freq],
        "epoch_seconds": [args.tmin, args.tmax],
        "reject_peak_to_peak_uv": args.reject_uv,
        "total_epochs": total_epochs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {manifest_path}")
    print(f"Wrote {output_dir / 'summary.json'}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="eeg-prepare")
    commands = root.add_subparsers(dest="command", required=True)

    inspect_cmd = commands.add_parser("inspect", help="Inspect and optionally plot one EDF")
    inspect_cmd.add_argument("edf", type=Path)
    inspect_cmd.add_argument("--plot", action="store_true")

    build_cmd = commands.add_parser("build", help="Create subject-separated epoch archives")
    build_cmd.add_argument("--data-dir", type=Path, required=True)
    build_cmd.add_argument("--output-dir", type=Path, required=True)
    build_cmd.add_argument("--subjects", type=int, nargs="+")
    build_cmd.add_argument("--runs", type=int, nargs="+", default=list(IMAGINED_LEFT_RIGHT_RUNS))
    build_cmd.add_argument("--l-freq", type=float, default=7.0)
    build_cmd.add_argument("--h-freq", type=float, default=35.0)
    build_cmd.add_argument("--tmin", type=float, default=0.5)
    build_cmd.add_argument("--tmax", type=float, default=4.0)
    build_cmd.add_argument("--reject-uv", type=float, default=500.0)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "inspect":
        inspect_edf(args.edf.expanduser().resolve(), args.plot)
    else:
        build_dataset(args)


if __name__ == "__main__":
    main()
