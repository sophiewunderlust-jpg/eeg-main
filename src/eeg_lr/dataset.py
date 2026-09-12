"""Dataset constants, discovery, and run-dependent event semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGINED_LEFT_RIGHT_RUNS = (4, 8, 12)
EXECUTED_LEFT_RIGHT_RUNS = (3, 7, 11)
LEFT_RIGHT_RUNS = frozenset(IMAGINED_LEFT_RIGHT_RUNS + EXECUTED_LEFT_RIGHT_RUNS)
N_EXPECTED_EEG_CHANNELS = 64
EXPECTED_SFREQ = 160.0
CHANNELS = tuple("FC5 FC3 FC1 FCZ FC2 FC4 FC6 C5 C3 C1 CZ C2 C4 C6 CP5 CP3 CP1 CPZ CP2 CP4 CP6 FP1 FPZ FP2 AF7 AF3 AFZ AF4 AF8 F7 F5 F3 F1 FZ F2 F4 F6 F8 FT7 FT8 T7 T8 T9 T10 TP7 TP8 P7 P5 P3 P1 PZ P2 P4 P6 P8 PO7 PO3 POZ PO4 PO8 O1 OZ O2 IZ".split())


@dataclass(frozen=True)
class Recording:
    subject: int
    run: int
    edf_path: Path
    event_path: Path

    @property
    def subject_id(self) -> str:
        return f"S{self.subject:03d}"

    @property
    def run_id(self) -> str:
        return f"R{self.run:02d}"


def recording_paths(data_dir: Path, subject: int, run: int) -> Recording:
    """Construct canonical local EEGMMIDB paths."""
    subject_id = f"S{subject:03d}"
    stem = f"{subject_id}R{run:02d}.edf"
    edf_path = data_dir / subject_id / stem
    return Recording(subject, run, edf_path, edf_path.with_suffix(".edf.event"))


def discover_subjects(data_dir: Path) -> list[int]:
    """Return numeric IDs for directories matching S###."""
    subjects: list[int] = []
    for path in data_dir.glob("S[0-9][0-9][0-9]"):
        if path.is_dir():
            subjects.append(int(path.name[1:]))
    return sorted(subjects)


def validate_left_right_runs(runs: tuple[int, ...]) -> None:
    invalid = sorted(set(runs) - LEFT_RIGHT_RUNS)
    if invalid:
        raise ValueError(
            f"Runs {invalid} are not left-vs-right fist runs; choose from "
            f"{sorted(LEFT_RIGHT_RUNS)}"
        )


def semantic_label(run: int, annotation: str) -> int:
    """Map a run-specific annotation to 0=left fist or 1=right fist."""
    if run not in LEFT_RIGHT_RUNS:
        raise ValueError(f"Run {run} is not a left-vs-right fist run")
    try:
        return {"T1": 0, "T2": 1}[annotation]
    except KeyError as exc:
        raise ValueError(f"Annotation {annotation!r} is not a class cue") from exc


def condition_for_run(run: int) -> str:
    if run in IMAGINED_LEFT_RIGHT_RUNS:
        return "imagined"
    if run in EXECUTED_LEFT_RIGHT_RUNS:
        return "executed"
    raise ValueError(f"Run {run} is not a left-vs-right fist run")
