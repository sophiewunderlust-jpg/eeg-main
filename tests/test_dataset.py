from pathlib import Path

import pytest

from eeg_lr.dataset import (
    condition_for_run,
    discover_subjects,
    recording_paths,
    semantic_label,
    validate_left_right_runs,
)


def test_recording_paths() -> None:
    recording = recording_paths(Path("/dataset"), 1, 4)
    assert recording.edf_path == Path("/dataset/S001/S001R04.edf")
    assert recording.event_path == Path("/dataset/S001/S001R04.edf.event")


def test_discover_subjects_ignores_other_entries(tmp_path: Path) -> None:
    (tmp_path / "S002").mkdir()
    (tmp_path / "S001").mkdir()
    (tmp_path / "notes").mkdir()
    assert discover_subjects(tmp_path) == [1, 2]


@pytest.mark.parametrize("run", [3, 4, 7, 8, 11, 12])
def test_left_right_event_mapping(run: int) -> None:
    assert semantic_label(run, "T1") == 0
    assert semantic_label(run, "T2") == 1


def test_rejects_other_run_families() -> None:
    with pytest.raises(ValueError, match="not left-vs-right"):
        validate_left_right_runs((5, 9, 13))


def test_condition_mapping() -> None:
    assert condition_for_run(4) == "imagined"
    assert condition_for_run(3) == "executed"
