import json
from pathlib import Path

import pytest

from cardiac_agent.benchmark import evaluate_studies
from cardiac_agent.cli import make_fixture
from cardiac_agent.pipeline import save_json


def manual_fixture(tmp_path, name):
    path = make_fixture(tmp_path / name)
    data = json.loads(path.read_text())
    data.update(mask_source="manual", view="A4C")
    save_json(path, data)
    return path


class ThresholdSegmenter:
    def segment(self, image):
        return image[:, :, 0] > 80


def test_paired_metrics_and_failed_case_denominator(tmp_path):
    valid = manual_fixture(tmp_path, "valid")
    invalid = manual_fixture(tmp_path, "invalid")
    data = json.loads(invalid.read_text())
    data["view"] = "A2C"
    save_json(invalid, data)
    output = tmp_path / "evaluation"
    summary = evaluate_studies([valid, invalid], output, ThresholdSegmenter())
    assert summary["completed"] == 1 and summary["failed"] == 1
    assert summary["failure_rate"] == .5
    assert summary["mean_dice"] == 1 and summary["fac_mae_pp"] == 0
    rows = json.loads((output / "cases.json").read_text())
    request = json.loads((Path(rows[0]["predicted_run"]) / "request.json").read_text())
    assert request["ed_mask"] is None and request["mask_source"] == "predicted"


def test_all_failed_has_no_fabricated_scores(tmp_path):
    path = manual_fixture(tmp_path, "case")

    class EmptySegmenter:
        def segment(self, image):
            return image[:, :, 0] < 0

    summary = evaluate_studies([path], tmp_path / "output", EmptySegmenter())
    assert summary["failed"] == 1 and summary["frames_evaluated"] == 0
    assert summary["mean_dice"] is None and summary["fac_mae_pp"] is None


def test_duplicate_manifests_rejected(tmp_path):
    path = manual_fixture(tmp_path, "case")
    with pytest.raises(ValueError, match="unique"):
        evaluate_studies([path, path], tmp_path / "output", ThresholdSegmenter())
