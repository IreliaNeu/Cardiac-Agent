import json
import threading

import numpy as np
import pytest

from cardiac_agent.cli import load_study, make_fixture
from cardiac_agent.discussion import Draft, discuss, validate_report


class Client:
    model = "fake"

    def __init__(self):
        self.calls = []
        self.telemetry = {}

    def complete(self, system, context):
        self.calls.append(context)
        self.telemetry = {"attempts": 1}
        if "available_tools" in context:
            return json.dumps({"motion": "none", "focus": "area_only", "reason": "two frames"})
        result = {"evidence": context["evidence"], "narrative": "Research findings only."}
        if "draft" in context:
            result.update(verdict="limited", concerns=["No clinical validation."])
        return json.dumps(result)


class Segmenter:
    def segment(self, image):
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[12:40, 12:40] = True
        return mask


def test_single_round_no_reference_leak_and_resume(tmp_path):
    study = load_study(make_fixture(tmp_path / "fixture"))
    client = Client()
    directory = tmp_path / "result"
    def execute():
        return discuss(study, directory, client, Segmenter(), threading.Lock(), {})
    execute()
    assert len(client.calls) == 3
    serialized = json.dumps(client.calls)
    for forbidden in (str(study.ed_image), str(study.ed_mask), "dice", "reference_fac"):
        assert forbidden not in serialized
    result = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert result["role_turns"] == 3
    assert result["effective_verdict"] == "refer"
    assert result["qc_override"]
    assert (directory / "segmentation" / "ED_overlay.png").exists()
    execute()
    assert len(client.calls) == 3
    client.model = "different"
    with pytest.raises(ValueError, match="cached_role_request_changed"):
        execute()


def test_report_rejects_changed_evidence_and_new_numbers():
    with pytest.raises(ValueError, match="evidence_mismatch"):
        validate_report(Draft(evidence={"x": "2"}, narrative="finding"), {"x": "1"})
    with pytest.raises(ValueError, match="numeric_prose"):
        validate_report(Draft(evidence={"x": "1"}, narrative="EF is 55%"), {"x": "1"})


def test_report_accepts_grounded_rounding():
    facts = {"fac_percent": "40.1523", "status": "unavailable"}
    validate_report(Draft(evidence=facts, narrative="FAC: 40.15%"), facts)
    with pytest.raises(ValueError):
        validate_report(Draft(evidence=facts, narrative="FAC: 40.16%"), facts)


def test_failed_call_is_not_automatically_replayed(tmp_path):
    class Broken(Client):
        def complete(self, system, context):
            self.calls.append(context)
            raise RuntimeError("http_500")

    client = Broken()
    study = load_study(make_fixture(tmp_path / "fixture"))
    for _ in range(2):
        with pytest.raises(RuntimeError):
            discuss(study, tmp_path / "result", client, Segmenter(), threading.Lock(), {})
    assert len(client.calls) == 1
