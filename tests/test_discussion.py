import json
import runpy
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cardiac_agent.cli import load_study, make_fixture
from cardiac_agent.discussion import Draft, discuss, validate_report
from cardiac_agent.pipeline import save_json


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


def test_duplicate_role_keys_rejected_without_regeneration(tmp_path):
    class Duplicate(Client):
        def complete(self, system, context):
            self.calls.append(context)
            return ('{"motion":"none","motion":"farneback",'
                    '"focus":"area_only","reason":"research"}')

    client = Duplicate()
    study = load_study(make_fixture(tmp_path / "fixture"))
    for _ in range(2):
        with pytest.raises(ValueError, match="duplicate_json_key"):
            discuss(study, tmp_path / "result", client, Segmenter(), threading.Lock(), {})
    assert len(client.calls) == 1


def test_changed_image_cannot_reuse_cached_measurements(tmp_path):
    client = Client()
    study = load_study(make_fixture(tmp_path / "fixture"))
    directory = tmp_path / "result"
    discuss(study, directory, client, Segmenter(), threading.Lock(), {})
    with Image.open(study.ed_image) as im:
        changed = np.asarray(im).copy()
    changed[0, 0] = 123
    Image.fromarray(changed).save(study.ed_image)
    with pytest.raises(ValueError, match="cached_execution_changed"):
        discuss(study, directory, client, Segmenter(), threading.Lock(), {})
    assert len(client.calls) == 3


def test_failure_report_completes_only_unspoken_roles(tmp_path):
    functions = runpy.run_path(str(Path(__file__).parents[1] / "scripts/finalize_failed_cases.py"))
    save_json(tmp_path / "planner.json", {
        "status": "received", "round": 1, "role": "planner", "response": json.dumps({
            "motion": "none", "focus": "area_only", "reason": "two frames"})})
    facts = {"ed_mask_nonzero_pixels": "20", "es_mask_nonzero_pixels": "0",
             "fac_percent": "unavailable"}
    client = Client()
    for _ in range(2):
        functions["failure_report"](tmp_path, facts, client)
    assert len(client.calls) == 2
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "tool_failed_reported"
    assert report["effective_verdict"] == "refer"
    assert report["report"]["evidence"]["fac_percent"] == "unavailable"
    assert len(json.loads((tmp_path / "agent_trace.json").read_text())) == 3


def test_failure_evidence_json_string_decoded_without_changing_values():
    functions = runpy.run_path(str(Path(__file__).parents[1] / "scripts/finalize_failed_cases.py"))
    cls = functions["FailureDraft"]
    facts = {"fac_percent": "unavailable", "es_mask_nonzero_pixels": "0"}
    draft = cls.model_validate({"evidence": json.dumps(facts), "narrative": "Tool failure."})
    assert draft.evidence == facts
    with pytest.raises(ValueError, match="duplicate_json_key"):
        cls.model_validate({"evidence": '{"x":"0","x":"1"}', "narrative": "failure"})
