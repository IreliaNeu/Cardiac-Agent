import json

import pytest

from cardiac_agent.cli import load_study, make_fixture
from cardiac_agent.explanation import evidence_claims, explain
from cardiac_agent.pipeline import run
from cardiac_agent.schemas import MedicalKnowledgePacket
from cardiac_agent.smoke import smoke


def packet_at(tmp_path, deformation):
    study = load_study(make_fixture(tmp_path / "fixture"))
    output = run(study, tmp_path / "runs", deformation=deformation)
    return MedicalKnowledgePacket.model_validate_json((output / "knowledge.json").read_text())


def test_motion_facts_are_forwarded_and_rendered(tmp_path):
    packet = packet_at(tmp_path, "intensity_difference")
    facts = {c.evidence_id: c.value for c in evidence_claims(packet)}
    assert facts["deformation_unit"] == "normalized_intensity_difference"
    assert facts["deformation_mean"] == f"{packet.measurements.deformation_mean:.6f}"
    assert "squared units" in explain(packet).text


def test_no_motion_does_not_invent_statistics(tmp_path):
    packet = packet_at(tmp_path, "none")
    facts = {c.evidence_id: c.value for c in evidence_claims(packet)}
    assert facts["deformation_method"] == "none"
    assert "deformation_mean" not in facts and "deformation_var" not in facts


def test_duplicate_json_evidence_rejected(tmp_path):
    packet = packet_at(tmp_path, "none")

    class DuplicateClient:
        def organize(self, claims):
            valid = json.dumps({c.evidence_id: c.value for c in claims})
            return valid[:-1] + ', "area_ed_px": "1257"}'

    assert explain(packet, DuplicateClient()).mode == "template_fallback"


def test_acceptance_offline_success_and_no_overwrite(tmp_path):
    output = tmp_path / "smoke"
    result = smoke(output)
    assert result["passed"] and result["rwma_status"] == "unavailable"
    assert result["verified_artifacts"] > 5
    with pytest.raises(FileExistsError):
        smoke(output)


def test_requested_qwen_fallback_fails_acceptance(tmp_path):
    class FailingClient:
        model = "test"

        def __init__(self):
            self.telemetry = {}

        def organize(self, claims):
            raise RuntimeError("missing_api_key")

    result = smoke(tmp_path / "smoke", client=FailingClient())
    assert not result["passed"] and result["explanation_mode"] == "template_fallback"
