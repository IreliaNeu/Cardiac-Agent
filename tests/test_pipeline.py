import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from cardiac_agent.cli import load_study, make_fixture
from cardiac_agent.evaluation import segmentation_metrics
from cardiac_agent.explanation import QwenClient, explain
from cardiac_agent.function import measure
from cardiac_agent.pipeline import digest, run
from cardiac_agent.roi import read_mask
from cardiac_agent.schemas import MedicalKnowledgePacket, Prediction, Study


@pytest.fixture
def study(tmp_path):
    return load_study(make_fixture(tmp_path / "fixture"))


def test_full_cpu_pipeline_and_independent_geometry(study, tmp_path):
    output = run(study, tmp_path / "runs")
    m = json.loads((output / "measurements.json").read_text())
    assert m["area_ed_px"] == 1257
    assert m["area_es_px"] == 709
    assert m["fac_percent"] == pytest.approx(548 / 1257 * 100)
    assert m["area_ed_mm2"] == 314.25
    assert json.loads((output / "rwma.json").read_text())["status"] == "unavailable"
    ev = json.loads((output / "evaluation.json").read_text())
    assert ev["rwma_accuracy"] is None and ev["arithmetic_consistent"]
    for name, checksum in json.loads((output / "checksums.json").read_text()).items():
        assert digest(output / name) == checksum


@pytest.mark.parametrize("label", [0, 256])
def test_invalid_label(study, label):
    with pytest.raises(ValueError):
        Study.model_validate({**study.model_dump(), "foreground_label": label})


def test_missing_mask_and_bad_spacing(study):
    for changed in ({"ed_mask": None}, {"spacing_mm": (0, 1)}):
        with pytest.raises(ValueError):
            Study.model_validate({**study.model_dump(), **changed})


def test_mask_not_silently_relabelled(study):
    with pytest.raises(ValueError, match="Empty LV"):
        read_mask(study.ed_mask, (64, 64), 255)


def test_mask_shape_rejected(study):
    with pytest.raises(ValueError, match="dimensions"):
        read_mask(study.ed_mask, (32, 32), 1)


def test_swapped_phases_abstain(study, tmp_path):
    swapped = study.model_copy(update={"ed_image": study.es_image, "es_image": study.ed_image,
                                      "ed_mask": study.es_mask, "es_mask": study.ed_mask})
    output = run(swapped, tmp_path / "runs")
    assert json.loads((output / "rwma.json").read_text())["status"] == "abstained"


def test_empty_mask_preserves_failure(study, tmp_path):
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(study.ed_mask)
    with pytest.raises(ValueError):
        run(study, tmp_path / "runs")
    assert len(list((tmp_path / "runs").glob("*/failure.json"))) == 1


def test_no_masks_requires_backend(study, tmp_path):
    study = study.model_copy(update={"ed_mask": None, "es_mask": None})
    with pytest.raises(ValueError, match="precomputed"):
        run(study, tmp_path / "runs")


def test_segmentation_metrics_not_identity():
    pred = np.array([[1, 1], [0, 0]])
    truth = np.array([[1, 0], [1, 0]])
    metrics = segmentation_metrics(pred, truth)
    assert metrics["dice"] == .5 and metrics["iou"] == pytest.approx(1 / 3)


def test_none_and_unknown_deformation():
    image = np.zeros((4, 4, 3), np.uint8)
    mask = np.ones((4, 4), bool)
    m, d = measure(image, image, mask, mask, deformation="none")
    assert m.deformation_mean is None and d is None
    with pytest.raises(ValueError):
        measure(image, image, mask, mask, deformation="unknown")


@pytest.fixture
def packet(study, tmp_path):
    output = run(study, tmp_path / "runs")
    return MedicalKnowledgePacket.model_validate_json((output / "knowledge.json").read_text())


def test_qwen_success_and_no_images(packet, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-not-real")
    def respond(request):
        data = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer test-not-real"
        claims = json.loads(data["messages"][1]["content"])
        assert len(claims) == 9 and "image_url" not in request.content.decode()
        assert packet.study_id not in request.content.decode()
        assert {c["evidence_id"] for c in claims} >= {
            "deformation_method", "deformation_mean", "deformation_var", "deformation_unit"}
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps({"claims": list(reversed(claims))})}}]})
    result = explain(packet, QwenClient(httpx.MockTransport(respond)))
    assert result.mode == "qwen" and result.verified


@pytest.mark.parametrize("body", ["not json", '{"claims": []}', '{"extra": "diagnosis"}'])
def test_invalid_llm_claims_fallback(packet, monkeypatch, body):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-not-real")
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
        "choices": [{"message": {"content": body}}]}))
    result = explain(packet, QwenClient(transport))
    assert result.mode == "template_fallback"
    assert result.api_status == "invalid_claim_contract"


def test_http_error_redacted(packet, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "secret-sentinel")
    transport = httpx.MockTransport(lambda _: httpx.Response(403, text="secret-sentinel"))
    result = explain(packet, QwenClient(transport))
    assert result.api_status == "http_403"
    assert "secret-sentinel" not in result.model_dump_json()


def test_api_retry_bounded(packet, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-not-real")
    monkeypatch.setattr("cardiac_agent.explanation.time.sleep", lambda _: None)
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(429)
    result = explain(packet, QwenClient(httpx.MockTransport(respond)))
    assert len(requests) == 3 and result.api_status == "http_429"


def test_missing_key_fallback(packet, monkeypatch):
    monkeypatch.setenv("CARDIAC_API_KEY_ENV", "TEST_MISSING_KEY")
    monkeypatch.delenv("TEST_MISSING_KEY", raising=False)
    assert explain(packet, QwenClient()).api_status == "missing_api_key"


def test_request_paths_relative_to_manifest(study):
    assert isinstance(study.ed_image, Path) and study.ed_image.is_absolute()


def test_fake_predictor_injection_not_clinical_model(study, tmp_path):
    class FakePredictor:
        def predict(self, features):
            return Prediction(status="predicted", probability=.8, label=1,
                              reason="test-only stub")
    output = run(study.model_copy(update={"rwma_label": 1}), tmp_path / "runs",
                 predictor=FakePredictor())
    assert json.loads((output / "evaluation.json").read_text())["rwma_accuracy"] == 1


@pytest.mark.parametrize("data", [
    {"status": "predicted"}, {"status": "unavailable", "label": 0},
    {"status": "abstained", "probability": .3},
])
def test_rwma_status_contract(data):
    with pytest.raises(ValueError):
        Prediction(reason="test", **data)


def test_fabricated_number_rejected(packet, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-not-real")
    def respond(request):
        claims = json.loads(json.loads(request.content)["messages"][1]["content"])
        claims[0]["value"] = "999999"
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps({"claims": claims})}}]})
    result = explain(packet, QwenClient(httpx.MockTransport(respond)))
    assert result.mode == "template_fallback" and "999999" not in result.text


def test_flat_provider_evidence_map(packet, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-not-real")
    def respond(request):
        claims = json.loads(json.loads(request.content)["messages"][1]["content"])
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
            {c["evidence_id"]: c["value"] for c in claims})}}]})
    assert explain(packet, QwenClient(httpx.MockTransport(respond))).mode == "qwen"
