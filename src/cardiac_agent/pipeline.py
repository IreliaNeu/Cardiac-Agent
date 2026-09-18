import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from .evaluation import evaluate_qa
from .explanation import explain
from .function import measure
from .roi import roi_stage
from .rwma import UnavailableRWMA, predict_rwma
from .schemas import MedicalKnowledgePacket, Study


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest(path)}


def save_json(path: Path, obj):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run(study: Study, output: Path, client=None, segmenter=None, predictor=None,
        deformation="intensity_difference", model_provenance=None) -> Path:
    directory = output / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    trace = []
    try:
        save_json(directory / "request.json", study.model_dump(mode="json"))
        ed, es, med, mes = roi_stage(study, segmenter)
        trace.append("roi")
        refs = {"ed_image": artifact(study.ed_image), "es_image": artifact(study.es_image)}
        if study.ed_mask is not None:
            refs.update(ed_mask_input=artifact(study.ed_mask), es_mask_input=artifact(study.es_mask))
        for name, mask in (("ed_mask", med), ("es_mask", mes)):
            path = directory / f"{name}.png"
            Image.fromarray(mask.astype(np.uint8)).save(path)
            refs[name] = artifact(path)
        m, dmap = measure(ed, es, med, mes, study.spacing_mm, deformation)
        if dmap is not None:
            path = directory / "deformation.npy"
            np.save(path, np.where(med, dmap, 0), allow_pickle=False)
            refs["deformation"] = artifact(path)
        save_json(directory / "measurements.json", m.model_dump())
        trace.append("function")
        y = predict_rwma(m, predictor or UnavailableRWMA())
        save_json(directory / "rwma.json", y.model_dump())
        trace.append("rwma")
        packet = MedicalKnowledgePacket(
            study_id=study.study_id, view=study.view, synthetic=study.synthetic,
            phase_source=study.phase_source,
            mask_source=study.mask_source if study.ed_mask else "predicted",
            measurements=m, rwma=y, artifacts=refs)
        save_json(directory / "knowledge.json", packet.model_dump())
        trace.append("knowledge_bridge")
        explanation = explain(packet, client)
        save_json(directory / "explanation.json", explanation.model_dump())
        trace.append("explanation")
        save_json(directory / "evaluation.json", evaluate_qa(packet, study.rwma_label))
        trace.append("qa_evaluation")
        save_json(directory / "run.json", {
            "status": "completed", "stages": trace,
            "created_at": datetime.now(UTC).isoformat(),
            "synthetic": study.synthetic, "rwma_status": y.status,
            "llm_model": client.model if client else None,
            "api_telemetry": client.telemetry if client else {},
            "segmentation": model_provenance or {"backend": "precomputed_masks"},
            "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
            "explanation_mode": explanation.mode})
        save_json(directory / "checksums.json", {
            p.name: digest(p) for p in directory.iterdir() if p.is_file()})
        return directory
    except Exception as exc:
        save_json(directory / "failure.json", {
            "status": "failed", "completed_stages": trace, "error_type": type(exc).__name__})
        raise
