"""One-pass, three-role discussion over computed evidence, never reference labels."""
import hashlib
import json
import re
import shutil
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field

from .explanation import evidence_claims, unique_object
from .pipeline import digest, run, save_json
from .schemas import Contract, MedicalKnowledgePacket


class Plan(Contract):
    motion: Literal["farneback", "none"]
    focus: Literal["area_and_motion", "area_only"]
    reason: str = Field(min_length=1, max_length=1200)


class Draft(Contract):
    evidence: dict[str, str]
    narrative: str = Field(min_length=1, max_length=4000)


class Review(Draft):
    verdict: Literal["limited", "refer"]
    concerns: list[str]


LIMITATIONS = (
    "本结果仅用于研究流程验证，不构成临床诊断。FAC 是双帧心腔面积变化，不是 LVEF。"
    "稀疏心腔光流不是心肌应变；未分析完整心动周期。当前没有经过验证的 RWMA 分类器，"
    "不能据此判断局部室壁运动正常或异常。跨数据集分割模型尚未完成临床验证。"
)
BOUNDARY = (
    "You are part of a cardiac research workflow, not a diagnostic service. "
    "Return ONLY valid JSON, without markdown fences. Use concise Chinese prose. "
    "You do not see images. Reference masks and clinical labels are unavailable. "
    "Do not diagnose, infer normal/abnormal wall motion, or claim to inspect images. "
    "FAC is cavity area change, not EF; cavity flow is not myocardial strain. "
    "Prefer qualitative prose. Any numeric value must come from input evidence, "
    "with its correct meaning and units; do not introduce counts or thresholds. "
    "The application appends an exact numerical evidence table. "
)


def role_call(directory, role, client, system, context, contract):
    request = {"system": system, "context": context, "model": client.model}
    signature = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    path = directory / f"{role}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["request_sha256"] != signature:
            raise ValueError("cached_role_request_changed")
        if record["status"] != "received":
            raise RuntimeError("previous_role_attempt_failed_no_automatic_replay")
    else:
        record = {"role": role, "round": 1, "request_sha256": signature,
                  "request": request, "status": "started"}
        save_json(path, record)
        try:
            record["response"] = client.complete(system, context)
            record["status"] = "received"
        finally:
            record["telemetry"] = client.telemetry
            save_json(path, record)
    return contract.model_validate(json.loads(record["response"], object_pairs_hook=unique_object))


def validate_report(report, facts):
    if report.evidence != facts:
        raise ValueError("evidence_mismatch")
    prose = report.narrative + " ".join(getattr(report, "concerns", []))
    values = []
    for value in facts.values():
        try:
            number = Decimal(value)
            if number.is_finite():
                values.append(number)
        except InvalidOperation:
            pass
    for token in re.findall(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?", prose):
        number = Decimal(token)
        if not any(value.quantize(Decimal(1).scaleb(number.as_tuple().exponent),
                                  rounding=ROUND_HALF_UP) == number for value in values):
            raise ValueError("ungrounded_numeric_prose")


def export_images(study, execution, directory):
    for folder in ("original", "segmentation"):
        (directory / folder).mkdir(exist_ok=True)
    for phase in ("ed", "es"):
        source = getattr(study, phase + "_image")
        shutil.copy2(source, directory / "original" / f"{phase.upper()}.png")
        mask = np.asarray(Image.open(execution / f"{phase}_mask.png")) > 0
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            directory / "segmentation" / f"{phase.upper()}_mask.png")
        rgb = np.asarray(Image.open(source).convert("RGB")).copy()
        rgb[mask] = (0.65 * rgb[mask] + 0.35 * np.array([0, 255, 100])).astype(np.uint8)
        Image.fromarray(rgb).save(directory / "segmentation" / f"{phase.upper()}_overlay.png")


def discuss(study, directory: Path, client, segmenter, lock, provenance):
    """Three role turns; cached responses never trigger a second semantic discussion."""
    directory.mkdir(parents=True, exist_ok=True)
    study = study.model_copy(update={"ed_mask": None, "es_mask": None,
                                    "mask_source": "predicted", "rwma_label": None})
    with Image.open(study.ed_image) as im:
        dimensions = list(im.size)
    plan = role_call(directory, "planner", client, BOUNDARY +
        'Select a real tool plan: {"motion":"farneback" or "none", '
        '"focus":"area_and_motion" or "area_only", "reason":"..."}. '
        "Use farneback only for exploratory sparse motion; otherwise none. "
        "Keep focus consistent with motion.",
        {"view": study.view, "image_dimensions": dimensions,
         "phase_source": study.phase_source, "frames": "ED and ES only",
         "available_tools": ["LV segmentation", "cavity area", "sparse Farneback flow"],
         "rwma_classifier": "unavailable"}, Plan)
    if (plan.motion == "none") != (plan.focus == "area_only"):
        raise ValueError("inconsistent_plan")
    execution_path = directory / "execution.json"
    identity = {"ed": digest(study.ed_image), "es": digest(study.es_image),
                "plan": plan.model_dump(), "model": provenance,
                "spacing_mm": study.spacing_mm,
                "code": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}}
    identity = json.loads(json.dumps(identity))
    if execution_path.exists():
        cached = json.loads(execution_path.read_text())
        if cached["identity"] != identity:
            raise ValueError("cached_execution_changed")
        execution = Path(cached["directory"])
        checks = json.loads((execution / "checksums.json").read_text())
        if any(digest(execution / name) != sha for name, sha in checks.items()):
            raise ValueError("execution_checksum_mismatch")
    else:
        with lock:
            execution = run(study, directory / "execution", segmenter=segmenter,
                            deformation=plan.motion, model_provenance=provenance)
        save_json(execution_path, {"directory": str(execution), "identity": identity})
    export_images(study, execution, directory)
    packet = MedicalKnowledgePacket.model_validate_json(
        (execution / "knowledge.json").read_text())
    facts = {c.evidence_id: c.value for c in evidence_claims(packet)}
    context = {"plan": plan.model_dump(), "evidence": facts,
               "qc_flags": packet.measurements.qc_flags, "limitations": LIMITATIONS}
    draft = role_call(directory, "analyst", client, BOUNDARY +
        'Return {"evidence":exact unchanged input evidence object,"narrative":"..."}. '
        "Summarize available measurements and limitations, addressing QC flags.", context, Draft)
    validate_report(draft, facts)
    review = role_call(directory, "reviewer", client, BOUNDARY +
        'Review the analyst once. Return {"evidence":exact unchanged evidence object, '
        '"narrative":"final corrected report", "verdict":"limited" or "refer", '
        '"concerns":["..."]}. Choose refer for segmentation/phase QC warnings. '
        "Do not request another discussion. Preserve evidence and acknowledge limitations.",
        {**context, "draft": draft.model_dump()}, Review)
    validate_report(review, facts)
    blocking = any(q.startswith(("mask_touches", "non_decreasing"))
                   for q in packet.measurements.qc_flags)
    effective = "refer" if blocking else review.verdict
    result = {"status": "completed", "rounds": 1, "role_turns": 3,
              "model": client.model, "report": review.model_dump(),
              "effective_verdict": effective, "qc_override": blocking,
              "clinical_validation": False, "limitations": LIMITATIONS,
              "qc_flags": packet.measurements.qc_flags}
    save_json(directory / "report.json", result)
    table = "\n".join(f"| {key} | {value} |" for key, value in facts.items())
    text = (f"# {study.study_id}\n\n## 模型报告\n\n{review.narrative}\n\n"
            f"## 审查\n\n状态：{effective}\n\n" + "\n".join(review.concerns) +
            f"\n\n## 计算证据\n\n| 指标 | 数值 |\n|---|---|\n{table}\n\n"
            f"## 固定适用边界\n\n{LIMITATIONS}\n\nQC: " +
            "; ".join(packet.measurements.qc_flags) + "\n")
    (directory / "report.md").write_text(text, encoding="utf-8")
    save_json(directory / "agent_trace.json", [
        json.loads((directory / f"{role}.json").read_text(encoding="utf-8"))
        for role in ("planner", "analyst", "reviewer")])
    return execution
