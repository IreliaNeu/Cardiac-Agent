"""Complete unspoken roles for empty-mask failures, without repeating discussion."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from pydantic import field_validator

from cardiac_agent.cli import load_study
from cardiac_agent.discussion import (
    BOUNDARY,
    LIMITATIONS,
    Draft,
    Plan,
    Review,
    export_images,
    role_call,
    validate_report,
)
from cardiac_agent.evaluation import segmentation_metrics
from cardiac_agent.explanation import QwenClient, unique_object
from cardiac_agent.pipeline import digest, save_json
from cardiac_agent.roi import EchoNetSegmenter, read_image


def decode_evidence(value):
    # Some providers encode the evidence object as a JSON string; decode losslessly.
    return json.loads(value, object_pairs_hook=unique_object) if isinstance(value, str) else value


class FailureDraft(Draft):
    decode = field_validator("evidence", mode="before")(decode_evidence)


class FailureReview(Review):
    decode = field_validator("evidence", mode="before")(decode_evidence)


def verify_tree(root):
    checks = json.loads((root / "checksums.json").read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and p != root / "checksums.json"}
    if checks.keys() != actual or any(digest(root / p) != h for p, h in checks.items()):
        raise ValueError("source_batch_integrity_failure")


def seal(root):
    save_json(root / "checksums.json", {
        str(p.relative_to(root)): digest(p) for p in root.rglob("*")
        if p.is_file() and p != root / "checksums.json"})


def failure_report(directory, facts, client):
    planner = json.loads((directory / "planner.json").read_text(encoding="utf-8"))
    if planner["status"] != "received" or planner["round"] != 1:
        raise ValueError("no_valid_existing_planner")
    plan = Plan.model_validate(json.loads(planner["response"], object_pairs_hook=unique_object))
    context = {"plan": plan.model_dump(), "evidence": facts,
               "tool_failure": "Empty predicted cavity mask; measurement stage refused input.",
               "limitations": LIMITATIONS}
    instructions = (BOUNDARY + "The segmentation tool failed. Mask pixel counts describe "
                    "algorithm output, NOT anatomical cavity area. Do not compute FAC, area ratio "
                    "or diagnose. Explain missing evidence and request human review. ")
    draft = role_call(directory, "analyst", client, instructions +
        'Return {"evidence":exact unchanged evidence,"narrative":"failure report draft"}.',
        context, FailureDraft)
    validate_report(draft, facts)
    review = role_call(directory, "reviewer", client, instructions +
        'Review once, return {"evidence":exact unchanged evidence, '
        '"narrative":"final failure report","verdict":"refer","concerns":["..."]}.',
        {**context, "draft": draft.model_dump()}, FailureReview)
    validate_report(review, facts)
    record = {"status": "tool_failed_reported", "rounds": 1, "role_turns": 3,
              "model": client.model, "report": review.model_dump(),
              "effective_verdict": "refer", "qc_override": True,
              "clinical_validation": False, "limitations": LIMITATIONS,
              "qc_flags": ["empty_predicted_mask_measurements_unavailable"]}
    save_json(directory / "report.json", record)
    table = "\n".join(f"| {key} | {value} |" for key, value in facts.items())
    (directory / "report.md").write_text(
        "# 工具失败：需人工复核\n\n" + review.narrative +
        "\n\n## 强制边界\n\n本例未完成有效测量。空 mask 是模型失败，不能解释为心腔不存在，"
        "不能计算 FAC 或得出临床结论。ED/ES 图像和原始预测均保留，未使用人工 mask 修补。\n\n" +
        "\n".join(review.concerns) + "\n\n| 证据 | 值 |\n|---|---|\n" + table +
        "\n\n" + LIMITATIONS + "\n", encoding="utf-8")
    save_json(directory / "agent_trace.json", [
        json.loads((directory / f"{role}.json").read_text(encoding="utf-8"))
        for role in ("planner", "analyst", "reviewer")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/echonet.official.json"))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("delivery_must_be_separate_from_source")
    verify_tree(source)
    origin = {"source": str(source), "source_checksums_sha256": digest(source / "checksums.json")}
    if output.exists():
        if json.loads((output / "delivery_provenance.json").read_text()) != origin:
            raise ValueError("delivery_origin_changed")
    else:
        shutil.copytree(source, output)
        save_json(output / "delivery_provenance.json", origin)
    load_dotenv(args.env)
    import torch
    torch.set_num_threads(2)
    config = json.loads(args.config.read_text())
    checkpoint = (args.config.resolve().parent / config["checkpoint"]).resolve()
    summary = json.loads((source / "summary.json").read_text())
    if any(value != summary["segmentation"].get(key) for key, value in config.items()):
        raise ValueError("segmentation_configuration_changed")
    if QwenClient().model != summary["model"]:
        raise ValueError("reporting_model_changed")
    if digest(checkpoint) != summary["segmentation"]["checkpoint_sha256"]:
        raise ValueError("segmentation_checkpoint_changed")
    segmenter = EchoNetSegmenter(checkpoint, config["mean"], config["std"], config["device"])
    rows = json.loads((source / "results.json").read_text())
    for row in rows:
        if row["status"] == "completed":
            continue
        directory = output / row["patient"]
        if row["status"] != "failed" or any((source / row["patient"] / f"{r}.json").exists()
                                             for r in ("analyst", "reviewer")):
            raise ValueError("only_pre_analyst_tool_failures_supported")
        failures = list((directory / "execution").glob("*/failure.json"))
        if len(failures) != 1 or json.loads(failures[0].read_text())["completed_stages"]:
            raise ValueError("not_an_initial_roi_failure")
        study = load_study(args.data / "converted" / f"{row['patient']}-4CH" / "study.json")
        diagnostic = directory / "failure_diagnostic"
        diagnostic.mkdir(exist_ok=True)
        counts, metrics = {}, {}
        for phase in ("ed", "es"):
            pred = segmenter.segment(read_image(getattr(study, phase + "_image")))
            path = diagnostic / f"{phase}_mask.png"
            if path.exists() and not np.array_equal(np.asarray(Image.open(path)) > 0, pred):
                raise ValueError("diagnostic_prediction_changed")
            Image.fromarray(pred.astype(np.uint8)).save(path)
            counts[f"{phase}_mask_nonzero_pixels"] = str(int(pred.sum()))
        if "0" not in counts.values():
            raise ValueError("diagnostic_did_not_confirm_empty_mask")
        export_images(study, diagnostic, directory)
        facts = {**counts, "segmentation_status": "failed_empty_mask",
                 "fac_percent": "unavailable", "area_ratio": "unavailable",
                 "rwma_status": "unavailable", "deformation_method": "none"}
        save_json(diagnostic / "evidence.json", facts)
        save_json(diagnostic / "provenance.json", {
            "ed_image": str(study.ed_image), "es_image": str(study.es_image),
            "ed_sha256": digest(study.ed_image), "es_sha256": digest(study.es_image),
            "segmentation": summary["segmentation"], "script_sha256": digest(Path(__file__)),
            "purpose": "Diagnostic re-execution preserves raw empty predictions; no repair."})
        failure_report(directory, facts, QwenClient())
        for phase in ("ed", "es"):
            pred = np.asarray(Image.open(diagnostic / f"{phase}_mask.png")) > 0
            reference = np.asarray(Image.open(getattr(study, phase + "_mask"))) == 1
            metrics[phase] = segmentation_metrics(pred, reference)
        save_json(directory / "independent_evaluation.json", {
            "segmentation": metrics, "fac_absolute_error_pp": None,
            "reference_used_by_agents": False, "measurement_eligible": False})
        row.update(status="failed_reported", report=f"{row['patient']}/report.md", verdict="refer",
                   dice=float(np.mean([m["dice"] for m in metrics.values()])),
                   iou=float(np.mean([m["iou"] for m in metrics.values()])),
                   fac_absolute_error_pp=None)
        save_json(directory / "result.json", row)
        seal(directory)
        print(json.dumps(row), flush=True)
    summary.update(reported=len(rows), tool_failures_reported=sum(
        r["status"] == "failed_reported" for r in rows),
        all_attempts_mean_dice=float(np.mean([r["dice"] for r in rows])),
        all_attempts_mean_iou=float(np.mean([r["iou"] for r in rows])),
        note="completed/failed retain original measurement outcomes; reported includes failures.")
    save_json(output / "summary.json", summary)
    save_json(output / "results.json", rows)
    with (output / "index.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["patient", "status", "report", "verdict",
                                                   "dice", "iou", "fac_absolute_error_pp"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output / "README.md").write_text(
        (source / "README.md").read_text(encoding="utf-8") +
        "\n## 交付补充\n\n空 mask 失败病例补齐了原始预测图及失败报告，"
        "但仍计为测量失败，FAC 不可用。只补充尚未发言的 Analyst 和 Reviewer，"
        "没有重新生成 Planner 或增加讨论轮次。原始批次另行保留，不覆盖。\n",
        encoding="utf-8")
    seal(output)
    verify_tree(output)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
