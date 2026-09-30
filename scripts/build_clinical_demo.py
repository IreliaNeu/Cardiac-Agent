"""Build a read-only five-case demonstration from sealed results. No model calls."""
import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from cardiac_agent.evaluation import segmentation_metrics
from cardiac_agent.pipeline import digest, save_json

NOTES = {
    "good": ("较好分割与一致的数值", "先展示完整的输入、分割、计算和报告链条。", [
        "对照 ED、ES 原图与人工轮廓，说明预测 mask 才是数值计算的来源。",
        "较高 Dice 和较小 FAC 误差仅说明本例与人工标注较一致，不代表已完成临床验证。"],
        "本例是有意挑选的较好案例。人工标注未进入 Agent；不可把这一例当作总体表现。"),
    "typical": ("中位附近的分割与误差抵消", "说明单一 FAC 误差不能替代分割质量评估。", [
        "先比较 ED、ES 预测像素数与人工参考像素数，观察两相位是否同时少分。",
        "当 ED、ES 面积出现相近比例的偏差时，比值和 FAC 仍可能很接近参考结果。"],
        "接近的是本批完成计算病例的 Dice 中位数，不是所有数据或所有临床场景的代表。"),
    "missed": ("QC 未识别的分割偏差", "展示数值自洽和报告流畅不能保证图像证据正确。", [
        "重点看 ES 预测与人工轮廓的空间重合，再看 ES Dice。",
        "本例面积顺序和触边检查未触发，Reviewer 也看不到原图或人工 Dice。",
        "报告仍为 limited；它不是通过临床审核，也不是正常诊断。"],
        "后续应优先增加无人工标注的分割质量评估。原文比值叙述的方向应以结构化 ES/ED 定义为准。"),
    "qc": ("面积顺序异常触发人工复核", "展示确定性 QC 与模型审查如何配合。", [
        "ES 预测面积大于 ED，故按程序公式得到负 FAC；它是需要核查的算法输出。",
        "程序强制 refer，Reviewer 也建议核查。不能将负值解释为负射血分数或疾病诊断。"],
        "QC 的存在不等于完整临床安全评估。是否属于分割错误、时相问题或其他原因，需要回到图像核查。"),
    "failure": ("空 ES mask 与明确失败报告", "说明系统如何保留失败，而不是制造一个测量值。", [
        "ES 二值图为空，叠加图没有绿色区域；这是预测结果，不是文件缺失。",
        "模型 FAC 和面积比为不可用，不能用空 mask 算出百分之百收缩。",
        "人工参考仍能用于独立评估，但没有被拿来替换模型预测。"],
        ("原文“ED相位掩码有效”至多说明非空，不代表准确；原文将 RWMA 不可用归因于输入缺失也不完整，"
         "项目实际上没有部署经验证的 RWMA 分类器。保留原文并提出这些修订意见，不能静默润色为已验证结论。")),
}


def select_cases(rows):
    complete = [r for r in rows if r["status"] == "completed"]
    clean = [r for r in complete if r["verdict"] == "limited" and not r.get("qc_flags")]
    median = statistics.median(r["dice"] for r in complete)
    groups = [
        ("good", sorted([r for r in clean if r["fac_absolute_error_pp"] <= 5],
                        key=lambda r: (-r["dice"], r["patient"]))),
        ("typical", sorted(clean, key=lambda r: (abs(r["dice"] - median), r["patient"]))),
        ("missed", sorted(clean, key=lambda r: (r["dice"], r["patient"]))),
        ("qc", sorted([r for r in complete if r["verdict"] == "refer" and r.get("qc_flags")],
                      key=lambda r: (r["dice"], r["patient"]))),
        ("failure", sorted([r for r in rows if r["status"] == "failed_reported"],
                           key=lambda r: r["patient"])),
    ]
    chosen, used = [], set()
    for category, candidates in groups:
        available = [r for r in candidates if r["patient"] not in used]
        if not available:
            raise ValueError(f"No distinct case for category {category}")
        chosen.append((category, available[0]))
        used.add(available[0]["patient"])
    return median, chosen


def verify_tree(root):
    checks = json.loads((root / "checksums.json").read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and p != root / "checksums.json"}
    if checks.keys() != actual or any(digest(root / p) != sha for p, sha in checks.items()):
        raise ValueError("Source results failed integrity check")


def edge(mask):
    p = np.pad(mask, 1, constant_values=False)
    interior = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~interior


def export_case(source, data, output, category, row):
    patient = row["patient"]
    original = source / patient
    dest = output / "cases" / patient
    dest.mkdir(parents=True)
    report = json.loads((original / "report.json").read_text(encoding="utf-8"))
    evaluation = json.loads((original / "independent_evaluation.json").read_text())
    references, predictions, hashes = {}, {}, {}
    for phase in ("ED", "ES"):
        image = original / "original" / f"{phase}.png"
        pred_path = original / "segmentation" / f"{phase}_mask.png"
        reference_path = data / "converted" / f"{patient}-4CH" / f"{phase.lower()}_mask.png"
        converted_image = reference_path.with_name(f"{phase.lower()}.png")
        if digest(image) != digest(converted_image):
            raise ValueError("Original/converted image mismatch")
        pred = np.asarray(Image.open(pred_path)) > 0
        ref = np.asarray(Image.open(reference_path)) == 1
        measured = segmentation_metrics(pred, ref)
        expected = evaluation["segmentation"][phase.lower()]
        if any(not np.isclose(measured[k], expected[k], rtol=0, atol=1e-12) for k in measured):
            raise ValueError("Reference no longer matches archived independent evaluation")
        shutil.copy2(image, dest / f"{phase}_original.png")
        shutil.copy2(pred_path, dest / f"{phase}_mask.png")
        shutil.copy2(original / "segmentation" / f"{phase}_overlay.png",
                     dest / f"{phase}_prediction.png")
        Image.fromarray(ref.astype(np.uint8) * 255).save(dest / f"{phase}_reference.png")
        rgb = np.asarray(Image.open(image).convert("RGB")).copy()
        rgb[edge(ref)] = [255, 165, 0]
        rgb[edge(pred)] = [0, 220, 255]
        Image.fromarray(rgb).save(dest / f"{phase}_comparison.png")
        references[phase], predictions[phase] = int(ref.sum()), int(pred.sum())
        hashes[phase] = {"image": digest(image), "prediction": digest(pred_path),
                         "reference": digest(reference_path)}
    for name in ("report.md", "report.json", "planner.json", "analyst.json", "reviewer.json",
                 "agent_trace.json", "independent_evaluation.json"):
        shutil.copy2(original / name, dest / name)
    roles = {}
    for role in ("planner", "analyst", "reviewer"):
        raw = json.loads((original / f"{role}.json").read_text(encoding="utf-8"))
        content = json.loads(raw["response"])
        roles[role] = content["reason"] if role == "planner" else content["narrative"]
    evidence = report["report"]["evidence"]
    title, purpose, points, note = NOTES[category]
    result = {"patient": patient, "category": category, "title": title, "purpose": purpose,
              "talking_points": points, "editor_note": note, "dice": row["dice"],
              "iou": row["iou"], "fac_error": row["fac_absolute_error_pp"],
              "fac": None if row["status"] == "failed_reported" else float(evidence["fac_percent"]),
              "ratio": None if row["status"] == "failed_reported" else float(evidence["area_ratio"]),
              "reference_fac": 100 * (references["ED"] - references["ES"]) / references["ED"],
              "pred_ed": predictions["ED"], "pred_es": predictions["ES"],
              "ref_ed": references["ED"], "ref_es": references["ES"],
              "ed_dice": evaluation["segmentation"]["ed"]["dice"],
              "es_dice": evaluation["segmentation"]["es"]["dice"],
              "verdict": report["effective_verdict"], "rwma_status": evidence["rwma_status"],
              "qc_flags": report["qc_flags"], "narrative": report["report"]["narrative"],
              "concerns": report["report"]["concerns"], "limitations": report["limitations"],
              "roles": roles, "source_hashes": hashes}
    save_json(dest / "demo_case.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--assets", type=Path,
                        default=Path(__file__).resolve().parents[1] / "demo/clinical_review")
    args = parser.parse_args()
    source, data, output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    if source == output or source in output.parents or data == output or data in output.parents:
        raise ValueError("Demonstration output must not modify source data or results")
    if not args.guide.is_file():
        raise ValueError("Word guide is required")
    verify_tree(source)
    source_hash = digest(source / "checksums.json")
    rows = json.loads((source / "results.json").read_text())
    median, selected = select_cases(rows)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("index.html", "style.css", "app.js"):
        shutil.copy2(args.assets / name, output / name)
    shutil.copy2(args.guide, output / "Cardiac-Agent_医工演示讲解稿.docx")
    cases = [export_case(source, data, output, cat, row) for cat, row in selected]
    summary = json.loads((source / "summary.json").read_text())
    payload = {"summary": summary, "cases": cases}
    (output / "cases.js").write_text("window.CARDIAC_DEMO = " +
        json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c") + ";\n",
        encoding="utf-8")
    save_json(output / "selection.json", {
        "source": str(source), "source_checksums_sha256": source_hash,
        "cohort_size": len(rows), "completed_case_median_dice": median,
        "rules": ["Highest Dice among limited/QC-clear cases with FAC error <=5 pp (demo rule only)",
                  "Closest to completed-case median Dice", "Lowest Dice without current QC flags",
                  "Lowest Dice among numerical cases referred by QC", "First empty-mask failure"],
        "selected": [{"category": cat, "patient": row["patient"]} for cat, row in selected],
        "not_a_random_or_clinical_representative_sample": True,
        "new_model_or_api_calls": 0, "source_results_modified": False,
        "short_demo": [c["patient"] for c in cases if c["category"] in {"good", "missed", "failure"}]})
    with (output / "selected_cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["patient", "category", "title", "dice",
                                                   "fac", "reference_fac", "fac_error", "verdict"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cases)
    shutil.copy2(source / "index.csv", output / "all_100_cases_index.csv")
    (output / "README.md").write_text(
        "# Cardiac-Agent 医工演示包\n\n打开 index.html 查看五例；Word 讲解稿可独立阅读。"
        "本展示页是历史结果回放，不是实时推理界面。所有资源均为相对路径，不依赖网络、GPU或API。\n\n"
        "完整演示顺序：0020、0004、0042、0039、0033。简版三例：0020、0042、0033。"
        "病例中的 report.md 与角色 JSON 是原始输出；editor_note 是另外撰写的讲解旁注。\n\n"
        "人工标注仅用于独立评估和此处可视化，未输入 Agent。空 mask 如实保留。"
        "不把 limited 解释为正常、不把 refer 解释为阳性诊断、不把 FAC 解释为 LVEF。\n\n"
        "本包含真实公开数据集图像，仅留服务器，不加入 GitHub。整个目录复制或解压后可离线展示。\n",
        encoding="utf-8")
    verify_tree(source)
    if digest(source / "checksums.json") != source_hash:
        raise ValueError("Source checksum manifest changed")
    save_json(output / "checksums.json", {str(p.relative_to(output)): digest(p)
              for p in output.rglob("*") if p.is_file() and p != output / "checksums.json"})
    print(json.dumps({"output": str(output), "selected": [c["patient"] for c in cases],
                      "source_unchanged": True, "new_model_or_api_calls": 0}))


if __name__ == "__main__":
    main()
