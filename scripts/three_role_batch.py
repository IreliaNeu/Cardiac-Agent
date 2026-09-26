"""Download deterministic CAMUS A4C cases and export a one-round discussion batch."""
import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from cardiac_agent.camus import import_camus
from cardiac_agent.cli import load_study
from cardiac_agent.discussion import discuss
from cardiac_agent.evaluation import segmentation_metrics
from cardiac_agent.explanation import QwenClient
from cardiac_agent.pipeline import digest, save_json
from cardiac_agent.roi import EchoNetSegmenter

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
FOLDER = "63fde55f73e9f004868fb7ac"


def get(client, url, params=None):
    for attempt in range(3):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response
        except (httpx.HTTPError, OSError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download_patient(patient, root):
    name = patient["name"]
    if not name.startswith("patient") or not name[7:].isdigit():
        raise ValueError("unsafe_patient_name")
    raw = root / "raw" / name
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "download_manifest.json"
    expected = {f"{name}_4CH_{phase}.nii.gz" for phase in ("ED", "ES", "ED_gt", "ES_gt")}
    expected.update(("Info_4CH.cfg", "MANDATORY_CITATION.md"))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if {m["name"] for m in manifest} == expected and all(
                (raw / m["name"]).exists() and digest(raw / m["name"]) == m["sha256"]
                for m in manifest):
            return name
    manifest = []
    with httpx.Client(timeout=90, follow_redirects=True, trust_env=False) as client:
        items = {x["name"]: x for x in get(client, f"{API}/item", {
            "folderId": patient["_id"], "limit": 100}).json()}
        if not expected <= items.keys():
            raise ValueError("missing_official_files")
        for filename in sorted(expected):
            files = get(client, f"{API}/item/{items[filename]['_id']}/files").json()
            if len(files) != 1:
                raise ValueError("ambiguous_file")
            url = f"{API}/file/{files[0]['_id']}/download"
            content = get(client, url).content
            if len(content) != files[0]["size"]:
                raise ValueError("download_size_mismatch")
            path = raw / filename
            temp = path.with_name(path.name + ".part")
            temp.write_bytes(content)
            temp.replace(path)
            manifest.append({"name": filename, "url": url, "bytes": len(content),
                             "sha256": digest(path)})
    save_json(manifest_path, manifest)
    return name


def verify_case(directory):
    checks = json.loads((directory / "checksums.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*")
              if p.is_file() and p != directory / "checksums.json"}
    if actual != checks.keys() or any(digest(directory / p) != h for p, h in checks.items()):
        raise ValueError("case_checksum_mismatch")


def process(name, args, segmenter, lock, provenance):
    directory = args.output / name
    if (directory / "checksums.json").exists():
        verify_case(directory)
        return json.loads((directory / "result.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    try:
        converted = args.data / "converted" / f"{name}-4CH"
        study_path = converted / "study.json"
        if not study_path.exists():
            study_path = import_camus(args.data / "raw" / name, "4CH", converted)
        study = load_study(study_path)
        execution = discuss(study, directory, QwenClient(), segmenter, lock, provenance)
        metrics, counts = {}, []
        # References are evaluated only after all agent turns have finished.
        for phase in ("ed", "es"):
            pred = np.asarray(Image.open(execution / f"{phase}_mask.png")) > 0
            reference = np.asarray(Image.open(getattr(study, phase + "_mask"))) == 1
            metrics[phase] = segmentation_metrics(pred, reference)
            counts.append(int(reference.sum()))
        measured = json.loads((execution / "measurements.json").read_text())
        fac_reference = 100 * (counts[0] - counts[1]) / counts[0]
        evaluation = {"segmentation": metrics, "reference_fac_percent": fac_reference,
                      "fac_absolute_error_pp": abs(measured["fac_percent"] - fac_reference),
                      "reference_used_by_agents": False}
        save_json(directory / "independent_evaluation.json", evaluation)
        report = json.loads((directory / "report.json").read_text())
        result = {"patient": name, "status": "completed",
                  "dice": float(np.mean([m["dice"] for m in metrics.values()])),
                  "iou": float(np.mean([m["iou"] for m in metrics.values()])),
                  "fac_absolute_error_pp": evaluation["fac_absolute_error_pp"],
                  "verdict": report["effective_verdict"],
                  "motion": measured["deformation_method"],
                  "qc_flags": measured["qc_flags"], "report": f"{name}/report.md"}
        save_json(directory / "result.json", result)
        (directory / "failure.json").unlink(missing_ok=True)
        save_json(directory / "checksums.json", {
            str(p.relative_to(directory)): digest(p) for p in directory.rglob("*")
            if p.is_file() and p != directory / "checksums.json"})
        return result
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        result = {"patient": name, "status": "failed", "error_type": type(exc).__name__}
        save_json(directory / "failure.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/echonet.official.json"))
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.count <= 500 or not 1 <= args.workers <= 8:
        parser.error("count must be 1..500 and workers 1..8")
    args.data, args.output = args.data.resolve(), args.output.resolve()
    args.data.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    load_dotenv(args.env)
    import SimpleITK as sitk
    import torch
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    torch.set_num_threads(2)
    config = json.loads(args.config.read_text())
    checkpoint = (args.config.resolve().parent / config["checkpoint"]).resolve()
    provenance = {**config, "checkpoint_sha256": digest(checkpoint)}
    selection_path = args.output / "selection.json"
    with httpx.Client(timeout=90, trust_env=False) as client:
        patients = get(client, f"{API}/folder", {
            "parentType": "folder", "parentId": FOLDER, "limit": args.count,
            "sort": "name", "sortdir": 1}).json()
    if len(patients) != args.count or len({x["name"] for x in patients}) != args.count:
        raise ValueError("invalid_patient_selection")
    selection = {"api": API, "folder": FOLDER, "patients": patients,
                 "rule": "First N folders sorted by name, A4C ED/ES; no quality selection."}
    if selection_path.exists():
        previous = json.loads(selection_path.read_text())
        if [(x["name"], x["_id"]) for x in previous["patients"]] != [
                (x["name"], x["_id"]) for x in patients]:
            raise ValueError("selection_changed")
    else:
        save_json(selection_path, selection)
    ready, results = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_patient, p, args.data): p["name"] for p in patients}
        for future in as_completed(futures):
            name = futures[future]
            try:
                ready.append(future.result())
                print(f"downloaded {name} ({len(ready)}/{args.count})", flush=True)
            except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError) as exc:
                results.append({"patient": name, "status": "download_failed",
                                "error_type": type(exc).__name__})
                print(f"download failed {name}: {type(exc).__name__}", flush=True)
            save_json(args.output / "download_status.json", {"ready": sorted(ready),
                                                            "failures": results})
    segmenter = EchoNetSegmenter(checkpoint, config["mean"], config["std"], config["device"])
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, name, args, segmenter, lock, provenance)
                   for name in sorted(ready)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            save_json(args.output / "results.json", sorted(results, key=lambda r: r["patient"]))
            print(json.dumps(result, ensure_ascii=False), flush=True)
    completed = [r for r in results if r["status"] == "completed"]
    summary = {"requested": args.count, "completed": len(completed),
               "failed": args.count - len(completed), "rounds_per_case": 1,
               "role_turns_per_completed_case": 3, "model": QwenClient().model,
               "segmentation": provenance, "clinically_validated": False,
               "selection_is_held_out_benchmark": False,
               "mean_dice": float(np.mean([r["dice"] for r in completed])) if completed else None,
               "mean_iou": float(np.mean([r["iou"] for r in completed])) if completed else None,
               "fac_mae_pp": float(np.mean([r["fac_absolute_error_pp"] for r in completed]))
               if completed else None,
               "referred": sum(r["verdict"] == "refer" for r in completed),
               "motion_counts": {m: sum(r["motion"] == m for r in completed)
                                 for m in ("farneback", "none")}}
    save_json(args.output / "summary.json", summary)
    with (args.output / "index.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["patient", "status", "report", "verdict",
                                                   "dice", "iou", "fac_absolute_error_pp"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["patient"]))
    (args.output / "README.md").write_text(
        "# 三角色单轮 CAMUS 结果\n\n每个 patient 目录包含 original 原图、segmentation "
        "预测二值图和叠加图、report.md 模型报告、agent_trace.json 三角色记录，"
        "以及独立评估和校验和。原图为原始 NIfTI 的归一化 PNG，原始文件位于数据目录。\n\n"
        "selection.json 固定病例选择，index.csv 为索引，summary.json 为汇总。"
        "人工标注只用于独立评估，不进入角色提示。每角色一次语义调用；传输重试单独记录。"
        "本批是流程验证，不是临床验证或独立测试集性能声明。\n",
        encoding="utf-8")
    save_json(args.output / "checksums.json", {
        str(p.relative_to(args.output)): digest(p) for p in args.output.rglob("*")
        if p.is_file() and p != args.output / "checksums.json"})
    print("SUMMARY " + json.dumps(summary), flush=True)
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
