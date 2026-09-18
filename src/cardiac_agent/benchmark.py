"""Paired manual/model evaluation with explicit failed-study accounting."""
import json
from pathlib import Path

import numpy as np

from .cli import load_study
from .evaluation import segmentation_metrics
from .pipeline import run, save_json
from .roi import read_image, read_mask


def evaluate_studies(study_paths, output: Path, segmenter, provenance=None, client=None,
                     deformation="intensity_difference"):
    paths = [Path(p).resolve() for p in study_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide nonempty, unique study paths")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in paths:
        row = {"manifest": str(path), "status": "failed"}
        try:
            study = load_study(path)
            row.update(study_id=study.study_id, view=study.view)
            if study.ed_mask is None or study.mask_source != "manual":
                raise ValueError("Evaluation requires independent manual ED/ES masks")
            if study.view != "A4C":
                raise ValueError("This EchoNet pilot requires A4C studies")
            reference = run(study, output / "manual", deformation=deformation)
            predicted_study = study.model_copy(update={
                "ed_mask": None, "es_mask": None, "mask_source": "predicted"})
            predicted = run(predicted_study, output / "predicted", client=client,
                            segmenter=segmenter, model_provenance=provenance,
                            deformation=deformation)
            metrics = {}
            for phase in ("ed", "es"):
                shape = read_image(getattr(study, f"{phase}_image")).shape[:2]
                gold = read_mask(getattr(study, f"{phase}_mask"), shape, study.foreground_label)
                mask = read_mask(predicted / f"{phase}_mask.png", shape, 1)
                metrics[phase] = segmentation_metrics(mask, gold)
            gold_m = json.loads((reference / "measurements.json").read_text())
            pred_m = json.loads((predicted / "measurements.json").read_text())
            explanation = json.loads((predicted / "explanation.json").read_text())
            row.update(status="completed", manual_run=str(reference), predicted_run=str(predicted),
                       segmentation=metrics, manual_fac=gold_m["fac_percent"],
                       predicted_fac=pred_m["fac_percent"],
                       fac_absolute_error_pp=abs(pred_m["fac_percent"] - gold_m["fac_percent"]),
                       qc_flags=pred_m["qc_flags"], explanation_mode=explanation["mode"],
                       api_status=explanation["api_status"])
        except (ValueError, OSError, RuntimeError) as exc:
            row["error_type"] = type(exc).__name__
        rows.append(row)
        save_json(output / "cases.json", rows)
    successful = [r for r in rows if r["status"] == "completed"]
    frames = [r["segmentation"][p] for r in successful for p in ("ed", "es")]
    summary = {
        "requested": len(paths), "completed": len(successful),
        "failed": len(paths) - len(successful), "failure_rate": 1 - len(successful) / len(paths),
        "frames_evaluated": len(frames),
        "mean_dice": float(np.mean([r["dice"] for r in frames])) if frames else None,
        "mean_iou": float(np.mean([r["iou"] for r in frames])) if frames else None,
        "mean_absolute_area_error_px": float(np.mean([abs(r["area_error_px"]) for r in frames]))
            if frames else None,
        "fac_mae_pp": float(np.mean([r["fac_absolute_error_pp"] for r in successful]))
            if successful else None,
        "qwen_successes": sum(r["api_status"] == "success" for r in successful),
        "metric_scope": "Successful studies only; failure count reported separately.",
        "interpretation": "Small cross-dataset A4C pilot; not a clinical performance estimate.",
        "segmentation": provenance or {},
        "deformation_method": deformation,
    }
    save_json(output / "summary.json", summary)
    return summary
