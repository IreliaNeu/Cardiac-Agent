"""Single-patient CAMUS importer; SimpleITK is an optional CPU dependency."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .pipeline import save_json
from .schemas import Study


def export_pair(patient_id, view, ed, es, ed_mask, es_mask, spacing, output, source=None):
    arrays = [np.asarray(a).squeeze() for a in (ed, es, ed_mask, es_mask)]
    if any(a.ndim != 2 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Expected matching 2D ED/ES images and labels")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Non-finite medical image values")
    ed, es, ed_mask, es_mask = arrays
    for mask in (ed_mask, es_mask):
        if not np.isin(mask, [0, 1, 2, 3]).all() or not (mask == 1).any():
            raise ValueError("Expected CAMUS labels 0/1/2/3 with LV cavity label 1")
    study = Study(study_id=f"{patient_id}-{view}", patient_id=patient_id, view=f"A{view[0]}C",
                  ed_image=Path("ed.png"), es_image=Path("es.png"),
                  ed_mask=Path("ed_mask.png"), es_mask=Path("es_mask.png"),
                  mask_source="manual", foreground_label=1, spacing_mm=spacing,
                  phase_source="dataset")
    low, high = float(min(ed.min(), es.min())), float(max(ed.max(), es.max()))
    preserve = ed.dtype == np.uint8 and es.dtype == np.uint8
    if not preserve and high == low:
        raise ValueError("Constant ED/ES image pair")
    output.mkdir(parents=True, exist_ok=False)
    for name, array in zip(("ed", "es", "ed_mask", "es_mask"), arrays):
        if name in {"ed", "es"} and not preserve:
            array = np.rint((array.astype(np.float64) - low) / (high - low) * 255)
        Image.fromarray(array.astype(np.uint8)).save(output / f"{name}.png")
    save_json(output / "study.json", study.model_dump(mode="json"))
    save_json(output / "source.json", {
        "dataset": "CAMUS", "source": source or {}, "patient_id": patient_id,
        "normalization": "preserve_uint8" if preserve else "joint_ED_ES_minmax",
        "source_range": [low, high], "spacing_mm": list(spacing),
        "decoded_array_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
        "labels": {"0": "background", "1": "LV cavity", "2": "myocardium", "3": "LA"},
        "rwma_labels": "not_provided", "split_policy": "assign by patient before training",
    })
    return output / "study.json"


def import_camus(patient_dir: Path, view: str, output: Path):
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    if view not in {"2CH", "4CH"}:
        raise ValueError("CAMUS view must be 2CH or 4CH")
    patient = patient_dir.name
    files, images = [], []
    for suffix in ("ED", "ES", "ED_gt", "ES_gt"):
        stem = f"{patient}_{view}_{suffix}"
        candidates = [patient_dir / (stem + ext) for ext in (".mhd", ".nii.gz")]
        matches = [p for p in candidates if p.is_file()]
        if len(matches) != 1:
            raise ValueError(f"Expected one file for {stem} (.mhd or .nii.gz)")
        files.append(str(matches[0].resolve()))
        images.append(sitk.ReadImage(str(matches[0])))
    first = images[0]
    for image in images[1:]:
        if image.GetSize() != first.GetSize() or any(
            not np.allclose(getattr(image, method)(), getattr(first, method)())
            for method in ("GetSpacing", "GetOrigin", "GetDirection")
        ):
            raise ValueError("CAMUS pair has inconsistent physical grids")
    return export_pair(patient, view, *(sitk.GetArrayFromImage(i) for i in images),
                       spacing=first.GetSpacing()[:2], output=output,
                       source={"files": files})
