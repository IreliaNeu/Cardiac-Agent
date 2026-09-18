import numpy as np

from .schemas import MedicalKnowledgePacket


def segmentation_metrics(prediction, reference):
    if prediction.shape != reference.shape:
        raise ValueError("Segmentation evaluation requires matched image grids")
    a, b = prediction.astype(bool), reference.astype(bool)
    if not b.any():
        raise ValueError("Reference LV mask is empty")
    inter = int(np.logical_and(a, b).sum())
    return {"dice": 2 * inter / (int(a.sum()) + int(b.sum())),
            "iou": inter / int(np.logical_or(a, b).sum()),
            "area_error_px": int(a.sum()) - int(b.sum())}


def evaluate_qa(packet: MedicalKnowledgePacket, gold_rwma: int | None = None):
    m, y = packet.measurements, packet.rwma
    return {
        "questions": [
            {"id": "area_decreased", "answer": m.area_es_px < m.area_ed_px,
             "evidence": ["area_ed_px", "area_es_px"]},
            {"id": "rwma_present", "answer": bool(y.label) if y.label is not None else None,
             "evidence": ["rwma"], "status": y.status},
            {"id": "can_interpret_fac_as_lvef", "answer": False,
             "evidence": ["limitations"]},
        ],
        "arithmetic_consistent": abs(m.fac_percent - 100 * (1 - m.area_ratio)) < 1e-8,
        "rwma_accuracy": int(y.label == gold_rwma)
            if gold_rwma is not None and y.status == "predicted" else None,
        "rwma_evaluable": int(gold_rwma is not None and y.status == "predicted"),
        "note": "Consistency is not diagnostic accuracy; missing predictions are not negatives.",
    }
