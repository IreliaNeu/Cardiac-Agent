import numpy as np

from .schemas import Measurements


def measure(ed, es, ed_mask, es_mask, spacing=None, deformation="intensity_difference"):
    a_ed, a_es = int(ed_mask.sum()), int(es_mask.sum())
    if not a_ed or not a_es:
        raise ValueError("Cannot calculate area ratio from empty masks")
    flags = []
    if a_es >= a_ed:
        flags.append("non_decreasing_cavity_area_check_phase_or_segmentation")
    if any(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
           for m in (ed_mask, es_mask)):
        flags.append("mask_touches_image_border")
    dmap = None
    unit = None
    if deformation == "intensity_difference":
        dmap = np.abs(es.astype(np.float32).mean(2) - ed.astype(np.float32).mean(2)) / 255
        unit = "normalized_intensity_difference"
        flags.append("appearance_difference_is_not_physical_deformation")
    elif deformation == "farneback":
        import cv2
        cv2.setNumThreads(1)
        gray_ed = cv2.cvtColor(ed, cv2.COLOR_RGB2GRAY)
        gray_es = cv2.cvtColor(es, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray_ed, gray_es, None, .5, 3, 15, 3, 5, 1.2, 0)
        dmap = np.linalg.norm(flow, axis=2)
        unit = "pixels_per_ED_ES_pair"
        flags.append("sparse_cavity_flow_not_myocardial_strain")
    elif deformation != "none":
        raise ValueError("Unknown deformation method")
    values = dmap[ed_mask] if dmap is not None else None
    scale = spacing[0] * spacing[1] if spacing else None
    metrics = Measurements(
        area_ed_px=a_ed, area_es_px=a_es,
        area_ed_mm2=a_ed * scale if scale else None,
        area_es_mm2=a_es * scale if scale else None,
        fac_percent=100 * (a_ed - a_es) / a_ed, area_ratio=a_es / a_ed,
        deformation_method=deformation,
        deformation_mean=float(values.mean()) if values is not None else None,
        deformation_var=float(values.var()) if values is not None else None,
        deformation_unit=unit, qc_flags=flags)
    return metrics, dmap
