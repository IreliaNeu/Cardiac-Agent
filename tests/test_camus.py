import json

import numpy as np
import pytest
from PIL import Image

from cardiac_agent.camus import export_pair
from cardiac_agent.cli import load_study
from cardiac_agent.pipeline import run


def test_manual_label_import_keeps_multiclass_and_spacing(tmp_path):
    ed = np.arange(64, dtype=np.uint8).reshape(8, 8)
    mask = np.zeros((8, 8), np.uint8)
    mask[1:7, 1:7] = 2
    mask[2:6, 2:6] = 1
    smaller = mask.copy()
    smaller[2] = 2
    path = export_pair("patient0001", "4CH", ed, ed, mask, smaller,
                       (.3, .15), tmp_path / "case")
    study = load_study(path)
    assert study.mask_source == "manual" and study.rwma_label is None
    assert study.spacing_mm == (.3, .15)
    assert np.array_equal(np.asarray(Image.open(study.ed_mask)), mask)
    result = run(study, tmp_path / "runs")
    m = json.loads((result / "measurements.json").read_text())
    assert m["area_ed_px"] == 16 and m["area_es_px"] == 12
    assert m["fac_percent"] == 25


def test_joint_normalization_preserves_phase_scale(tmp_path):
    ed = np.array([[0, 50], [100, 200]], dtype=np.uint16)
    es = ed // 2
    mask = np.ones((2, 2), np.uint8)
    path = export_pair("patient0001", "2CH", ed, es, mask, mask,
                       (1, 1), tmp_path / "case")
    study = load_study(path)
    assert np.asarray(Image.open(study.es_image)).max() == 128


def test_unknown_camus_label_rejected(tmp_path):
    ed = np.zeros((2, 2), np.uint8)
    mask = np.full((2, 2), 255, np.uint8)
    with pytest.raises(ValueError, match="CAMUS labels"):
        export_pair("p", "4CH", ed, ed, mask, mask, (1, 1), tmp_path / "case")


@pytest.mark.parametrize("extension", [".nii.gz", ".mhd"])
def test_medical_file_import_roundtrip(tmp_path, extension):
    sitk = pytest.importorskip("SimpleITK")
    from cardiac_agent.camus import import_camus

    patient = tmp_path / "patient0001"
    patient.mkdir()
    array = np.arange(64, dtype=np.uint8).reshape(1, 8, 8)
    mask = np.zeros_like(array)
    mask[:, 2:6, 2:6] = 1
    for suffix in ("ED", "ES", "ED_gt", "ES_gt"):
        image = sitk.GetImageFromArray(mask if suffix.endswith("gt") else array)
        image.SetSpacing((.3, .15, 1.0))
        sitk.WriteImage(image, str(patient / f"patient0001_4CH_{suffix}{extension}"))
    study = load_study(import_camus(patient, "4CH", tmp_path / "converted"))
    np.testing.assert_allclose(study.spacing_mm, (.3, .15))
    np.testing.assert_array_equal(np.asarray(Image.open(study.ed_mask)), mask[0])


def test_medical_grid_mismatch_rejected(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    from cardiac_agent.camus import import_camus

    patient = tmp_path / "patient0001"
    patient.mkdir()
    for suffix in ("ED", "ES", "ED_gt", "ES_gt"):
        image = sitk.GetImageFromArray(np.ones((1, 8, 8), np.uint8))
        image.SetSpacing((.6 if suffix == "ES_gt" else .3, .3, 1.0))
        sitk.WriteImage(image, str(patient / f"patient0001_4CH_{suffix}.nii.gz"))
    with pytest.raises(ValueError, match="physical grids"):
        import_camus(patient, "4CH", tmp_path / "converted")
