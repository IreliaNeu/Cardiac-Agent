import numpy as np
import pytest

from cardiac_agent.function import measure


def test_farneback_identity_and_units():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    result, motion = measure(image, image, mask, mask, deformation="farneback")
    assert result.deformation_unit == "pixels_per_ED_ES_pair"
    assert result.deformation_mean < .05
    assert motion.shape == mask.shape and np.isfinite(motion).all()
    assert "sparse_cavity_flow_not_myocardial_strain" in result.qc_flags
