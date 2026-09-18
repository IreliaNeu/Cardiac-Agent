from typing import Protocol

from .schemas import Measurements, Prediction


class RWMAPredictor(Protocol):
    def predict(self, features: Measurements) -> Prediction: ...


class UnavailableRWMA:
    def predict(self, features: Measurements) -> Prediction:
        return Prediction(status="unavailable", reason="No trained RWMA checkpoint or labels yet")


def predict_rwma(features: Measurements, predictor: RWMAPredictor) -> Prediction:
    if any(flag.startswith(("non_decreasing", "mask_touches")) for flag in features.qc_flags):
        return Prediction(status="abstained", reason="Input or segmentation QC requires review")
    return predictor.predict(features)
