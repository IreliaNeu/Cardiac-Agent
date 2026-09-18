from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Study(Contract):
    study_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    view: str = Field(min_length=1)
    ed_image: Path
    es_image: Path
    ed_mask: Path | None = None
    es_mask: Path | None = None
    mask_source: Literal["predicted", "manual", "synthetic"] = "predicted"
    foreground_label: int = Field(default=1, ge=1, le=255)
    spacing_mm: tuple[float, float] | None = None
    phase_source: Literal["manual", "dataset", "model", "synthetic"] = "dataset"
    synthetic: bool = False
    rwma_label: int | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_inputs(self):
        if (self.ed_mask is None) != (self.es_mask is None):
            raise ValueError("ED and ES masks must be supplied together")
        if self.spacing_mm and any(x <= 0 for x in self.spacing_mm):
            raise ValueError("Pixel spacing must be positive")
        if self.mask_source == "synthetic" and not self.synthetic:
            raise ValueError("Synthetic masks require synthetic=true")
        return self


class Measurements(Contract):
    area_ed_px: int = Field(gt=0)
    area_es_px: int = Field(gt=0)
    area_ed_mm2: float | None = None
    area_es_mm2: float | None = None
    fac_percent: float
    area_ratio: float
    deformation_method: Literal["none", "intensity_difference", "farneback"]
    deformation_mean: float | None = None
    deformation_var: float | None = None
    deformation_unit: str | None = None
    qc_flags: list[str] = Field(default_factory=list)


class Prediction(Contract):
    status: Literal["unavailable", "abstained", "predicted"]
    probability: float | None = Field(default=None, ge=0, le=1)
    label: int | None = Field(default=None, ge=0, le=1)
    reason: str
    model_sha256: str | None = None
    calibrated: bool = False

    @model_validator(mode="after")
    def consistent_status(self):
        if self.status == "predicted" and (self.label is None or self.probability is None):
            raise ValueError("A prediction requires both class and probability")
        if self.status != "predicted" and (self.label is not None or self.probability is not None):
            raise ValueError("Unavailable or abstained results cannot contain predictions")
        return self


class MedicalKnowledgePacket(Contract):
    schema_version: str = "1.0"
    study_id: str
    view: str
    synthetic: bool
    phase_source: str
    mask_source: str
    measurements: Measurements
    rwma: Prediction
    artifacts: dict[str, dict[str, str]]
    limitations: list[str] = Field(default_factory=lambda: [
        "FAC is a 2D cavity area-change surrogate, not LVEF.",
        "Two frames do not capture full-cycle regional wall motion.",
        "Cavity optical flow or intensity difference is not myocardial strain.",
    ])


class Claim(Contract):
    evidence_id: str
    value: str


class Explanation(Contract):
    mode: Literal["template", "qwen", "template_fallback"]
    text: str
    claims: list[Claim]
    verified: bool
    api_status: str = "not_requested"
