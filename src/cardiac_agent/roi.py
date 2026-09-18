from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .schemas import Study


class Segmenter(Protocol):
    def segment(self, image: np.ndarray) -> np.ndarray: ...


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.width * image.height > 4_000_000:
            raise ValueError("Image exceeds the CPU-stage limit of 4 million pixels")
        return np.asarray(image.convert("RGB"))


def read_mask(path: Path, shape: tuple[int, int], label: int) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2 or mask.shape != shape:
        raise ValueError("Mask must be single-channel and match its image dimensions")
    if not np.issubdtype(mask.dtype, np.integer) and mask.dtype != bool:
        raise ValueError("Mask must contain integer labels")
    selected = mask == label
    if not selected.any():
        raise ValueError("Empty LV mask: check foreground_label and segmentation")
    return selected


def roi_stage(study: Study, segmenter: Segmenter | None = None):
    ed, es = read_image(study.ed_image), read_image(study.es_image)
    if ed.shape != es.shape:
        raise ValueError("ED and ES must share image dimensions and pixel calibration")
    if study.ed_mask is not None:
        masks = (read_mask(study.ed_mask, ed.shape[:2], study.foreground_label),
                 read_mask(study.es_mask, es.shape[:2], study.foreground_label))
    elif segmenter is not None:
        masks = (segmenter.segment(ed), segmenter.segment(es))
    else:
        raise ValueError("Provide precomputed ED/ES masks or a configured segmentation backend")
    for mask in masks:
        if mask.shape != ed.shape[:2] or mask.dtype != bool or not mask.any():
            raise ValueError("Segmenter must return nonempty boolean masks on the source image grid")
    return ed, es, *masks


class EchoNetSegmenter:
    """Lazy adapter for the official one-class DeepLabV3 checkpoint.

    Training normalization is required explicitly; never assume ImageNet statistics.
    """

    def __init__(self, checkpoint: Path, mean: list[float], std: list[float],
                 device: str = "cpu"):
        if len(mean) != 3 or len(std) != 3 or not np.isfinite(mean + std).all():
            raise ValueError("Provide finite RGB training mean/std on the 0-255 scale")
        if any(x <= 0 for x in std):
            raise ValueError("Standard deviations must be positive")
        import torch
        from torchvision.models.segmentation import deeplabv3_resnet50
        self.torch = torch
        self.device = device
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None, None]
        self.model = deeplabv3_resnet50(weights=None, weights_backbone=None,
                                      num_classes=1, aux_loss=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        self.model.to(device).eval()

    def segment(self, image: np.ndarray) -> np.ndarray:
        frame = Image.fromarray(image).resize((112, 112), Image.Resampling.BILINEAR)
        array = np.asarray(frame, dtype=np.float32).transpose(2, 0, 1)
        tensor = self.torch.from_numpy((array - self.mean) / self.std)[None].to(self.device)
        with self.torch.inference_mode():
            logits = self.model(tensor)["out"]
            logits = self.torch.nn.functional.interpolate(
                logits, size=image.shape[:2], mode="bilinear", align_corners=False)
        return logits[0, 0].cpu().numpy() > 0
