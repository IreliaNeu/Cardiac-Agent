import argparse
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

from .explanation import QwenClient
from .pipeline import digest, run, save_json
from .roi import EchoNetSegmenter
from .schemas import Study


def make_fixture(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    yy, xx = np.mgrid[:64, :64]
    for phase, radius in (("ed", 20), ("es", 15)):
        mask = ((xx - 32) ** 2 + (yy - 32) ** 2 <= radius ** 2)
        image = np.full((64, 64, 3), 40, dtype=np.uint8)
        image[mask] = 120
        Image.fromarray(image).save(directory / f"{phase}.png")
        Image.fromarray(mask.astype(np.uint8)).save(directory / f"{phase}_mask.png")
    study = Study(study_id="synthetic-001", patient_id="synthetic-patient", view="synthetic",
                  ed_image=Path("ed.png"), es_image=Path("es.png"),
                  ed_mask=Path("ed_mask.png"), es_mask=Path("es_mask.png"),
                  mask_source="synthetic", phase_source="synthetic", synthetic=True,
                  spacing_mm=(0.5, 0.5))
    path = directory / "study.json"
    save_json(path, study.model_dump(mode="json"))
    return path


def load_study(path: Path) -> Study:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("ed_image", "es_image", "ed_mask", "es_mask"):
        if data.get(key):
            p = Path(data[key])
            data[key] = p if p.is_absolute() else (path.parent / p).resolve()
    return Study.model_validate(data)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture")
    fixture.add_argument("directory", type=Path)
    importer = sub.add_parser("import-camus")
    importer.add_argument("patient_dir", type=Path)
    importer.add_argument("--view", choices=["2CH", "4CH"], default="4CH")
    importer.add_argument("--output", type=Path, required=True)
    runner = sub.add_parser("run")
    runner.add_argument("study", type=Path)
    runner.add_argument("--output", type=Path, default=Path("runs"))
    runner.add_argument("--env", type=Path)
    runner.add_argument("--qwen", action="store_true")
    runner.add_argument("--deformation", choices=["none", "intensity_difference", "farneback"],
                        default="intensity_difference")
    runner.add_argument("--segmentation-config", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "fixture":
        print(make_fixture(args.directory))
    elif args.command == "import-camus":
        from .camus import import_camus
        print(import_camus(args.patient_dir, args.view, args.output))
    elif args.command == "verify":
        checksums = json.loads((args.directory / "checksums.json").read_text())
        if any(digest(args.directory / name) != value for name, value in checksums.items()):
            raise ValueError("Artifact checksum mismatch")
        packet = json.loads((args.directory / "knowledge.json").read_text())
        for ref in packet["artifacts"].values():
            if digest(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("Input or evidence checksum mismatch")
        print("Artifacts and source evidence verified")
    else:
        if args.env:
            if not args.env.is_file():
                raise ValueError("Requested environment file does not exist")
            load_dotenv(args.env, override=False)
        else:
            load_dotenv(Path.cwd() / ".env", override=False)
        segmenter, provenance = None, None
        study = load_study(args.study)
        if args.segmentation_config:
            if study.ed_mask:
                raise ValueError("Choose masks or model segmentation, not both")
            config = json.loads(args.segmentation_config.read_text())
            checkpoint = Path(config["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = args.segmentation_config.parent / checkpoint
            segmenter = EchoNetSegmenter(checkpoint, config["mean"], config["std"],
                                        config.get("device", "cpu"))
            provenance = {"backend": "echonet_deeplabv3", "checkpoint_sha256": digest(checkpoint),
                          "config_sha256": digest(args.segmentation_config)}
        print(run(study, args.output, QwenClient() if args.qwen else None,
                  segmenter=segmenter, deformation=args.deformation,
                  model_provenance=provenance))


if __name__ == "__main__":
    main()
