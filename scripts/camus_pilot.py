"""Download three official CAMUS cases and audit the manual-mask pipeline."""
import argparse
import json
from pathlib import Path

import httpx
import numpy as np
import SimpleITK as sitk
from dotenv import load_dotenv
from PIL import Image

from cardiac_agent.camus import import_camus
from cardiac_agent.cli import load_study
from cardiac_agent.explanation import QwenClient
from cardiac_agent.pipeline import digest, run, save_json

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
FOLDER = "63fde55f73e9f004868fb7ac"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    if args.env:
        load_dotenv(args.env)
    manifest, results = [], []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        response = client.get(f"{API}/folder", params={
            "parentType": "folder", "parentId": FOLDER,
            "limit": 3, "sort": "name", "sortdir": 1})
        response.raise_for_status()
        patients = response.json()
        if len(patients) != 3:
            raise ValueError("Expected three official patient folders")
        for patient in patients:
            name = patient["name"]
            if not name.startswith("patient") or not name[7:].isdigit():
                raise ValueError("Unexpected patient folder name")
            raw = args.output / "raw" / name
            raw.mkdir(parents=True)
            expected = {f"{name}_{v}_{s}.nii.gz" for v in ("2CH", "4CH")
                        for s in ("ED", "ES", "ED_gt", "ES_gt")}
            expected.update({"Info_2CH.cfg", "Info_4CH.cfg", "MANDATORY_CITATION.md"})
            response = client.get(f"{API}/item", params={"folderId": patient["_id"], "limit": 100})
            response.raise_for_status()
            items = {item["name"]: item for item in response.json()}
            if not expected <= items.keys():
                raise ValueError("Missing official files")
            for filename in sorted(expected):
                item = items[filename]
                response = client.get(f"{API}/item/{item['_id']}/files")
                response.raise_for_status()
                files = response.json()
                if len(files) != 1:
                    raise ValueError("Expected a single file per item")
                url = f"{API}/file/{files[0]['_id']}/download"
                response = client.get(url)
                response.raise_for_status()
                path = raw / filename
                path.write_bytes(response.content)
                if path.stat().st_size != files[0]["size"]:
                    raise ValueError("Download size mismatch")
                manifest.append({"file": str(path.relative_to(args.output)), "url": url,
                                 "item_id": item["_id"], "sha256": digest(path),
                                 "bytes": path.stat().st_size})
                save_json(args.output / "download_manifest.json", manifest)
            print(f"Downloaded {name}", flush=True)
            for view in ("2CH", "4CH"):
                converted = args.output / "converted" / f"{name}-{view}"
                study_path = import_camus(raw, view, converted)
                study = load_study(study_path)
                directory = run(study, args.output / "runs")
                measures = json.loads((directory / "measurements.json").read_text())
                counts = []
                for phase in ("ED", "ES"):
                    original = sitk.ReadImage(str(raw / f"{name}_{view}_{phase}_gt.nii.gz"))
                    array = sitk.GetArrayFromImage(original).squeeze()
                    exported = np.asarray(Image.open(converted / f"{phase.lower()}_mask.png"))
                    np.testing.assert_array_equal(array, exported)
                    counts.append(int((array == 1).sum()))
                expected_fac = 100 * (counts[0] - counts[1]) / counts[0]
                np.testing.assert_equal([measures["area_ed_px"], measures["area_es_px"]], counts)
                np.testing.assert_allclose(measures["fac_percent"], expected_fac, rtol=1e-12)
                pixel_area = float(np.prod(study.spacing_mm))
                np.testing.assert_allclose(
                    [measures["area_ed_mm2"], measures["area_es_mm2"]],
                    np.asarray(counts) * pixel_area, rtol=1e-12)
                checksums = json.loads((directory / "checksums.json").read_text())
                if any(digest(directory / n) != h for n, h in checksums.items()):
                    raise ValueError("Run artifact checksum mismatch")
                result = {"study": study.study_id, "run": str(directory),
                          "original_lv_pixels": counts, "expected_fac_percent": expected_fac,
                          "mask_roundtrip_exact": True, "measurements": measures}
                results.append(result)
                save_json(args.output / "summary.json", results)
                print(json.dumps(result), flush=True)
        if args.env:
            study = load_study(args.output / "converted" / f"{patients[0]['name']}-4CH" / "study.json")
            directory = run(study, args.output / "runs", QwenClient())
            save_json(args.output / "qwen_run.json", {"directory": str(directory)})
            print(f"Qwen run: {directory}", flush=True)


if __name__ == "__main__":
    main()
