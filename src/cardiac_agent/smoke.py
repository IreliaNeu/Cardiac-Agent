"""One-command acceptance test for synthetic or existing manual-mask studies."""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from cardiac_agent.cli import load_study, make_fixture
from cardiac_agent.explanation import QwenClient
from cardiac_agent.pipeline import digest, run, save_json


def smoke(output: Path, study_path=None, client=None, deformation="intensity_difference"):
    output.mkdir(parents=True, exist_ok=False)
    try:
        study_path = study_path or make_fixture(output / "input")
        study = load_study(study_path)
        directory = run(study, output / "runs", client=client, deformation=deformation)
        checksums = json.loads((directory / "checksums.json").read_text())
        for name, expected in checksums.items():
            if digest(directory / name) != expected:
                raise ValueError("Artifact checksum mismatch")
        packet = json.loads((directory / "knowledge.json").read_text())
        for ref in packet["artifacts"].values():
            if digest(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("Evidence checksum mismatch")
        explanation = json.loads((directory / "explanation.json").read_text())
        evaluation = json.loads((directory / "evaluation.json").read_text())
        status = json.loads((directory / "run.json").read_text())
        passed = (status["status"] == "completed" and evaluation["arithmetic_consistent"]
                  and explanation["verified"]
                  and (client is None or explanation["api_status"] == "success"))
        summary = {"passed": passed, "run": str(directory.resolve()),
                   "synthetic": study.synthetic, "deformation": deformation,
                   "rwma_status": packet["rwma"]["status"],
                   "explanation_mode": explanation["mode"],
                   "api_status": explanation["api_status"],
                   "stages": status["stages"], "verified_artifacts": len(checksums),
                   "note": "Engineering acceptance only; RWMA can remain unavailable."}
    except (ValueError, OSError, RuntimeError) as exc:
        save_json(output / "acceptance.json", {"passed": False, "error_type": type(exc).__name__})
        raise
    save_json(output / "acceptance.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--env", type=Path)
    parser.add_argument("--deformation", choices=["none", "intensity_difference", "farneback"],
                        default="intensity_difference")
    args = parser.parse_args()
    if args.env:
        if not args.env.is_file():
            parser.error("Environment file does not exist")
        load_dotenv(args.env)
    result = smoke(args.output, args.study, QwenClient() if args.qwen else None, args.deformation)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
