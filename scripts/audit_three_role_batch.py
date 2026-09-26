"""Read-only batch acceptance. Does not invoke models or modify sealed results."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from cardiac_agent.discussion import Draft, Review, validate_report
from cardiac_agent.explanation import evidence_claims
from cardiac_agent.pipeline import digest
from cardiac_agent.schemas import MedicalKnowledgePacket


def audit(root, count):
    checks = json.loads((root / "checksums.json").read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and p != root / "checksums.json"}
    assert actual == checks.keys(), "Unexpected/missing artifacts"
    assert all(digest(root / name) == sha for name, sha in checks.items()), "Hash mismatch"
    summary = json.loads((root / "summary.json").read_text())
    results = json.loads((root / "results.json").read_text())
    assert len(results) == count == summary["requested"]
    assert len({r["patient"] for r in results}) == count
    completed = [r for r in results if r["status"] == "completed"]
    assert len(completed) == summary["completed"]
    assert len(completed) + summary["failed"] == count
    roles, attempts, total_tokens = Counter(), Counter(), 0
    concerns, overrides, disagreeing = Counter(), 0, 0
    for row in completed:
        directory = root / row["patient"]
        execution = Path(json.loads((directory / "execution.json").read_text())["directory"])
        packet = MedicalKnowledgePacket.model_validate_json(
            (execution / "knowledge.json").read_text())
        facts = {c.evidence_id: c.value for c in evidence_claims(packet)}
        request = json.loads((execution / "request.json").read_text())
        assert request["ed_mask"] is None and request["es_mask"] is None
        assert request["rwma_label"] is None
        trace = json.loads((directory / "agent_trace.json").read_text(encoding="utf-8"))
        assert [r["role"] for r in trace] == ["planner", "analyst", "reviewer"]
        for role in trace:
            assert role["round"] == 1 and role["status"] == "received"
            roles[role["role"]] += 1
            attempts[role["telemetry"]["attempts"]] += 1
            total_tokens += role["telemetry"].get("usage", {}).get("total_tokens", 0)
            context = json.dumps(role["request"]["context"])
            for forbidden in (row["patient"], "reference_fac", "reference_mask", '"dice"'):
                assert forbidden not in context, "Reference/identifier leak"
        draft = Draft.model_validate_json(trace[1]["response"])
        review = Review.model_validate_json(trace[2]["response"])
        validate_report(draft, facts)
        validate_report(review, facts)
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        assert report["report"] == review.model_dump()
        assert report["rounds"] == 1 and report["role_turns"] == 3
        overrides += int(report["qc_override"])
        disagreeing += int(report["effective_verdict"] != review.verdict)
        concerns.update(report["qc_flags"])
        for phase in ("ED", "ES"):
            original = np.asarray(Image.open(directory / "original" / f"{phase}.png"))
            mask = np.asarray(Image.open(directory / "segmentation" / f"{phase}_mask.png"))
            overlay = np.asarray(Image.open(directory / "segmentation" / f"{phase}_overlay.png"))
            assert set(np.unique(mask)) == {0, 255}
            assert original.shape[:2] == mask.shape == overlay.shape[:2]
            assert original.var() > 0 and overlay.var() > 0
            assert digest(directory / "original" / f"{phase}.png") == digest(
                Path(request[phase.lower() + "_image"]))
    return {"requested": count, "completed": len(completed), "passed": len(completed) == count,
            "verified_files": len(checks), "role_calls": dict(roles),
            "http_attempt_distribution": dict(attempts), "reported_total_tokens": total_tokens,
            "qc_flags": dict(concerns), "qc_forced_referrals": overrides,
            "reviewer_disagreements_overridden": disagreeing,
            "original_images": len(completed) * 2, "binary_masks": len(completed) * 2,
            "overlays": len(completed) * 2, "reports": len(completed),
            "clinical_correctness_checked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    result = audit(args.directory.resolve(), args.count)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
