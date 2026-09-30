import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "clinical_demo", Path(__file__).parents[1] / "scripts/build_clinical_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def row(patient, dice, verdict="limited", status="completed", flags=None):
    return {"patient": patient, "dice": dice, "verdict": verdict,
            "status": status, "qc_flags": flags or [], "fac_absolute_error_pp": 1}


def test_stratified_selection_is_order_independent():
    rows = [row("good", .99), row("middle", .8), row("upper", .85), row("missed", .2),
            row("referral", .01, "refer", flags=["non_decreasing"]),
            row("failure", .3, "refer", "failed_reported")]
    first = demo.select_cases(rows)
    assert first == demo.select_cases(list(reversed(rows)))
    assert [r["patient"] for _, r in first[1]] == [
        "good", "middle", "missed", "referral", "failure"]
    assert len({r["patient"] for _, r in first[1]}) == 5


def test_missing_stratum_does_not_silently_substitute_success():
    with pytest.raises(ValueError, match="category qc"):
        demo.select_cases([row("a", .99), row("b", .8), row("c", .2)])


def test_edges_handle_empty_and_border_masks():
    assert not demo.edge(np.zeros((4, 4), dtype=bool)).any()
    full = demo.edge(np.ones((4, 4), dtype=bool))
    assert full.sum() == 12
    assert not full[1:3, 1:3].any()


def test_integrity_checks_inventory_and_contents(tmp_path):
    item = tmp_path / "item.txt"
    item.write_text("sealed")
    demo.save_json(tmp_path / "checksums.json", {"item.txt": demo.digest(item)})
    demo.verify_tree(tmp_path)
    item.write_text("modified")
    with pytest.raises(ValueError, match="integrity"):
        demo.verify_tree(tmp_path)
    item.write_text("sealed")
    (tmp_path / "extra.txt").write_text("extra")
    with pytest.raises(ValueError, match="integrity"):
        demo.verify_tree(tmp_path)
