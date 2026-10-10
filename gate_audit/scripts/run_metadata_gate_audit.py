#!/usr/bin/env python3
"""POST-HOC explanatory audit of a WoodPhys-DG metadata rule.

This script is NEW REVISION-AUDIT documentation, NOT a historical frozen
model/gate implementation and NOT an independent validation. It has no
outcome/target-label inputs and does not train or re-estimate any model.
All source/target descriptors are curated from manuscript metadata.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SOURCE = "USDA_reference"
INPUT = HERE / "inputs" / "regime_metadata.csv"
OUTPUT = HERE / "outputs" / "five_dimension_decisions.csv"
ALLOWED_COLS = {
    "regime_id", "role", "observation_unit", "specimen_condition",
    "bending_test_regime", "predictor_harmonization",
    "pooling_justification", "evidence_basis",
}

def status_equal(source, target, key):
    a, b = source[key], target[key]
    if not a or not b or a == "unverified" or b == "unverified":
        return "UNVERIFIED", f"insufficient metadata: {a or '?'} vs {b or '?'}"
    return ("PASS" if a == b else "VETO"), f"{a} vs {b}"

def status_documented(target, key):
    descriptor = target[key]
    if descriptor == "documented":
        return "PASS", "harmonization/pooling documented in curated metadata"
    if descriptor == "unsupported":
        return "VETO", "direct scientific pooling unsupported by manuscript metadata"
    return "UNVERIFIED", "not established; cannot be treated as a PASS"

def run():
    with INPUT.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        assert set(reader.fieldnames) == ALLOWED_COLS, "metadata schema changed (possible outcome column)"
        regimes = {row["regime_id"]: row for row in reader}
    source = regimes[SOURCE]
    output = []
    for name, row in regimes.items():
        if name == SOURCE:
            continue
        checks = [
            ("observation_unit", *status_equal(source, row, "observation_unit")),
            ("specimen_condition", *status_equal(source, row, "specimen_condition")),
            ("bending_test_regime", *status_equal(source, row, "bending_test_regime")),
            ("predictor_semantics", *status_documented(row, "predictor_harmonization")),
            ("scientific_pooling", *status_documented(row, "pooling_justification")),
        ]
        # Non-compensatory gate: one established hard mismatch alone forbids transport.
        # Unknown metadata is withheld, never interpreted as a positive compatibility test.
        gate = ("VETO" if any(x[1] == "VETO" for x in checks) else
                "UNVERIFIED_WITHHOLD" if any(x[1] == "UNVERIFIED" for x in checks) else "PASS")
        for dim, state, rationale in checks:
            output.append({"target": name, "role": row["role"], "dimension": dim,
                           "g_status": state, "rationale": rationale, "final_gate": gate})
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    decisions = {x["target"]:x["final_gate"] for x in output}
    assert decisions == {
        "USDA_internal_source_strata":"PASS",
        "Pinus_patula_boards":"VETO", "Spruce_lamellae":"VETO",
    }
    assert len(output)==15
    payload = {
        "type": "post_hoc_revision_documentation_not_new_validation",
        "input_columns": sorted(ALLOWED_COLS),
        "contains_MOE_MOR_or_other_target_outcomes": False,
        "num_dimensions": 5, "num_regimes_compared":3,
        "decisions":decisions,
        "caveat": "USDA internal PASS is a metadata-level demonstration only, not an independent positive transport test."
    }
    (OUTPUT.parent/"audit_summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print("TEST PASS: metadata schema excludes target outcomes")
    print("TEST PASS: 3 comparison regimes x 5 dimensions = 15 recorded decisions")
    print("TEST PASS: source-to-Pinus veto and source-to-Spruce veto")
    print("TEST PASS: USDA internal metadata pass (not independent external validation)")
    print("TEST PASS: no model training, target outcomes, or target result files read")
    return payload

if __name__ == "__main__":
    run()
