from __future__ import annotations
import json, sys
from pathlib import Path

out = Path(__file__).resolve().parent / "stage0b_output"
required = [
    "STATUS.json", "SUCCESS_MARKER.json", "CHECKPOINT.json", "CONFIG_SNAPSHOT.yaml",
    "RUN_LOG.txt", "ERROR_LOG.txt", "OUTPUT_VERIFICATION.json",
    "SCIENTIFIC_GATE_DECISION.json", "GATE_REPORT_AR.md",
    "FILE_MANIFEST.csv", "HASHES_SHA256.csv", "UPLOAD_THIS_STAGE0B_RESULT.zip",
    "data/mechanical_master_candidate.csv", "data/mechanical_master_candidate.xlsx",
    "data/usda_mechanical_raw.csv", "data/figshare_pinus_nigra_raw.csv",
    "reports/SOURCE_SUMMARY.csv", "reports/SCHEMA_COVERAGE.csv",
    "reports/EXTRACTION_QA.csv", "reports/TAXONOMY_RESOLUTION_QUEUE.csv",
    "reports/DUPLICATE_AND_LEAKAGE_RISKS.csv", "reports/FEASIBILITY_METRICS.json",
]
missing = []
for rel in required:
    p = out / rel
    allow_empty = rel == "ERROR_LOG.txt"
    if not p.exists() or (p.is_file() and p.stat().st_size == 0 and not allow_empty):
        missing.append(rel)
if missing:
    print("MISSING/EMPTY:")
    for x in missing: print("-", x)
    sys.exit(2)
status = json.loads((out / "STATUS.json").read_text(encoding="utf-8"))
marker = json.loads((out / "SUCCESS_MARKER.json").read_text(encoding="utf-8"))
verify = json.loads((out / "OUTPUT_VERIFICATION.json").read_text(encoding="utf-8"))
if status.get("status") != "COMPLETED" or not marker.get("success") or not verify.get("passed"):
    print("Status/marker/output verification did not pass.")
    sys.exit(3)
print("Verification PASS")
