# Metadata gate audit (retrospective)

Run from repository root:

```bash
python gate_audit/scripts/run_metadata_gate_audit.py
```

The audit reads the curated `gate_audit/inputs/regime_metadata.csv` file and writes `gate_audit/outputs/five_dimension_decisions.csv` and `gate_audit/outputs/audit_summary.json`.

This audit was prepared to document the five decision dimensions after the initial transport experiments. It contains no target mechanical-property measurements. It does not establish a prospective gate test or a same-regime independent external acceptance result.

Output codes `PASS`, `VETO`, and `UNVERIFIED_WITHHOLD` correspond, respectively, to eligible, rejected, and unresolved metadata states. `PASS` indicates eligibility for assessment only; it does not guarantee model accuracy.
