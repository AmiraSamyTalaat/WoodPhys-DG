# Archived workflow scripts

The historical scripts preserve early-stage dataset screening, model development, frozen predictions, and subsequent analyses with distinct dependencies. Stage 0B–3B code is retained for provenance, alongside Stage 4H–4I diagnostic analysis. `source_freeze` uses earlier stage input archives; `frozen_external` consumes archived source and pine input results; `metadata_gate_original` is the later binary veto implementation; `cluster_shift` and `baselines_physics` use separately archived input-stage ZIP files.

These files are not a single standalone pipeline. The main analysis scripts under `scripts/` and the revision metadata reconstruction under `gate_audit/` are independently separated to avoid conflating the frozen prediction experiment with later methodology development.
