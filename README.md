# WoodPhys-DG

Research code and archived aggregate outputs for the manuscript **Outcome-blind non-compensatory regime gating and group-budgeted adaptation for reliable structural-wood prediction under dataset shift**.

The original source prediction pipeline was frozen prior to evaluation of the pine outcomes. The metadata-based compatibility gate was developed **subsequently**, following the initial pine transport results. Its retrospective application is documented separately from the frozen-source evaluation. No independent external compatible-regime acceptance control is claimed.

## Contents

| Directory | Purpose |
| --- | --- |
| `scripts/` | Primary scientific evaluation of target models, uncertainty, and group-aware prediction. |
| `gate_audit/` | Five-dimension metadata decision reconstruction used for the manuscript revision. |
| `historical/` | Archived research scripts for the frozen source system, transport evaluation, original metadata veto, and diagnostic analyses. |
| `protocol/` | Recorded analysis specifications and claim boundaries. |
| `tables/` | Archived **aggregate** result tables for result inspection (not raw observation files). |
| `docs/` | Data sources, command descriptions, software environment, and scope limitations. |
| `data/` | User-provided local data directory; no third-party source records are redistributed. |

## Quick checks

Python 3.13 is the recorded environment. Create a virtual environment, install `requirements.txt`, then run:

```bash
python -m compileall -q scripts gate_audit historical
python gate_audit/scripts/run_metadata_gate_audit.py
python scripts/stage4r/run_all_stage4r.py
python scripts/verify_repo.py
```

These are **audit and consistency checks**, not retraining runs. Primary archived summary outputs are held in `tables/`.

## Recompute analyses

First obtain the cited datasets and prepare the exact input CSV filenames listed in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md). Place the input files in `data/`. No redistribution permission is implied by the public availability of an external dataset.

For the pine analysis:

```bash
python scripts/stage4r/run_all_stage4r.py --force
```

For the spruce target analysis:

```bash
python scripts/run_all_stage4q.py
```

Other historical scripts need their original input-stage archives; see [docs/REPRODUCIBILITY_SCOPE.md](docs/REPRODUCIBILITY_SCOPE.md). Recomputing the entire frozen source project is not provided as a single end-to-end command here.

## Method chronology and limitations

The original frozen source evaluation precedes the exploratory metadata compatibility gate. The five-dimensional metadata audit is a later explanatory reconstruction, not an independently pre-registered algorithm or an external positive-control experiment. The spruce sample-name prefixes are not proven independent trees; prefix-disjoint evaluation only prevents the same recorded prefix appearing in both training and test.

## Data and attribution

External datasets are attributed in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md). The MIT license applies to this repository's code, not to the independently sourced datasets. The code release should be cited using `CITATION.cff` and the version-specific Zenodo DOI after a release is archived.
