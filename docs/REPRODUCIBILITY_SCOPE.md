# Reproducibility scope and evaluation chronology

## What this release can directly verify without raw datasets

- Presence and parseability of the scientific Python scripts.
- Archived aggregate summary files and their descriptive shapes.
- The exact group-budget and model-parameter specification used in the archived source scripts.
- Retrospective five-dimension gate decisions based **only on curated metadata**, with a schema guard excluding MOE and MOR outcome columns.

## What requires external data

- Recomputing original and grouped pine model fits.
- Recomputing spruce grouped target models, learned risk, and conformal intervals.
- Recreating the frozen development ensemble from upstream stage inputs.
- Rebuilding the complete figure series for the paper.

The Stage 0B–3B scripts under `historical/`, including `historical/source_freeze` and `historical/frozen_external`, are original workflow components. They rely on stage-input ZIP artifacts and do not run directly from the public tree without those historical inputs. The original metadata veto script operated at regime level, and must not be presented as a five-dimension prospective evaluator.

## Core distinction

1. **Frozen source-model evaluation:** model and parameters frozen before original pine target outcomes were evaluated.
2. **Retrospective compatibility gate:** formulated after initial pine results; it can use measurement-regime descriptors without consuming target outcomes when assigning a decision.
3. **Separately trained target-model analysis:** uses target-regime labels within group-separated training folds after transport is rejected.
4. **Metadata audit:** later documentation of five gate dimensions for the revision; it does not add an independent compatible-regime dataset.

A release with a DOI certifies preservation of this source tree, not that all reported models were retrained or all experimental claims independently validated.
