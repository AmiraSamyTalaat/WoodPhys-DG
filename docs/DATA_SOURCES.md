# Source data and local inputs

The original study uses three distinct measurement regimes. Third-party observation-level records are not bundled in the public release. Obtain the data from the original providers and respect their stated licenses and conditions.

1. USDA clear-wood species/group data (422 complete model-ready observations). Source: Kretschmann, D. E., *Wood Handbook—Wood as an Engineering Material*, GTR FPL-GTR-190, Chapter 5. https://doi.org/10.2737/FPL-GTR-190. The recorded canonical input is `usda_clearwood_primary_model_ready.csv`.
2. Pine (207 *Pinus patula* boards, 24 trees). Erasmus, J., Drew, D. M., & Wessels, C. B. (2020), SUNScholarData. https://doi.org/10.25413/sun.12241268. The recorded canonical input is `Board_Data.csv`.
3. Spruce (2,522 validated lamellae, 88 sample-name-prefix groups after filtering). Achatz et al., *Spruce Lamellae Dataset*. https://doi.org/10.5281/zenodo.16925468. The original repository has `dataset_spruce.csv`; the recorded analysis input is `spruce_structural_board_separate_regime.csv`.

Local computation using the existing scripts expects the following data filenames in repository root `data/`:

```
Board_Data.csv
EXTERNAL_FROZEN_PREDICTIONS.csv
SPRUCE_FROZEN_EXTERNAL_PREDICTIONS.csv
SPRUCE_SOURCE_WEIGHTS.csv
spruce_structural_board_separate_regime.csv
usda_clearwood_primary_model_ready.csv
APPLICABILITY_WEIGHTED_DEVELOPMENT_PREDICTIONS.csv
USDA_DOMAIN_AND_TAXON_SPLIT_MANIFEST.csv
```

The frozen prediction files and exact preprocessing/feature mapping came from the original archived analysis. They must not be replaced by re-fitted estimates when checking the frozen zero-shot experiment.

**Preparation caveat:** this release preserves scientific scripts and archived aggregate tables, but the complete original raw-to-model-ready transformation and upstream input-stage archives are not packaged as one fully executable public pipeline. Source records, their licenses, and individual experiment-specific provenance should be checked before any independent release of the canonical CSVs.
