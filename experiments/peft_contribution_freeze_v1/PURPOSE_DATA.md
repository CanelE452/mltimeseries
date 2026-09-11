# Study31 data preparation contract

This data-only contract supports `peft_contribution_freeze_v1`. The downstream decision is whether a single training trajectory can decide when LoRA updates may be omitted while the head continues to adapt. The data must therefore make the contribution/freezing comparison reviewable without changing source choice, target choice, or evaluation period after looking at model outcomes.

The domain is hourly multivariate time-series forecasting for a Chronos2 PEFT diagnostic. The consumer expects 336 hours of context, a 48-hour target horizon, daily origins, two scored target channels, and separate fit and holdout archives. The quality bar is development ML: temporal leakage is not allowed, train statistics must be fit on the train interval only, and every scored target/split must have at least 70% finite target-window labels.

## Panels

`bdg2` uses `data_external/bdg2_coarse_supervision_v1/raw/electricity.csv` with metadata and QC from `data_external/bdg2_coarse_supervision_v1/raw/metadata.csv` and `data_external/bdg2_coarse_supervision_v1/qc.json`. The fixed four-channel order is the first four Eagle selected target ids already recorded in the BDG2 QC file: `Eagle_office_Elias`, `Eagle_office_Elvis`, `Eagle_office_Flossie`, `Eagle_office_Francis`. The first two are scored targets. The latter two are retained as context-only covariates and are masked out of target loss and scoring.

`jena` uses `data/jena_mpi_roof/mpi_roof_2023a.csv` and `data/jena_mpi_roof/mpi_roof_2023b.csv`. The fixed four-channel order is the first four numeric variables from the existing Jena parser order: `p (mbar)`, `T (degC)`, `Tpot (K)`, `Tdew (degC)`. The first two are scored targets. The latter two are context-only covariates and are masked out of target loss and scoring. The raw Jena files are ten-minute records; this preparation keeps only top-of-hour rows and does not average future rows.

## Fixed chronology

Both panels start at `05-04 00:00` in their respective years: BDG2 at `2017-05-04 00:00:00`, Jena at `2023-05-04 00:00:00`. The reused archive writer applies the Study20 split: 14 days precontext, 91 days train, 2-day train-to-validation embargo, 31 days validation, 2-day validation-to-calibration embargo, 21 days calibration, and 81 days evaluation. This yields 90 train origins, 30 validation origins, 20 calibration origins, and 80 evaluation origins.

Fit archives contain train and validation origins only and are physically truncated at the validation boundary. Holdout archives contain calibration and evaluation origins only. Context values use a precontext-only fallback median followed by causal forward fill. Target values preserve raw NaNs. Fit mean, std, and median are recomputed from observed train rows only.

## Exposure limits

These are not wholly unseen sources. BDG2 raw/QC and part of 2017 were already used in the coarse-supervision work, although the proposed late evaluation interval is nonoverlapping with the documented April-June 2017 fine evaluation. Jena 2024 was already used in the adaptation-scope pilot, so Jena 2023 is a nonoverlapping time block within the same source family rather than an independent new source family. FM pretraining overlap is unknown for both.

No model prediction files or E performance results are read by `prepare.py`.

## Preparation correction before any training or E forecast

The Jena 2023b source contains 24 duplicate top-of-hour timestamps from 2023-10-20 06:00 through 2023-10-21 05:00. All four selected channel values are identical in each pair. The fixed parser collapses exact duplicate selected-channel rows to one; conflicting values raise an error. It does not average, interpolate or choose an outcome-dependent record. The unique grid must still pass the complete-hourly test. The duplicates fall inside E; this is a raw alignment correction made before any model outcome, and is explicitly part of the protocol.

Preparation had two failures: a string/datetime conversion error, then the duplicate-grid rejection. The first pair of partial BDG2 archives was deleted by the data agent before being regenerated, contrary to failure-artifact preservation. The second pair was retained and reused after content/QC validation. Original raw sources and prior studies were not modified. See `runs/peft_contribution_freeze_v1/preparation_failures.json` for the available incident record; do not imply deleted artifacts or unavailable original logs were preserved.
