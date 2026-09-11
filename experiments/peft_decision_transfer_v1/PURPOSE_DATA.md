# Study33 data contract: decision-transfer fresh-period preparation

This data preparation supports `peft_decision_transfer_v1`. It prepares two chronological episodes for two four-channel hourly panels: Jena MPI Roof and BMRA generation. The downstream decision is whether a bounded development episode can select an adaptation policy that transfers to a later sealed episode without using sealed forecast outcomes.

The archive contract follows the validated Study20/31 layout: `336h` context, `48h` horizon, daily origins, `precontext14 + train91 + gap2 + V31 + gap2 + cal21 + E81`. The produced archives keep all `90 train`, `30 V`, `20 cal`, and `80 E` origins. A later run may subsample E origins for the bounded screen, but the data archive itself is not subsampled.

## Panels and fixed periods

Jena uses local `data/jena_mpi_roof/mpi_roof_2021a.csv`, `2021b.csv`, `2022a.csv`, and `2022b.csv`. The channel order is the same as Study31: `p (mbar)`, `T (degC)`, `Tpot (K)`, `Tdew (degC)`. The first two channels are scored targets and the next two are context-only covariates.

BMRA uses `data_external/ucp_path_pilot_v1/extracted/bmra/BMRA_Data/standardFormat/BMRA_data.parquet` and metadata from the paired `BMRA_metadata.parquet`. The raw 30-minute grid is aggregated to hourly means. The fixed channels are `E_BNWKW-1`, `E_BRYBW-1`, `E_BURBO`, and `E_DALSW-1`. The first two channels are scored targets and the next two are context-only covariates.

The episodes are fixed before forecast evaluation:

- `dev/jena`: block `2021-05-04 00:00:00` to `2022-01-01 00:00:00`.
- `test/jena`: block `2022-05-04 00:00:00` to `2023-01-01 00:00:00`.
- `dev/bmra`: block `2022-01-04 00:00:00` to `2022-09-03 00:00:00`.
- `test/bmra`: block `2023-01-04 00:00:00` to `2023-09-03 00:00:00`.

## Missingness and leakage controls

`target_values` preserve raw hourly NaNs. The target loss mask scores only finite cells in target channels `[0, 1]`; no target labels are imputed for loss or evaluation. `context_values` use the original past-only rule: a precontext-only median fallback is fitted before the first train origin, then a causal forward fill is applied. Fit mean, standard deviation, and median are fitted only on observed train-period rows.

The between-split target windows do not overlap under half-open intervals. Train-to-V and V-to-cal have 48-hour embargoes. The last calibration target end equals the first evaluation origin, so cal/E have no overlapping target timestamp.

BMRA channel selection used a pre-run availability screen over the proposed split windows: active `E_` generator columns with finite fraction at least 70% and nonconstant values in every split part, then the first four by BMU ID among the passing columns. This is a quality/availability screen, not a forecast-performance screen, and claims must disclose it.

## Freshness limits

These are new target-label periods under checked local experiment histories, not new raw sources. Jena raw/source family has been used elsewhere, with checked target-label exposure in Study31 for 2023 and S1 for 2024. BMRA raw/source has been processed in UCP work for 2019/2020/2021. Foundation-model pretraining overlap remains unknown.
