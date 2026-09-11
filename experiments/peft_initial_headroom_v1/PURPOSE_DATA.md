# Study34 data contract: development-only initial-headroom diagnostic

This data preparation supports `peft_initial_headroom_v1`. It prepares two chronological four-channel hourly panels for a development diagnostic only. The downstream decision is whether initial LoRA+head adaptation has any selected-output headroom beyond a frozen head, an equal-parameter wide head, and a small output-correction baseline before investing in another controller or mechanism intervention.

The archive contract reuses the validated Study33 layout: `336h` context, `48h` horizon, daily origins, `precontext14 + train91 + gap2 + V31 + gap2 + cal21 + D81`. The produced archives keep all `90 train`, `30 V`, `20 cal`, and `80 D` origins. The run forecasts `D` origins at indices `0,4,...,76` for this bounded screen. The calibration split is unused.

## Fixed panels and periods

Jena uses local `data/jena_mpi_roof/mpi_roof_2019a.csv` and `mpi_roof_2019b.csv`. The channel order is the same as Study31 and Study33: `p (mbar)`, `T (degC)`, `Tpot (K)`, `Tdew (degC)`. The first two channels are scored targets and the next two are context-only covariates.

BMRA uses `data_external/ucp_path_pilot_v1/extracted/bmra/BMRA_Data/standardFormat/BMRA_data.parquet` and metadata from the paired `BMRA_metadata.parquet`. The raw 30-minute grid is aggregated to hourly means. The fixed channels are the Study33 channels `E_BNWKW-1`, `E_BRYBW-1`, `E_BURBO`, and `E_DALSW-1`. The first two channels are scored targets and the next two are context-only covariates.

The diagnostic blocks are fixed before any model outcome is read:

- `jena`: block `2019-05-04 00:00:00` to `2020-01-01 00:00:00`.
- `bmra`: block `2018-01-04 00:00:00` to `2018-09-03 00:00:00`.

## Missingness and leakage controls

`target_values` preserve raw hourly NaNs. The target loss mask scores only finite cells in target channels `[0, 1]`; no target labels are imputed for loss or evaluation. `context_values` use the original past-only rule: a precontext-only median fallback is fitted before the first train origin, then a causal forward fill is applied. Fit mean, standard deviation, and median are fitted only on observed train-period rows.

The between-split target windows do not overlap under half-open intervals. Train-to-V and V-to-cal have 48-hour embargoes. The last calibration target end equals the first diagnostic origin, so cal/D have no overlapping target timestamp.

BMRA channel selection reuses the fixed Study33 channel order. A pre-run availability check confirmed the 2018 block has finite fractions of 0.9590, 0.9962, 0.9985, and 0.9910 for the four channels, with three all-missing hourly rows. This is an availability/quality check, not a forecast-performance screen.

## Exposure limits

These are new target-label periods under the checked local PEFT experiment history, not new raw source families. Jena raw/source family has been used elsewhere, and later Jena periods in 2021, 2022, 2023, and 2024 have already been inspected in prior PEFT diagnostics. BMRA raw/source family has been processed in UCP work for 2019/2020/2021 and Study33 used 2022/2023. The 2018 BMRA diagnostic block is earlier than those checked target-label windows. Foundation-model pretraining overlap remains unknown.
