# UCP-PATH-PILOT-v1 — data contract

Phase A audit, instruction section 5. Everything below was read from the actual
Zenodo metadata, the author code and the archive files, not assumed. Where a
value contradicts the instruction's provisional default or the author's own
docstring, the contradiction is stated rather than smoothed over.

Audit date: 2026-09-06. Branch `uncertain-covariate-path-pilot-v1`, base
`36b01d2b84df16f03447a8e214723fc61ae147fd`.

## 1. Sources

| key | Zenodo record | DOI | file | bytes | md5 verified |
|---|---|---|---|---|---|
| ens_2019_2020 | 13255991 | 10.5281/zenodo.13255991 | `ecmwf_ens_2019-2020.tar.gz` | 36,034,983,151 | in progress |
| ens_2021 | 13256007 | 10.5281/zenodo.13256007 | `ecmwf_ens_2021.tar.gz` | 18,003,586,760 | pending |
| bmra | 13256014 | 10.5281/zenodo.13256014 | `BMRA_Data.tar.gz` | 330,104,976 | yes |
| forecasting_software | 13309948 | 10.5281/zenodo.13309948 | `forecasting_software.tar.gz` | 117,912 | yes |
| read_bmra | 13309890 | 10.5281/zenodo.13309890 | `read_bmra.tar.gz` | 239,259 | yes |

All five records are `access_right: open`, published 2024-11-14. Four carry
`cc-by-4.0` in the Zenodo licence field. Record 13256014 (BMRA) carries no
licence id in its Zenodo metadata; its own README states the data is subject to
the Elexon copyright licence for BMRS data and contains "BMRS data © Elexon
Limited copyright and database right 2024". Research use of the archived copy is
therefore governed by Elexon's terms, not by CC-BY. Recorded, not resolved to a
redistribution right: nothing from this record is committed to the repository.

## 2. Author pipeline, as read

`probFor` (record 13309948) and `read_bmra` (record 13309890) were downloaded,
checksummed and read before any large archive was fetched.

- The ensemble archive is named `ECMWF/ENS` but the author code addresses it as
  `TIGGE` throughout (`setup["folders"]["nwp"]["TIGGE"]`, `read_TIGGE`). Files
  are `<year>/tigge-GB_<YYYYMMDDHH>.nc`.
- **The member dimension is never averaged anywhere in the author pipeline.**
  `read_TIGGE` writes `data[var][horizon][0, number, 0, lat, lon]`; every
  downstream step (`_selectDomain`, `_selectReleases`, `concat_hrz_pcm`) slices
  the release, latitude and longitude axes and passes the member axis through
  untouched. Path identity across leads is therefore available in the source and
  survives the author's contract. `PATH_IDENTITY_NOT_AVAILABLE` does not apply.
- The author's TIGGE variable set is `["u10", "v10", "t2m"]`; `["u10", "v10",
  "u100", "v100"]` belongs to the separate HRES deterministic archive, which this
  pilot does not use.
- Farm-level extraction uses a 2x2 grid box around the site
  (`lat_pts=2, lon_pts=2` for every TIGGE model in `example.py`).

### Where the author's docstring is wrong about its own data

`read_TIGGE`'s docstring states "Release freq.: 24 hours" and `example.py` sets
`setup["nwp"]["release_freq"]["TIGGE"] = 24`. The archive contains both 00 and
12 UTC base times — `tigge-GB_2019032312.nc`, `tigge-GB_2019081612.nc` and
`tigge-GB_2019102012.nc` are all present in the first 15 members of the tar
stream. The instruction's provisional default (00 and 12 UTC) matches the files;
the author's configuration constant does not. Base times are taken from the
filenames.

`read_TIGGE` also indexes a `step` variable. The netCDF files have no `step`
variable: the lead axis is the `time` dimension, and the author's positional
index `cc = horizon / 6` lands on it correctly by accident of ordering.

## 3. Ensemble NWP tensor, as measured

Read from `tigge-GB_2019032312.nc`, `tigge-GB_2019070200.nc` and
`tigge-GB_2019082400.nc`, recovered from the partially downloaded archive.

| property | measured value |
|---|---|
| base times | 00 and 12 UTC, from the filename; equals the file's first valid time |
| lead axis | `time` dimension, 61 values, +0 h to +360 h in exact 6 h steps |
| members | `number` dimension, 50, values 1..50 |
| control member | **absent** — `number` starts at 1, so these are the 50 perturbed members only |
| grid | latitude 59.5 down to 50.0 (20 points), longitude -8.0 to 3.5 (24 points), 0.5 deg |
| variables | `u10`, `v10`, `t2m`, `d2m`, `msl`, `tcw`, `tcc`, `ssr`, `tp` |
| storage | int16 with per-file `scale_factor` / `add_offset`; netCDF4 unpacks to physical units |
| file size | 26,356,124 bytes each, identical across files |

Members are the same axis for every lead within a file, so member `k` at +6 h and
member `k` at +72 h are the same perturbed forecast. That is what makes the
P versus D contrast testable at all; test T05/A06 checks it on the extracted
tensors rather than trusting this note.

### Selected weather variables

`D = 4`: `u10`, `v10`, `t2m` (the author's TIGGE set) plus `ws10 =
sqrt(u10^2 + v10^2)`.

`ws10` is a deterministic transform of two variables already in the set, and the
author's own preprocessing builds exactly this quantity from reanalysis winds
(`_createWSseries` in `bmra_wind.py`). It is computed per member and per lead, so
it cannot leak path information into one arm and not another — every arm sees the
same four channels.

`ssr` (J m^-2) and `tp` (kg m^-2) are excluded. Both are accumulations whose
window is not documented in the file, and instruction section 9 says to drop
variables whose accumulation semantics are unclear rather than guess. `d2m`,
`msl`, `tcw` and `tcc` are dropped to stay with the author's variable contract.
This choice is fixed here, before any model is fitted.

## 4. Target

`BMRA_data.parquet`, metered power output in **MW**, 30-minute resolution,
2015-01-01 00:00 to 2023-12-31 23:30, 157,776 rows and 184 farm columns, with a
gap-free 30-minute index. NaN marks timestamps with non-zero Bid Acceptance
Volume, that is, curtailed periods.

`BMRA_data_final.parquet` is **not** used. It is the metered series normalised by
`BMRA_power_availability.parquet`, and that availability series is estimated from
ERA5 reanalysis 100 m wind speed inside `_nomPowerCorrection`. Reanalysis is not
available at forecast time and the normaliser is fitted across the whole record,
so using it as the target would put an availability estimate built from future
information into the label. Instruction section 8 forbids the normalised target
before a leakage audit; this pilot does not use it at all. `TARGET_UNIT_UNRESOLVED`
does not apply: the unit is MW, stated by the source README.

Target at lead `t` is the metered value at the exact timestamp
`origin + 6t hours`, joined on datetime, never on positional index. No temporal
integration is applied; the author's 60-minute averaging is a choice of their
forecasting tool, not a property of the data.

## 5. Time and split contract

- Origin = NWP base time, 00 and 12 UTC.
- Leads 6, 12, ..., 72 h, so `T = 12`, taken from the ENS `time` axis by
  timestamp match, not by assumed index.
- Chronos history: the 1344 half-hourly target values strictly before the origin,
  that is, 28 days. Future weather never enters Chronos.
- Chronos forecast: 144 half-hourly steps covering 72 h; the 12 lead features are
  selected by datetime join against the ENS valid times.
- Splits: 2019 train, 2020 validation, 2021 test. An origin is dropped when any
  of its future targets crosses the split boundary.
- The 2021 test set is read once.

Metered data is treated as available at the origin. Real BMRA publication latency
was not verified, so the result is reported as a **base-time-aligned retrospective
operational forecast experiment**, not as a deployable operational result.

## 6. Missing data

An origin is kept for a farm only when all twelve future targets are present.
Measured effect, which the raw coverage number hides completely:

| year | origins | median farm keeps (all 12 leads) | median 30-min coverage |
|---|---|---|---|
| 2019 | 730 | 88.2% | 98.7% |
| 2020 | 732 | 79.6% | 96.2% |
| 2021 | 730 | 21.3% | 75.3% |

GB curtailment in 2021 blanks the metered series in clustered runs, so 2021 is
far more expensive than its coverage figure suggests. Farm eligibility is
therefore stated in usable origins, not in coverage.

## 7. Farm selection

Eight sites, ranked on 2019 information only. Full rule and the eligibility table
are in `farm_selection_rule.json` and `farm_eligibility_table.csv`.

| farm | name | type | lat | lon | usable origins 2019 / 2020 / 2021 |
|---|---|---|---|---|---|
| E_BURBO | Burbo Bank Offshore Windfarm | Offshore | 53.49 | -3.18 | 724 / 700 / 260 |
| T_BEATO-1 | Beatrice Offshore Wind Unit 1 | Offshore | 58.09 | -2.95 | 706 / 720 / 224 |
| T_GANW-13 | Galloper Offshore Windfarm 13 | Offshore | 51.88 | 2.04 | 718 / 698 / 421 |
| T_RMPNO-1 | Rampion Offshore Windfarm 1 | Offshore | 50.64 | -0.18 | 718 / 715 / 350 |
| E_GFLDW-1 | Goole Fields 1 Windfarm | Onshore | 53.67 | -0.89 | 724 / 726 / 305 |
| E_BNWKW-1 | Burn of Whilk Windfarm | Onshore | 58.36 | -3.20 | 653 / 520 / 267 |
| T_COUWW-1 | Cour Wind Farm | Onshore | 55.68 | -5.48 | 717 / 674 / 320 |
| T_AKGLW-2 | Aikengall 2 Wind Farm Generation | Onshore | 55.92 | -2.49 | 696 / 389 / 291 |

These are eight wind farm sites, not eight datasets.

Two things to read with the table:

- `E_BURBO` is labelled `Onshore wind` in the source `generator_list.csv` while
  its own `Name` field reads "Burbo Bank Offshore Windfarm". Rows whose Name and
  Type contradict each other are relabelled to Offshore; on this snapshot that
  rule fires on exactly one row, and it needs no knowledge outside the file.
- The closest pair is across groups: `T_BEATO-1` and `E_BNWKW-1` sit 33 km apart
  in far north Scotland and will draw on neighbouring ENS grid cells. The
  farthest-point rule is applied within each group, as instruction section 7
  specifies, so it cannot separate a cross-group pair. Noted before any model was
  fitted; the rule was not changed afterwards.

## 8. Hard-stop tokens

None raised.

- `DATA_CONTRACT_UNRESOLVED` — no. Every field above was measured.
- `PATH_IDENTITY_NOT_AVAILABLE` — no. 50 members, stable across the lead axis.
- `TARGET_UNIT_UNRESOLVED` — no. MW, per the source README.
- `LICENSE_UNRESOLVED` — no, for use. The BMRA record's licence is Elexon's
  rather than CC-BY, which restricts redistribution, not analysis. No raw data is
  committed.
- `DISK_BLOCKED` — no. 890 GB free on E:, against about 54 GB of archives.

## 9. Environment

Two interpreters, split so neither is disturbed:

- extraction and evaluation: Anaconda base, Python 3.13.9, numpy 2.3.5,
  pandas 2.3.3, xarray 2025.10.1, netCDF4 1.7.4, lightgbm 4.7.0
- training and Chronos: conda env `mlts`, Python 3.11.16, torch 2.11.0+cu128,
  chronos 2.3.1, transformers 5.16.1, RTX 4070 12.88 GB

`autogluon/chronos-2-synth` resolves on the Hub at revision
`3607918a9fd027d5c465d8213e46b98e2c041cea`, loads as `Chronos2Pipeline`, and
reports native quantile levels 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
0.8, 0.9, 0.95, 0.99. Its `predict_quantiles` takes `inputs` shaped
`(n_series, n_variates, history_length)`. Frozen: no fine-tuning, no LoRA.

## 10. Download cost

Measured single-stream throughput from Zenodo is 1.68 MB/s. Four parallel range
streams delivered 1.23 MB/s in aggregate, so the limit is the link or a per-IP
cap, not per-connection throttling, and parallel fetching was abandoned. The two
ENS archives therefore take roughly nine hours to transfer. That is outside the
8 GPU-hour training budget, as instruction section 29 allows.
