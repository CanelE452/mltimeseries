# OA-RESOLUTION-PILOT-v1 - figure captions

Generated 2026-09-06 16:47 by `experiments/oa_resolution_pilot_v1/figures.py`.
These are analysis-mode figures: they are meant to be read together with the
numbers in STATUS.md, not as standalone paper figures.

## fig1_o_vs_m_unseen_interpolation.png

Relative reduction of the primary loss of the interval-integrated arm O over the
centre-time metadata arm M at the two resolutions that were never trained
(r = 3, 6), for each of the eight dataset x operation x r cells, next to the
paired moving-block bootstrap 95% intervals over all cells and per dataset; the
macro is -0.049% with an interval of [-0.279, +0.175] that contains zero, and
the per-dataset intervals contain zero as well, so the pre-registered +1.0%
condition is not approached from either side.
Limitation: the intervals resample time origins only and not model seeds, and the
two seeds put the macro at -0.300% and +0.202%, a spread wider than the estimate.

## fig2_arms_by_resolution_role.png

Primary loss of the three arms against the resolution factor r, with one panel per
resolution role so that trained resolutions, unseen interpolation and unseen
extrapolation are never read off a single axis, and one row per dataset because
the loss scales differ by a factor of three; the ordering of the arms flips
between the datasets, R being lowest on jena and M and O being lowest on uci
(O vs M at unseen interpolation is -0.139% on jena and +0.042% on uci).
Limitation: each point is a single number aggregated over two model seeds and
carries no interval here; the intervals are in fig1 and in bootstrap.json.

## fig3_training_curves.png

Validation primary loss against update for the three arms and both seeds, with the
selected checkpoint marked, shown at full range and again from update 2000 on
where the arms separate; validation uses only the trained resolutions r = 2, 4, 8,
so nothing about the unseen resolutions enters checkpoint selection, and the same
dataset-dependent ordering as in fig2 is already visible during training.
Limitation: validation loss is not the quantity the pilot asks about, and the gap
between the two seeds is of the same size as the gap between M and O.

## fig4_accuracy_vs_cost.png

Unseen-interpolation primary loss against wall time and against parameter count for
the three trained arms and for FlowState r1.1 read zero-shot, the bars giving the
spread over the four unseen-interpolation cells; R costs about 2.5 times the wall
time of M and O for its reconstruction step, while M and O have the same parameter
count, the same wall time to within the seed-to-seed spread, and land on the same
loss.
Limitation: FlowState is not a matched-budget competitor. It never saw these
series, its pre-training compute is not on the wall-time axis, and it is read on
the resampled variant because its native-rate variant defines only the 60-minute
term and only at r = 2, 3, 6.

## sanity floors

Every arm in every figure sits far below the naive anchors: persistence is
0.565 (jena) and 1.428 (uci), seasonal naive at one day is 0.373 and 0.745.

Recorded decision for the pilot: INCONCLUSIVE.

Files:
- `results/oa_resolution_pilot_v1/figures/fig1_o_vs_m_unseen_interpolation.png`
- `results/oa_resolution_pilot_v1/figures/fig2_arms_by_resolution_role.png`
- `results/oa_resolution_pilot_v1/figures/fig3_training_curves.png`
- `results/oa_resolution_pilot_v1/figures/fig4_accuracy_vs_cost.png`
