# HQ-TOKEN-PILOT-v1 - figure captions

Generated 2026-09-06 13:38 by `experiments/hq_token_pilot_v1/figures.py`.
These are analysis-mode figures: they are meant to be read together with the
numbers in STATUS.md, not as standalone paper figures.

## fig1_primary_contrast.png

Relative improvement in test MSE of the horizon-conditioned pooler (C) over the
input-only pooler (I) for each dataset x horizon cell, with paired moving-block
bootstrap 95% intervals, shown at full scale next to the pre-registered +1.0% gate
and again on a 3.2x narrower x-scale; the macro over the six cells is -0.037%
[-0.128, +0.033], about 27 times smaller than the gate it had to
clear.
Limitation: the intervals resample time origins only and not model seeds, so they
do not carry the seed-to-seed spread of up to 5.05% measured on ETTm2, and the two
electricity cells rest on 4 effective time blocks, which makes their intervals
narrow for a reason that has nothing to do with precision.

## fig2_accuracy_vs_latency.png

Test MSE against measured end-to-end latency for every configuration that was
actually trained - the five 32-token arms (U, H_STATIC, I, C, R) and the 64-token
DENSE arm - one panel per dataset x horizon cell, showing that learned pooling is
worth about +2.84% over uniform pooling at the same token budget while the
choice of query is worth -0.04%, and that the 64 to 32 compression itself
gave up +3.72% of accuracy.
Limitation: filled markers are the single model seed 2026090601, the only seed that
has all six arms, the open markers show the second seed for the three core arms and
the gap between seeds is often wider than the gap between arms; latency was measured
on this 1.5M-parameter pilot, where the pooling scorer costs more than the shorter
sequence saves, so the x-axis must not be read as the cost profile of a large model.

## fig3_pooling_weights.png

Mean pooling weight on the first patch of each of the 32 groups at H=96 against
H=336 for the same inputs: on the three real datasets the two horizon curves are
indistinguishable (mean absolute weight change 0.00051, 0.00122, 0.00135, and no
single group moves more than 0.00224), while the same architecture trained on a
synthetic task where the horizon must matter separates the two horizons by 0.813
on an identical y-axis.
Limitation: the right panel is a capacity check on a synthetic task and is not a
benchmark result, and the left panel is a mean over 256 test windows of the first
weight in a group of two, so it shows that the horizon-conditioned query is inert on
average, not that no individual window ever moved.
