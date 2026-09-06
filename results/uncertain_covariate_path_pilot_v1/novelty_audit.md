# Novelty audit — UCP-PATH-PILOT-v1

Instruction section 27. Searched 2026-09-06 across the seven themes the
instruction lists. This is a targeted search, not a systematic review: it can
show that something exists, it cannot show that nothing does.

The question is narrow. Not "is uncertainty propagation new" — it is not, and the
prior note for this topic already records the 2002 GP-with-uncertain-inputs work
and the 2003 weather-ensemble demand forecasting work. The question is whether a
method functionally identical to **TemporalEncoder(member path) followed by a
permutation-invariant pool over members** already exists.

## What the search found

### The pooling half already exists

Höhlein, Schulz, Westermann and Lerch, *Postprocessing of Ensemble Weather
Forecasts Using Permutation-invariant Neural Networks*, Artificial Intelligence
for the Earth Systems 3(1), 2024. DeepSets over the ensemble: an encoder MLP is
applied to each member separately, then a permutation-invariant pooling function
summarises them. Mean, extremum and attention pooling were all tried and the
paper reports the choice of pooling mechanism to be of minor importance. This is
the D arm's mechanism, at a single forecast instance.
<https://journals.ametsoc.org/view/journals/aies/3/1/AIES-D-23-0070.1.xml>

Ben Bouallègue et al., *Improving Medium-Range Ensemble Weather Forecasts with
Hierarchical Ensemble Transformers* (PoET), Artificial Intelligence for the Earth
Systems 3(1), 2024. Self-attention **across the ensemble-member dimension**, applied
to the members themselves rather than to a predictive distribution, and agnostic to
ensemble size. Again a set operation over members, richer than mean pooling.
<https://journals.ametsoc.org/view/journals/aies/3/1/AIES-D-23-0027.1.xml>

### The member-wise temporal half also already exists

Roberts, *Ensemble-size-dependence of deep-learning post-processing methods that
minimize an (un)fair score*, arXiv:2602.15830, submitted 2026-02-17, preprint, not
peer reviewed. It introduces a "trajectory transformer", an adaptation of PoET
that "applies the transformer self-attention mechanism across the forecast
lead-time dimension independently to each member". The paper's own framing is that
this gives "zero information exchange between members during inference".
<https://arxiv.org/abs/2602.15830>

This is the P arm's first step, published seven months before this pilot.

### The composition was not found

Roberts' model keeps the ensemble as an ensemble: one calibrated member out per
member in, with no aggregation step anywhere. Its purpose is the opposite of
pooling — conditional independence between members is what makes the fair CRPS
usable. In the vocabulary of this pilot, that architecture sits closer to the MC
arm than to P.

No work was found that puts the two halves together in the order P uses: encode
each member's own path in time, **then** pool across members into a single
conditioning representation for a downstream forecaster with a different target
variable. Nor was one found that contrasts the two pooling orders under matched
parameter counts, which is the actual question this pilot asks.

### Adjacent families that solve a different problem

Ensemble copula coupling, the Schaake shuffle, COBASE and the composite-loss dual
graph neural network of Lakatos et al. (QJRMS, 2026) all restore spatio-temporal
dependence **in the predictive output**. This pilot is about dependence **in the
input covariate**, and the two are not substitutes: none of them changes what the
forecaster is told about the weather.

For wind power specifically, the 2025-2026 work found was about NWP correction,
graph networks over farm clusters, quantile-regression decomposition and diffusion
scenario generation. Nothing found encodes member trajectories as a conditioning
representation.

## Verdict

**NOVELTY_RISK_HIGH is not triggered** on the instruction's own criterion, because
no functionally identical Temporal-then-SetPool method was found.

**The risk is nevertheless material, and closer than comfortable.** Both halves
exist in the ensemble postprocessing literature, one of them from February 2026,
and the step from "attention over lead time per member" to "pool the results" is
small enough that an independent group could have taken it in work this search did
not surface. A reviewer who knows the postprocessing literature will ask this
question first.

What this pilot can defensibly own, if the screens come back positive, is the
**controlled contrast** — D against P at identical parameter count with the same
phi, the same temporal encoder and the same head, plus the P_BROKEN control that
keeps every per-lead marginal and destroys only the cross-lead member linkage —
and the setting, which is conditioning a frozen foundation model on an uncertain
future covariate rather than calibrating the weather forecast itself.

Whatever the verdict token says, this run does not license the phrase
NOVEL_METHOD_CONFIRMED.

## Sources

- <https://journals.ametsoc.org/view/journals/aies/3/1/AIES-D-23-0070.1.xml> — Höhlein et al. 2024, permutation-invariant postprocessing
- <https://arxiv.org/abs/2309.04452> — the same work as an arXiv preprint
- <https://journals.ametsoc.org/view/journals/aies/3/1/AIES-D-23-0027.1.xml> — Ben Bouallègue et al. 2024, PoET
- <https://arxiv.org/abs/2303.17195> — PoET preprint
- <https://arxiv.org/abs/2602.15830> — Roberts 2026, trajectory transformer (preprint)
- <https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.70119> — Lakatos et al. 2026, composite-loss graph neural network
- <https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.70138> — COBASE copula-based shuffling
- <https://doi.org/10.1002/we.70079> — Dantas and Browell, seamless short- to mid-term probabilistic wind power forecasting, the source of this pilot's data
