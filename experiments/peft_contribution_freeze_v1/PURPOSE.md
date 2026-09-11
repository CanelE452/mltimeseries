# Study31: single-trajectory adapter contribution and freezing

Approved continuation of study30. Freeze this protocol and implementation before new fit/E prediction. Diagnostic pilot, not a novelty claim. No threshold search or test-based adjustment.

Question: does the current head-conditioned contribution of active LoRA predict when further LoRA updates can be omitted? On/off evaluation is an intervention on current adapters, but is NOT an isolated causal estimate of representation quality or of future learning value because the head coadapted.

## Data and shared budget

BDG2 late2017 and Jena2023, existing raw sources, nonoverlapping evaluation blocks relative to the previously documented evaluations (see PURPOSE_DATA.md). Not wholly unseen sources, nor proof of absence from FM pretraining. 336h context, 48h target; train90/V30/cal20/E80 daily origins with original embargoes. Cal unused. V halves origins [0:14] and [16:30] have nonoverlapping target hours; their contexts and trajectories are dependent.

Conditions FULL90, SPREAD30 and RECENT30 retain prior index definitions. Seeds27000/27001. Same Chronos2 snapshot, native backbone frozen; rank8 alpha16 dropout0 on 96 attention projections plus residual MLP533. Initial trainable count1768949; after freezing589301. AdamW LR1e-4 for both groups, weight_decay0, clip1, batch8/micro4, threads2. No new LR search: this LR was selected on old Bike/Household; transfer limitation applies equally to all arms.

FULL90 cap180, checkpoints [0,15,30,60,120,180]; subsets cap60, checkpoints [0,5,10,20,40,60]. Common random minibatch prefix and initial weights per cell/seed. Best observed full-V checkpoint (strict improvement, earliest tie) selected; every selection sealed before any E inference.

## Four actually executed arms

- FULL: joint LoRA/head to cap.
- ES2: stop joint training after two consecutive nonimproving V checks, min_delta0; restore best seen checkpoint. Patience counts checks, not epochs.
- FIXED_FREEZE: at cap/3, freeze all LoRA parameters while keeping their learned outputs active, then train head to cap.
- CONTRIB_FREEZE: same trajectory; at each positive checkpoint before cap and before freeze, obtain V prediction with adapters off as well as on. C_half=100*(L_off_half-L_on_half)/initial full-V F0 score. C_0=0. At earliest cap/3 and after three positive observations, freeze when BOTH current per-update C slopes are <=25% of their own largest previous nonnegative slope. Includes the initial-to-first slope in reference. If never triggered, train jointly to cap. No off observation after freeze or at cap. Continue head after freezing. Final selection can precede freezing.

Cap/3 and .25 are prespecified screening constants, not statistically calibrated thresholds. Step-length normalization prevents raw gains over unequal intervals being treated as equal slopes. Decreasing/negative contribution can also trigger: E comparison must expose failures. Do not claim an unobserved complete future signal curve after freezing.

## Gates and actual costs

Primary candidate feasibility relative to FULL must hold in EACH seed: mean across six source-condition cells regret<=.25%F0; source mean regret<=.5%F0; sum of actual guarded fit walltime savings>5%. Inference costs reported separately, same test size. Fit walltime includes process/model loading, on/off measurement, checkpoint replay and output; excludes admission waits and E inference. Also report same criteria for ES2 and FIXED_FREEZE, and whether either dominates candidate (no larger macro regret and no more fit time, one strict). Passing vs FULL alone does not establish a useful new signal. Seeds/cells are correlated, no inflated sample size or significance claim.

Freeze is an actual intervention and tail is executed, not an extrapolated timing proxy. Persistent initial trainable-name set defines snapshots and the original frozen-backbone hash. Freezing must not omit LoRA from saved checkpoints. Check adapter weights constant across frozen tail, head changes, adapter activation retained, on/off restores original grad flags, exact selected-V replay. S0 forces freeze at step1 for a 3-step lifecycle check; it is not an outcome observation used to tune policy.

48 fits +50 E predictions +1 lifecycle smoke, serial GPU. Existing admission RAM>=5GiB and commit>=13GiB twice5s; runtime RAM>=5/commit>=6GiB, RSS<=8GiB, GPU<=10500MiB, temp<=85C, job900s. Failures preserved, no threshold relaxation or automatic retry.

## Primary prior art

Adaptive freezing is established: [AFLoRA, ACL2024](https://aclanthology.org/2024.acl-short.16/) proposes a freezing score for low-rank projections with feature-transformation vectors; [FreezeOut](https://arxiv.org/abs/1706.04983) progressively freezes layers. [AdaLoRA, ICLR2023](https://openreview.net/pdf?id=lq62uWRJjiY) adaptively allocates rank budget. This pilot does not implement those full methods or establish superiority to them. Its narrower contribution test is the usefulness of within-trajectory on/off response against ordinary stopping and fixed freezing. [Lightning EarlyStopping documentation](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html) defines patience in validation checks. Novelty and broad publication readiness remain unestablished even if this small gate passes.
