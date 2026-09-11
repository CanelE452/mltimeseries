# Study32: conditional future-update utility, not current adapter contribution

User approved the study31 follow-up and asked what evidence could support a paper. This is one bounded mechanism diagnostic, not a new adaptation policy, fresh-source evaluation, or publication claim. Preserve study31 sources/data/results. No E archive is opened.

## Fixed choices and alternatives

Chronos2, MLP533, attention LoRA rank8, two learning rates1e-4, AdamW weight_decay0, clip1, microbatch4/batch8, threads2, all twelve study31 source/condition/seed cells are retained to isolate subsequent update decisions. Choosing only favorable BDG2 SPREAD30 would be post-outcome selection. New models/sources are reserved for later validation, not added to this diagnostic. Raw data are not regenerated.

The original fit archive's V[0:14] is S (checkpoint selection and current-signal measurement), V[16:30] is D (diagnostic outcome). Two skipped origins separate target intervals. Contexts overlap and both intervals have already been observed in study31: D is not an independent or untouched holdout. Train normalization and sample indices are unchanged. This diagnostic cannot establish new-source transfer or infer an effective sample size of24 from dependent forks.

Replay one joint trajectory per cell to the original cap180 (FULL90) or60 (subsets). Store model parameters, Adam states and CPU/CUDA RNG states at cap/3 and2cap/3. All positive original schedule points before cap get adapter-on/off predictions; signals use S only. Verify every joint prediction against the corresponding saved study31 FULL trajectory. No learning-rate or threshold search.

At each fork continue to the SAME cap with the SAME precomputed mini-batch suffix:

- JOINT: original shared continuation, not a second independently initialized model.
- HEAD_ONLY: restore prefix, freeze LoRA weights while leaving its forward path active; retain head Adam state, perform ordinary head-only gradient clipping.
- MASKED_UPDATE: restore prefix, retain full backward and global clipping, then set LoRA gradients to None before AdamW.step. This holds the initial clipping computation equal to JOINT while suppressing LoRA weight updates. It is an offline control, NOT a compute-saving method.
- STOP: the prefix predictions with no further learning, requiring no additional trajectory.

The MASKED_UPDATE control addresses the possibility that global gradient clipping changes head updates when adapter gradients disappear. It is the less obvious alternative to attributing every HEAD_ONLY difference to representation learning. If MASKED and HEAD_ONLY differ, report this coupling rather than calling the whole difference a pure LoRA-weight effect. Head trajectories can diverge after treatment; estimates remain conditional on this model, initialization, optimizer and training suffix.

Retain full original-prefix checkpoint candidates for S-based selection in every branch; strict S improvement, earliest tie. Report fixed-final outcomes separately from globally selected outcomes. D never chooses checkpoints, training budgets, branches or signals. S signals: C=100*(S_off-S_on)/S_F0, current C slope per update, and preceding S improvement per update. D values are written with predictions but not used in training or selection.

## Frozen estimands and descriptive screening

For each cell/fork, normalize by its initial D F0 loss:

U_final =100*(D_HEAD_final-D_JOINT_final)/D_F0 (positive favors further LoRA updates).
U_selected uses the separately S-selected HEAD and JOINT checkpoints.
J_gain=100*(D_STOP-D_JOINT_final)/D_F0; H_gain analogous.
U_masked=100*(D_MASKED_final-D_JOINT_final)/D_F0.
Selection benefit per arm=100*(D_final-D_selected)/D_F0.

A +/-0.25%F0 band is a prespecified descriptive negligible-difference screen, not a statistical equivalence test or paper-acceptance threshold. Report all24 forks, source means, sign agreement across two seeds, and whether positive current contribution / positive C slope / positive preceding S improvement correctly indicates U_final>0.25. Compare constant continue and constant freeze accuracy on the same development cases; exclude negligible cases from nontrivial sign accuracy and always report counts. Any correlation is descriptive with dependent cells and only two source families. No learned selector, significance inflation or E tuning.

Competing explanations: (A) current contribution fails to predict marginal future benefit; (B) apparent gains mainly arise from checkpoint selection; (C) clipping-induced head-update changes explain a portion. Outcomes can support several. Stop after this diagnostic: do not tune C thresholds on D. If no reproducible condition-dependent remaining utility exists, close the on/off-based allocation direction. If it exists, the next research requirement is a cheap, prospective signal that beats fixed freezing and nearby literature methods on separate development/evaluation sources.

## Integrity and resources

One lifecycle smoke plus12 main GPU jobs, sequential, each<=900s. Existing admission RAM>=5GiB and commit>=13GiB twice5s; running RAM>=5/commit>=6GiB, RSS<=8GiB, GPU<=10500MiB/temp<=85C. No automatic retry of failed paths or threshold relaxation. Snapshot membership fixed at196 trainable names; original native weights immutable. Every restored fork matches model/optimizer/RNG hashes; head optimizer state retained. All branch-start predictions equal prefix; frozen LoRA unchanged; HEAD/MASKED head changes; JOINT replay smoke exactly matches shared trajectory. Raw predictions saved for independent loss/selection audit. Diagnostic walltime includes all controls and must not be sold as a deployment speedup.

## Prior evidence and paper requirements

[AFLoRA (ACL2024)](https://aclanthology.org/2024.acl-short.16/) already uses adaptive freezing. [AdaLoRA (ICLR2023)](https://openreview.net/forum?id=lq62uWRJjiY) adapts rank budget. [Kumar et al. (ICLR2022)](https://openreview.net/forum?id=UYneFzXSJWh) studies feature distortion and head/backbone adaptation under distribution shift; it is motivation for competing explanations, not proof of our time-series mechanism.

For a methods paper: a reproducible practically material failure of strong baselines, a measurable mechanism with controlled alternatives, a prospectively usable novel intervention, gains over same-capacity head/ordinary LoRA/ES2/fixed freezing and closest implemented prior under fair tuning and actual total cost, and transfer to additional source families/backbones with uncertainty and negative cases. No universal required accuracy percentage or dataset count guarantees acceptance. Roughly20% savings versus FULL is not a new-method contribution here because the simple fixed baseline already achieves it. An analysis paper could instead establish a robust new finding with strong controlled evidence; this one diagnostic alone is insufficient.
