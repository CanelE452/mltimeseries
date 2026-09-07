# Model substitution log

Section 4 of the study contract requires that any change to the model set be
recorded, with its reason, before results are interpreted.

## S1 — `tirex-2-fevbench` dropped; `NX-AI/TiRex-2` becomes the recurrent-family primary

**Original plan.** The recurrent (xLSTM) slot was to be filled by
`NX-AI/TiRex-2-fevbench`, because the TiRex-2 model card advertises it as the
fev-bench decontaminated checkpoint:

> "- [TiRex-2-f](https://huggingface.co/NX-AI/TiRex-2-fevbench): excludes all
> [fev-bench eval datasets](https://huggingface.co/datasets/autogluon/fev_datasets)
> from pretraining, using the same approach as for GiftEval."
> — https://huggingface.co/NX-AI/TiRex-2, retrieved 2026-09-07

Section 8 says to prefer an official benchmark-specific checkpoint where one
exists, so that repository was selected.

**What was found.** The published artifact in that repository is byte-identical
to the flagship checkpoint. Both the Hugging Face API file listing and the local
blob store agree:

| repository | `model.ckpt` size | `model.ckpt` LFS oid | `model-config.yaml` oid |
|---|---|---|---|
| `NX-AI/TiRex-2` | 380,613,375 | `5596fb4bd1ecc4fbf93c5d7d3c9c68bd4e8492b3` | `42345eb8ecba386bd81636bc288742a6e70e8653` |
| `NX-AI/TiRex-2-fevbench` | 380,613,375 | `5596fb4bd1ecc4fbf93c5d7d3c9c68bd4e8492b3` | `42345eb8ecba386bd81636bc288742a6e70e8653` |
| `NX-AI/TiRex-2-gifteval-zs` | 330,093,831 | `1a6342ed027c9cfdb58facdde12780e455b67a4a` | `902617df2d14a0d7ca7e0057b2ae9d54abeef5ca` |
| `NX-AI/TiRex-2-gifteval-pretrain` | 330,093,831 | `d37b46c397b35f2164792b771755772f791df6d0` | `902617df2d14a0d7ca7e0057b2ae9d54abeef5ca` |

The two GIFT-Eval variants really are distinct checkpoints — different size,
different weights, different config — so the vendor's decontamination pipeline
does produce separate artifacts. The fev-bench variant is the exception: as
published on 2026-09-07 it carries the flagship weights.

The first task run under the original plan confirmed the consequence
independently: `tirex-2-fevbench` and `tirex-2-general` returned the identical
SQL score `0.8543413436148345` on `fevbench::uci_air_quality_1H`, to every
digit.

**Substitution.**

- Original model: `NX-AI/TiRex-2-fevbench` (`tirex-2-fevbench`)
- Block reason: `DUPLICATE_OF_GENERAL_CHECKPOINT` — the advertised decontaminated
  weights are not present in the published artifact.
- Replacement: `NX-AI/TiRex-2` (`tirex-2`), same architecture family
  (`recurrent_xlstm`), same parameter count, same inference package.
- Why the comparison purpose survives: the recurrent-architecture slot in the
  primary gap map is about a non-Transformer family competing under the same
  information condition. The flagship checkpoint fills that slot exactly as the
  decontaminated one would have; what changes is not the comparison but the
  contamination label attached to it.

**Contamination consequence.** `tirex-2` cannot be labelled
`CLEAN_BY_OFFICIAL_EXCLUSION` for fev-bench. Its status becomes
`OVERLAP_RISK_UNKNOWN`, with the note that the vendor claims a fev-bench
exclusion but the published checkpoint does not carry it. On fev-bench the
contamination picture across the primary set is therefore:

| model | fev-bench contamination status |
|---|---|
| `chronos-2` | `OVERLAP_RISK_UNKNOWN` (exclusions documented against GIFT-Eval only) |
| `tirex-2` | `OVERLAP_RISK_UNKNOWN` (claimed exclusion not realised in the artifact) |
| `timesfm-3.0` | `CLEAN_BY_OFFICIAL_EXCLUSION` (card states fev-bench overlaps removed) |
| `chronos-2-synth` | `SYNTH_ONLY_DIAGNOSTIC` (no real data at all) |

Two of the three primary models therefore carry unresolved overlap risk. Under
Section 8 that is usable for failure discovery and is explicitly downweighted
for any claim about absolute ranking, and it raises the value of the synthetic
anchor rather than blocking the study.

**Timing.** This substitution was decided from checkpoint identity, not from
scores. One task (`fevbench::uci_air_quality_1H`, TRACK U) had been run when the
duplication was noticed; its `tirex-2-fevbench` cache entry was deleted and no
score influenced the decision — the deciding evidence is the LFS oid table above,
which is independent of any forecast.
