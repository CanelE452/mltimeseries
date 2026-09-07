# Specialist model availability

Section 7 forbids quietly swapping a requested model for another, so every requested
name is checked against the installed AutoGluon registry and the result recorded before
any training runs.

| requested | actual class | family | available | reason if unavailable |
|---|---|---|---|---|
| PatchTST | `PatchTSTModel` | patch transformer | yes | - |
| TiDE | `TiDEModel` | dense encoder | yes | - |
| DLinear | `DLinearModel` | linear decomposition | yes | - |
| DeepAR | `DeepARModel` | autoregressive RNN | yes | - |
| DirectTabular | `DirectTabularModel` | tabular regression | yes | - |

All 5 requested models are supported by autogluon.timeseries 1.6.1, so no
substitution was needed and the minimum of three distinct specialist families is met with room
to spare.

## Foundation models are installed but never used

AutoGluon ships Chronos, Chronos-2 and Toto wrappers, and installing it pulled
`chronos-forecasting` in as a dependency. None of them may appear in this study's specialist
suite: the whole question is whether a *non-foundation* task-trained model can beat the TSFM
envelope, so a Chronos inside the ensemble would answer a different question. The suite is
specified as an explicit hyperparameter dict rather than an AutoGluon preset for exactly this
reason, and `verify` check A18 reads the fitted model list back to confirm none slipped in.

Present in the registry but excluded: ChronosModel, Chronos2Model, TotoModel, Toto2Model.
