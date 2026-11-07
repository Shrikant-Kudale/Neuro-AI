# Audit of the original repository and what was done about it

Findings from reading the earlier version of this project against the published paper. Kept here so the research record is accurate.

| Area | Finding | Resolution in this version |
|---|---|---|
| "Transformer" | `module5_train.py` trained a scikit-learn `GradientBoostingClassifier`; the NumPy attention code in `module4_model.py` had random, untrained weights and fed an sklearn MLP. | A real PyTorch Transformer (`research/neuroai/models.py`), trained with the optimiser, schedule and early stopping stated in the paper. |
| "CNN" | A `RandomForestClassifier`. | A 3-layer Conv1d network. |
| Training curves | Generated from the final accuracy by a function named `generate_mock_training_log`. | Logged per epoch during real training. |
| Attention maps | Built by hand to put weight on tokens 7–10. | Extracted from the trained network (last layer and rollout). |
| PAC figure | Hard-coded bar heights. | Computed from simulated recordings. |
| Phase estimate | `arcsin` of the normalised signal. | Hilbert-transform phase and envelope. |
| Feature shape | Comments said 38 features per token while the code used 41. | 15 × 41 documented and enforced. |
| Behavioural-only ablation | Paper parameters make behavioural features almost perfectly separable, contradicting the reported 61.3%. | Reported as measured; "hard" profile added with class overlap and a shared confounder. |
| Parameter count | Paper states ≈412 k; the stated 4-block architecture has ≈545 k (three blocks ≈ 412 k). | Four blocks kept; true count reported. |
| Coupling strength | A literal depth of 0.18 produces a measured MI indistinguishable from the estimator noise floor. | Documented generative gain (`COUPLING_GAIN`) so measured MI is in a physiological range. |

The headline figures of the paper (87.4%, 0.93) are not reproduced by the corrected pipeline; the measured values are in the README and on the Results page. Consider a correction or erratum to the journal if the published tables are cited further.
