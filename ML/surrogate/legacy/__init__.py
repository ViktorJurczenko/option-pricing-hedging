"""
The legacy LSM surrogate pipeline, kept as a library: Longstaff-Schwartz
labelled calls and puts (generate_lsm_dataset), their dataset class
(lsm_dataset), the original net (lsm_model.LegacyMLP), its training
(lsm_training) and evaluation (lsm_evaluation). No CLI command uses it:
`train-ml`/`test-ml` run the tree surrogate (ML/surrogate/tree/tree_training.py,
tree_evaluation.py). Kept for the ablation against the old architecture
and for rerunning the LSM comparison.
"""
