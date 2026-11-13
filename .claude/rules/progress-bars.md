---
paths:
  - "optionpricing/ml/**"
  - "scripts/train_*.py"
  - "optionpricing/hedging/legacy/evaluation.py"
---

# Progress bars for long-running work

Any long-running dataset generation, training, or simulation loop must show a
live progress bar with an accurate ETA (e.g. `tqdm`), driven by actual
completed work rather than by how many inputs were submitted or queued.

- For `joblib.Parallel` jobs: don't wrap the input iterable in `tqdm`
  (`tqdm(args_list)`) — since the list is already materialized, that reports
  submission order and the bar fills instantly, not real progress. Instead
  drive the bar from job completions, e.g. the `tqdm_joblib` context manager
  in `optionpricing/ml/surrogate/chunks.py`.
- For epoch/step training loops: wrap the loop in `tqdm` over
  epochs/batches/episodes and show a running loss/reward/metric in the
  postfix.
