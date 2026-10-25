"""
Helpers for a chunked, resumable dataset generation, shared by both
generators (`generate_tree_dataset` and `legacy/generate_lsm_dataset`):
finished chunks are saved one file each, a `config.json` refuses a resume
with other settings, and a progress bar follows joblib's completed jobs.
"""
import json
import os
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np

@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager that makes a tqdm bar advance on each completed joblib
    batch, instead of on each submitted argument (the default `tqdm(args_list)`
    wrapping reports submission order, not completion, so it doesn't reflect
    actual progress/ETA for parallel work).
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

def chunk_file(chunk_dir: Path, idx: int) -> Path:
    return chunk_dir / f"chunk_{idx:05d}.npz"

def n_chunks_for(n_total: int, chunk_size: int) -> int:
    return -(-n_total // chunk_size)

def prepare_chunk_dir(chunk_dir: Path, config: dict) -> None:
    """
    Creates the chunk directory, or checks that the chunks already in it were
    produced with the same generation settings. Resuming with different
    settings would silently mix labels from two different datasets.
    """
    config_path = chunk_dir / "config.json"
    if config_path.exists():
        stored = json.loads(config_path.read_text())
        if stored != config:
            diff = {k: (stored.get(k), config.get(k)) for k in config if stored.get(k) != config[k]}
            raise ValueError(f"{chunk_dir} holds chunks generated with other settings ((stored, requested): {diff}). "
                             "Use the same flags to resume, or delete the directory to start over.")
    else:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2))

def save_chunk(path: Path, **arrays: np.ndarray) -> None:
    """Saves `arrays` under their names, atomically: a run killed mid-write
    leaves no half-written chunk that a resume would trust."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
