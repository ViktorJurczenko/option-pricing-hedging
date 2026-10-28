"""Factories and helpers shared across test files (fixtures would go in a
conftest.py; none are shared yet)."""
import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = str(ROOT / "main.py")

# Small enough that run() simulates and fits LSM in well under a second.
SMALL_SIMULATION = dict(nb_predictions=50, maturity=0.1, sampling=20, max_degree=2)


def make_fake_lsm(simulation, exercise_time, option_type="call", strike=100.0, delta=None, gamma=None):
    """Stand-in for a fitted LongstaffSchwartz with the attributes HedgingEnv
    reads. Greeks default to zero and have one row fewer than the simulation
    (none at maturity), like the real model's."""
    simulation = np.asarray(simulation, dtype=float)
    greeks_shape = (simulation.shape[0] - 1, simulation.shape[1])
    return SimpleNamespace(
        option_type=option_type,
        strike=strike,
        simulation=simulation,
        delta=np.zeros(greeks_shape) if delta is None else np.asarray(delta, dtype=float),
        gamma=np.zeros(greeks_shape) if gamma is None else np.asarray(gamma, dtype=float),
        exercise_time=np.asarray(exercise_time),
    )


def two_path_call_lsm(delta=None):
    """Two call paths (100 -> 105 -> 110 and 100 -> 95 -> 90), both run to maturity."""
    return make_fake_lsm([[100.0, 100.0], [105.0, 95.0], [110.0, 90.0]], [2, 2], delta=delta)


def make_state(delta, position=-0.3, spot=1.1, t=0.4, gamma=0.7, batch=None):
    """State in HedgingEnv's layout [S/K, t/T, delta, gamma, position], or
    `batch` copies of it as rows. The defaults are deliberately non-zero, so a
    wrong index changes results."""
    row = np.array([spot, t, delta, gamma, position])
    return row if batch is None else np.tile(row, (batch, 1))


def make_parser(script):
    """An ArgumentParser with a scripts/ module's flags."""
    parser = argparse.ArgumentParser()
    script.add_arguments(parser)
    return parser


def parse_script_args(script, **overrides):
    """A scripts/ module's arguments: its parser's defaults, a small
    simulation, then `overrides`. A misspelt override raises instead of being
    silently ignored by run()."""
    args = make_parser(script).parse_args([])
    for name, value in {**SMALL_SIMULATION, "checkpoint_path": "unused.pth", **overrides}.items():
        if not hasattr(args, name):
            raise AttributeError(f"{script.__name__} has no --{name} flag")
        setattr(args, name, value)
    return args


def run_python(*args, timeout=120):
    """Run `python <args>` from the project root, capturing output. The
    timeout catches a hang; slower callers (the ML pipeline) pass a larger one."""
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace",  # tqdm's partial-block characters are not decodable as cp1252
                          timeout=timeout, cwd=ROOT)


def run_cli(*args, timeout=120):
    """Run `python main.py <args>` from the project root, capturing output."""
    return run_python(MAIN, *args, timeout=timeout)


def write_tree_dataset(out_dir, n_rows=128, seed=3):
    """A real generate_tree_dataset file of `n_rows` puts in `out_dir`, returns
    its path. 50-step trees keep it fast; the labels' accuracy is the
    pricer's tests' job."""
    import optionpricing.ml.surrogate.tree.generate_tree_dataset as gen  # imported here: most test files never need it
    gen.main(["--n_rows", str(n_rows), "--chunk_size", str(min(n_rows, 64)), "--nb_steps", "50",
              "--nb_jobs", "1", "--seed", str(seed), "--output_dir", str(out_dir)])
    return Path(out_dir) / "tree_put_dataset.npz"
