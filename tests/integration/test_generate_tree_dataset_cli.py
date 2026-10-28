"""
`python -m optionpricing.ml.surrogate.tree.generate_tree_dataset` as a user runs it: real worker
processes (the unit tests run in-process with one job), so an import that
only works in the parent, or chunks written out of order, shows up here.
"""

import numpy as np

from tests.helpers import run_python


def _generate(out_dir, *extra):
    result = run_python(
        "-m",
        "optionpricing.ml.surrogate.tree.generate_tree_dataset",
        "--n_rows",
        "128",
        "--nb_steps",
        "100",
        "--nb_jobs",
        "2",
        "--output_dir",
        str(out_dir),
        *extra,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return np.load(out_dir / "tree_put_dataset.npz")


def test_generator_runs_as_a_module_and_its_chunking_does_not_change_the_dataset(tmp_path):
    a = _generate(tmp_path / "a", "--chunk_size", "16")
    b = _generate(tmp_path / "b", "--chunk_size", "64")
    for col in ("m", "T", "r", "sigma", "z", "price", "european_price", "delta", "gamma", "exercised"):
        np.testing.assert_array_equal(a[col], b[col], err_msg=col)
    assert a["price"].shape == (128,) and np.isfinite(a["price"]).all()
    # American >= European, up to the tree's error (~1e-5 K at 100 steps)
    assert (a["price"] - a["european_price"]).min() > -1e-4
    assert not (tmp_path / "a" / "tree_chunks").exists()
