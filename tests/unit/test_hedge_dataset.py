"""Tests for optionpricing.ml.hedging.hedge_dataset: loading a hedge bank and refusing banks that do not match."""

import math

import numpy as np
import pytest
import torch

import optionpricing.ml.hedging.generate_hedge_dataset as gen
from optionpricing.hedging.baselines import baseline_inputs, delta_hedge
from optionpricing.hedging.pnl import hedge_pnl
from optionpricing.ml.hedging.hedge_dataset import SHARED_SETTINGS, HedgeDataset, check_compatible
from tests.helpers import write_put_surrogate


def write_bank(out_dir, surrogate_path, *extra):
    """A real generate_hedge_dataset bank of 4 contracts x 2 paths of 5 to 12 days, with 50-step trees."""
    gen.main(
        [
            "--n_contracts", "4", "--paths_per_contract", "2", "--days_range", "5", "12", "--tree_steps", "50",
            "--nb_jobs", "1", "--surrogate", str(surrogate_path), "--output_dir", str(out_dir), *extra,
        ]
    )  # fmt: skip
    split = extra[extra.index("--split") + 1] if "--split" in extra else "train"
    return out_dir / f"hedge_{split}.npz"


@pytest.fixture(scope="module")
def banks(tmp_path_factory):
    """Paths of a train bank and a validation bank generated with the same settings."""
    out = tmp_path_factory.mktemp("banks")
    surrogate = write_put_surrogate(out / "surrogate.pth")
    return write_bank(out, surrogate), write_bank(out, surrogate, "--split", "val")


@pytest.fixture(scope="module")
def train(banks):
    return HedgeDataset(banks[0])


# loading


def test_a_scenario_carries_its_own_path_and_its_contracts_values(banks, train):
    raw = np.load(banks[0])
    item = train[3]  # scenario 3 is the second path of contract 1
    np.testing.assert_array_equal(item["spot"].numpy(), raw["spot"][3])
    np.testing.assert_array_equal(item["boundary"].numpy(), raw["boundary"][1])
    assert item["premium"].item() == pytest.approx(raw["premium"][1], rel=1e-6)
    assert item["sigma"].item() == pytest.approx(raw["sigma"][1], rel=1e-6)
    assert item["rate"].item() == pytest.approx(raw["risk_free_rate"][1], rel=1e-6)
    assert item["tau"].item() == raw["tau"][3]


def test_time_left_counts_down_to_zero_at_expiry_and_is_nan_after(train):
    days = int(train.days[train.contract[0]])
    time_left = train[0]["time_left"].numpy()
    np.testing.assert_allclose(time_left[: days + 1], (days - np.arange(days + 1)) / 365, rtol=1e-6)
    assert np.isnan(time_left[days + 1 :]).all()
    # spot is padded the same way, so the two agree on where a scenario ends
    np.testing.assert_array_equal(np.isnan(time_left), np.isnan(train[0]["spot"].numpy()))


def test_a_batch_index_gives_the_same_rows_as_single_items(train):
    batch = train[torch.tensor([5, 0, 2])]
    for row, i in enumerate([5, 0, 2]):
        for key, value in train[i].items():
            torch.testing.assert_close(batch[key][row], value, equal_nan=True, msg=key)


def test_dtype_sets_the_floating_tensors_and_tau_stays_an_integer(banks):
    item = HedgeDataset(banks[0], dtype=torch.float64)[0]
    assert item["spot"].dtype == item["delta"].dtype == item["premium"].dtype == torch.float64
    assert item["tau"].dtype == torch.int64


def test_settings_and_shift_come_back_as_python_values(train):
    assert train.settings["split"] == "train" and train.settings["days_range"] == (5, 12)
    assert train.shift is None and math.isnan(train.settings["shift"]) and train.strike == 100.0


def test_every_day_the_hedge_reads_is_finite_so_the_bank_feeds_hedge_pnl(train):
    # hedge_pnl reads the features of days before tau only, and a NaN there would make the P&L NaN
    b = train[torch.arange(len(train))]
    inputs = baseline_inputs(b["spot"], b["delta"], b["gamma"], b["time_left"], b["rate"][:, None])
    outcome = hedge_pnl(delta_hedge, inputs, b["spot"], b["tau"], b["premium"], train.strike, b["rate"])
    assert torch.isfinite(outcome.pnl).all()


def test_a_file_that_is_not_a_hedge_bank_is_refused(tmp_path):
    path = tmp_path / "other.npz"
    np.savez(path, spot=np.zeros((1, 2)), teacher="tree")
    with pytest.raises(ValueError, match="is not a hedge bank"):
        HedgeDataset(path)


# check_compatible


def test_banks_with_the_same_settings_and_distinct_seeds_are_accepted(banks, train):
    assert HedgeDataset(banks[1], reference=train.settings).settings["seed"] == gen.SPLIT_SEEDS["val"]


def test_a_bank_with_the_same_seed_is_refused(banks, train):
    with pytest.raises(ValueError, match="both banks have seed 1"):
        HedgeDataset(banks[0], reference=train.settings)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("surrogate_sha256", "0" * 64),
        ("exercise", "late"),
        ("shift", 0.1),
        ("days_range", (92, 365)),
        ("tree_steps", 1000),
        ("strike", 50.0),
    ],
)
def test_a_bank_with_another_shared_setting_is_refused_naming_it(train, key, value):
    other = {**train.settings, key: value, "seed": 99}
    with pytest.raises(ValueError, match=f"other settings.*'{key}'"):
        check_compatible(train.settings, other)


def test_every_shared_setting_is_stored_in_a_bank(train):
    # a setting compared but never stored would raise KeyError instead of comparing
    assert set(SHARED_SETTINGS) <= set(train.settings)
