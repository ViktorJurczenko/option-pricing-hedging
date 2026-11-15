"""Tests for optionpricing.ml.hedging.generate_hedge_dataset: the bank of hedging scenarios and its generator."""

import hashlib

import numpy as np
import pytest

import optionpricing.ml.hedging.generate_hedge_dataset as gen
from optionpricing.hedging.exercise import daily_boundary
from optionpricing.hedging.simulator import ScenarioBank, simulate_scenarios
from optionpricing.ml.hedging.generate_hedge_dataset import (
    CHUNK_COLUMNS,
    file_sha256,
    label_contracts,
    surrogate_greeks,
)
from optionpricing.ml.surrogate.tree.tree_pricer import PutSurrogate, SurrogateResult
from optionpricing.pricing.european.black_scholes import black_scholes_delta, black_scholes_gamma, black_scholes_price
from tests.helpers import write_put_surrogate

NAN = np.nan


class BlackScholesSurrogate:
    """Stand-in for PutSurrogate whose answers are the European put's, per unit of strike, so every value the
    generator stores can be recomputed independently at the day and moneyness it should come from."""

    def price(self, m, T, r, sigma):
        args = (np.asarray(m), 1.0, np.asarray(T), np.asarray(r), np.asarray(sigma))
        price = black_scholes_price(*args, "put")
        return SurrogateResult(price, black_scholes_delta(*args, "put"), black_scholes_gamma(*args), None)


def hand_bank():
    """Two contracts of 3 and 5 days at K = 100, one path each, NaN past expiry."""
    return ScenarioBank(
        spot=np.array([[100.0, 98.0, 96.0, 97.0, NAN, NAN], [105.0, 107.0, 104.0, 102.0, 103.0, 101.0]]),
        contract=np.array([0, 1]),
        days=np.array([3, 5]),
        sigma=np.array([0.2, 0.3]),
        risk_free_rate=np.array([0.05, 0.03]),
        moneyness=np.array([1.0, 1.05]),
        strike=100.0,
    )


# surrogate_greeks


def test_greeks_are_read_at_each_days_moneyness_and_time_left_in_shares_and_per_dollar():
    bank = hand_bank()
    delta, gamma = surrogate_greeks(BlackScholesSurrogate(), bank, tau=np.array([3, 5]))
    for row, k in [(0, 0), (0, 2), (1, 0), (1, 4)]:
        # day k of a contract of `days` days has days - k days left, read at m = S_k / K
        T = (bank.days[row] - k) / 365
        args = (bank.spot[row, k] / 100, 1.0, T, bank.risk_free_rate[row], bank.sigma[row])
        assert delta[row, k] == pytest.approx(black_scholes_delta(*args, "put"), rel=1e-12)
        # gamma per unit of strike is K d2V/dS2, so per dollar it is that over K
        assert gamma[row, k] == pytest.approx(black_scholes_gamma(*args) / 100, rel=1e-12)


def test_greeks_are_nan_from_tau_on_and_finite_before():
    delta, gamma = surrogate_greeks(BlackScholesSurrogate(), hand_bank(), tau=np.array([1, 4]))
    for values in (delta, gamma):
        np.testing.assert_array_equal(np.isfinite(values), [[1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]])


def test_greeks_of_puts_all_exercised_at_once_are_all_nan():
    delta, gamma = surrogate_greeks(BlackScholesSurrogate(), hand_bank(), tau=np.array([0, 0]))
    assert np.isnan(delta).all() and np.isnan(gamma).all() and delta.shape == (2, 6)


# label_contracts


@pytest.fixture(scope="module")
def surrogate(tmp_path_factory):
    return PutSurrogate.load(write_put_surrogate(tmp_path_factory.mktemp("surrogate") / "surrogate.pth"))


@pytest.fixture(scope="module")
def small_bank():
    # short contracts, so every day after the first week reaches the surrogate's tree route too
    return simulate_scenarios(
        6, 3, seed=4, days_range=(5, 12), sigma_range=(0.15, 0.3), rate_range=(0.02, 0.06), moneyness_range=(0.9, 1.1)
    )


def test_a_block_of_contracts_is_labelled_as_in_the_whole_bank(surrogate, small_bank):
    # The generator labels blocks in separate jobs and concatenates them: that must change nothing.
    whole = label_contracts(small_bank, surrogate, "tree", None, tree_steps=50)
    first = label_contracts(small_bank.select_contracts(0, 4), surrogate, "tree", None, tree_steps=50)
    second = label_contracts(small_bank.select_contracts(4, 6), surrogate, "tree", None, tree_steps=50)
    for col in CHUNK_COLUMNS:
        # The net runs in float32, and its matrix products round differently with the batch size: a block and the
        # whole bank agree to float32's precision, not bit for bit.
        np.testing.assert_allclose(np.concatenate([first[col], second[col]]), whole[col], rtol=1e-6, err_msg=col)


def test_premium_is_the_surrogates_price_at_t0_in_dollars(small_bank):
    # the stand-in prices like Black-Scholes, so the premium must be the European put at S_0, the full maturity
    premium = label_contracts(small_bank, BlackScholesSurrogate(), "never", None, tree_steps=50)["premium"]
    b = small_bank
    expected = black_scholes_price(b.moneyness * 100, 100.0, b.maturity, b.risk_free_rate, b.sigma, "put")
    np.testing.assert_allclose(premium, expected, rtol=1e-12)  # same formula, per unit of strike then x K


def test_premium_of_the_real_surrogate_is_never_below_the_payoff(surrogate, small_bank):
    # PutSurrogate's payoff floor holds even for an untrained net, so the hedger never sells below intrinsic value
    premium = label_contracts(small_bank, surrogate, "never", None, tree_steps=50)["premium"]
    assert np.all(premium >= np.maximum(100 - small_bank.moneyness * 100, 0))


def test_boundary_is_stored_for_every_holder_and_tau_follows_the_holder(surrogate, small_bank):
    never = label_contracts(small_bank, surrogate, "never", None, tree_steps=50)
    tree = label_contracts(small_bank, surrogate, "tree", None, tree_steps=50)
    np.testing.assert_array_equal(never["boundary"], daily_boundary(small_bank, nb_steps=50))
    np.testing.assert_array_equal(never["boundary"], tree["boundary"])
    np.testing.assert_array_equal(never["tau"], small_bank.days[small_bank.contract])
    assert np.all(tree["tau"] <= never["tau"])


# file_sha256


def test_file_sha256_is_the_digest_of_the_file_bytes(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert file_sha256(path) == hashlib.sha256(b"abc").hexdigest()


# main


def tiny_args(tmp_path, surrogate_path, *extra):
    return [
        "--n_contracts", "5", "--paths_per_contract", "2", "--days_range", "5", "12", "--tree_steps", "50",
        "--chunk_size", "2", "--nb_jobs", "1", "--surrogate", str(surrogate_path), "--output_dir", str(tmp_path),
        *extra,
    ]  # fmt: skip


@pytest.fixture
def surrogate_path(tmp_path):
    return write_put_surrogate(tmp_path / "surrogate.pth")


def test_main_saves_the_bank_its_labels_and_its_settings_and_removes_the_chunks(tmp_path, surrogate_path):
    gen.main(tiny_args(tmp_path, surrogate_path, "--split", "val"))
    data = np.load(tmp_path / "hedge_val.npz")
    bank = simulate_scenarios(5, 2, seed=gen.SPLIT_SEEDS["val"], **{**gen.DEFAULT_RANGES, "days_range": (5, 12)})
    # the blocks of 2 contracts (the last one of 1) reassemble in scenario order
    expected = label_contracts(bank, PutSurrogate.load(surrogate_path), "tree", None, tree_steps=50)
    np.testing.assert_array_equal(data["spot"], bank.spot.astype(np.float32))
    for col in CHUNK_COLUMNS:
        np.testing.assert_allclose(data[col], expected[col], rtol=1e-6, err_msg=col)  # float32 for daily arrays
    assert data["split"] == "val" and data["seed"] == 2 and data["exercise"] == "tree" and np.isnan(data["shift"])
    assert data["surrogate_sha256"] == file_sha256(surrogate_path)
    np.testing.assert_array_equal(data["days_range"], (5, 12))
    assert not (tmp_path / "hedge_chunks_val").exists()


def test_main_stores_the_late_holders_shift(tmp_path, surrogate_path):
    gen.main(tiny_args(tmp_path, surrogate_path, "--exercise", "late", "--shift", "0.1", "--seed", "9"))
    data = np.load(tmp_path / "hedge_train.npz")
    assert data["exercise"] == "late" and data["shift"] == 0.1 and data["seed"] == 9


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--exercise", "late"), "needs a shift"),
        (("--shift", "0.1"), "only used by method 'late'"),
        (("--tree_steps", "51"), "even number"),
        (("--chunk_size", "0"), "chunk_size must be >= 1"),
        (("--sigma_range", "0.3", "0.1"), "sigma_range must be"),
    ],
)
def test_bad_settings_fail_before_the_config_is_written(tmp_path, surrogate_path, extra, message):
    with pytest.raises(ValueError, match=message):
        gen.main([*tiny_args(tmp_path, surrogate_path), *extra])  # argparse keeps the last value of a flag
    assert not (tmp_path / "hedge_chunks_train").exists()


def test_a_file_that_is_not_a_surrogate_checkpoint_fails_before_the_config_is_written(tmp_path):
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(Exception):  # noqa: B017 - torch.load raises its own unpickling error type here
        gen.main(tiny_args(tmp_path, bad))
    assert not (tmp_path / "hedge_chunks_train").exists()


def test_interrupted_main_resumes_with_the_same_flags_and_refuses_other_ones(tmp_path, surrogate_path, monkeypatch):
    args = tiny_args(tmp_path, surrogate_path)

    def interrupted(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(gen, "generate_hedge_chunks", interrupted)
    with pytest.raises(KeyboardInterrupt):
        gen.main(args)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="other settings"):
        gen.main([*args, "--tree_steps", "60"])
    gen.main(args)
    assert np.load(tmp_path / "hedge_train.npz")["tau"].shape == (10,)


def test_generate_hedge_chunks_resumes_by_redoing_only_the_missing_block(
    tmp_path, surrogate_path, small_bank, monkeypatch
):
    run = (small_bank, tmp_path, 2, str(surrogate_path), "tree", None, 50, 1)
    gen.generate_hedge_chunks(*run)
    full = gen.load_hedge_chunks(tmp_path, 3)
    (tmp_path / "chunk_00001.npz").unlink()
    # n_jobs=1 runs the jobs in this process, through the patched name, so a spy sees what is redone
    labelled = []
    original = gen.generate_chunk
    monkeypatch.setattr(gen, "generate_chunk", lambda path, *a: (labelled.append(path.name), original(path, *a)))
    gen.generate_hedge_chunks(*run)
    assert labelled == ["chunk_00001.npz"]
    resumed = gen.load_hedge_chunks(tmp_path, 3)
    for col in CHUNK_COLUMNS:
        np.testing.assert_array_equal(resumed[col], full[col])
