"""Deep Hedging learns a band that beats the delta hedge under costs, on banks whose deltas are exact.

The holder never exercises early (`--exercise never`), so each put is European and held to expiry. Its premium,
delta and gamma are then overwritten with the Black-Scholes ones, so the untrained network (the delta hedge) is
the exact zero-cost hedge, and any gain must come from the band saving costs. This keeps the test independent of
a trained surrogate checkpoint.
"""

import numpy as np
import pytest

from optionpricing.ml.hedging.hedge_training import HedgeTrainConfig, train_hedger
from optionpricing.pricing.european.black_scholes import black_scholes_delta, black_scholes_gamma, black_scholes_price
from tests.helpers import write_hedge_bank, write_put_surrogate

N_CONTRACTS = 1000  # ~50 paths in a 95% tail


def black_scholes_bank(out_dir, surrogate, split):
    """A bank of European puts of 20 to 40 days with exact premiums and Greeks, returns its path."""
    path = write_hedge_bank(
        out_dir, surrogate, "--split", split, "--exercise", "never", "--n_contracts", str(N_CONTRACTS),
        "--paths_per_contract", "1", "--days_range", "20", "40",
    )  # fmt: skip
    raw = dict(np.load(path))
    strike, days_per_year = float(raw["strike"]), float(raw["days_per_year"])
    contract, days = raw["contract"], raw["days"]
    day = np.arange(raw["spot"].shape[1])
    time_left = (days[contract][:, None] - day) / days_per_year
    sigma, rate = raw["sigma"][contract][:, None], raw["risk_free_rate"][contract][:, None]
    alive = time_left > 0  # the hedge dates; expiry and the padding after it stay NaN
    spot = np.where(alive, raw["spot"], strike)  # any valid spot where the Greeks are not wanted
    t = np.where(alive, time_left, 1.0)
    raw["delta"] = np.where(alive, black_scholes_delta(spot, strike, t, rate, sigma, "put"), np.nan).astype(np.float32)
    raw["gamma"] = np.where(alive, black_scholes_gamma(spot, strike, t, rate, sigma), np.nan).astype(np.float32)
    raw["premium"] = black_scholes_price(
        raw["moneyness"] * strike, strike, days / days_per_year, raw["risk_free_rate"], raw["sigma"], "put"
    )
    np.savez(path, **raw)
    return path


@pytest.fixture(scope="module")
def banks(tmp_path_factory):
    out = tmp_path_factory.mktemp("bs_banks")
    surrogate = write_put_surrogate(out / "surrogate.pth")
    return black_scholes_bank(out, surrogate, "train"), black_scholes_bank(out, surrogate, "val")


def test_the_band_network_beats_the_delta_hedge_at_a_one_percent_cost(banks):
    config = HedgeTrainConfig(epochs=15, batch_size=500, proportional=0.01, lr=1e-2)
    result = train_hedger(*banks, config)
    delta, trained = result.history[0], result.history[result.best_epoch]
    # Measured on these banks: costs 1.94 -> 1.22 dollars and CVaR loss 4.20 -> 3.04 dollars, best at epoch 13
    # (docs/results/hedging-training.md, "Training checks"). The thresholds keep most of that gain as margin, so
    # they test that the band learned, not the exact numbers.
    assert trained["valid_mean_costs"] < 0.8 * delta["valid_mean_costs"]
    assert trained["valid_score"] < delta["valid_score"] - 0.25
