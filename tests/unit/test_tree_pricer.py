import numpy as np
import pytest
import torch

from optionpricing.ml.surrogate.legacy.lsm_model import LegacyMLP
from optionpricing.ml.surrogate.tree.generate_tree_dataset import DOMAIN, map_to_domain, sobol_points
from optionpricing.ml.surrogate.tree.tree_dataset import FeatureStats, build_features
from optionpricing.ml.surrogate.tree.tree_model import MLP
from optionpricing.ml.surrogate.tree.tree_pricer import EXERCISE_MIN_RATE, PutSurrogate
from optionpricing.pricing.american.batch_binomial_tree import binomial_put_batch
from optionpricing.pricing.european.black_scholes import black_scholes_delta, black_scholes_gamma, black_scholes_price


def _stats():
    p = map_to_domain(sobol_points(256, 4, seed=0))
    return FeatureStats.fit(build_features(p["m"], p["T"], p["r"], p["sigma"]))


def _surrogate(zero_net=False, dtype=torch.float64, **mlp_kwargs):
    """A seeded, untrained surrogate on the generator's domain (float64 so finite differences are precise)."""
    torch.manual_seed(0)
    model = MLP(**{"depth": 2, "width": 16, **mlp_kwargs}).to(dtype)
    if zero_net:  # premium identically 0: the surrogate is plain Black-Scholes on the net route
        torch.nn.init.zeros_(model.net[-1].weight)
        torch.nn.init.zeros_(model.net[-1].bias)
    return PutSurrogate(model, _stats(), DOMAIN)


# inside the domain: z = ln m / (sigma sqrt T) in [-3.5, 3], m in [0.4, 2.5]
M_IN = np.array([0.8, 0.95, 1.0, 1.05, 1.2])
T_IN, R_IN, S_IN = 0.5, 0.03, 0.25


# --- the net route ---


def test_net_route_is_black_scholes_plus_the_nets_premium():
    sur = _surrogate()
    res = sur.price(M_IN, T_IN, R_IN, S_IN)
    assert (res.route == "net").all()
    x = sur.stats.apply(build_features(M_IN, T_IN, R_IN, S_IN))
    premium = sur.model(x).squeeze(-1).detach().numpy()
    bs = black_scholes_price(M_IN, 1.0, T_IN, R_IN, S_IN, "put")
    np.testing.assert_allclose(res.price, bs + premium, rtol=0, atol=1e-14)
    assert np.abs(premium).min() > 1e-3  # the check above would pass trivially with a ~0 net


def test_zero_premium_net_reproduces_black_scholes_greeks():
    m = M_IN[1:]  # from 0.95 up: there the European put is above the payoff, so no floor applies
    res = _surrogate(zero_net=True).price(m, T_IN, R_IN, S_IN)
    args = (m, 1.0, T_IN, R_IN, S_IN)
    assert (black_scholes_price(*args, "put") > np.maximum(1 - m, 0)).all() and (res.route == "net").all()
    np.testing.assert_allclose(res.price, black_scholes_price(*args, "put"), atol=1e-15)
    np.testing.assert_allclose(res.delta, black_scholes_delta(*args, "put"), atol=1e-15)
    np.testing.assert_allclose(res.gamma, black_scholes_gamma(*args), atol=1e-12)


def test_net_price_below_the_payoff_in_the_money_is_floored_at_one_minus_m():
    # deep in the money the European put is worth less than exercising now;
    # with no premium the net's price is that European price
    m, T, r = 0.8, 2.0, 0.06
    assert black_scholes_price(m, 1.0, T, r, S_IN, "put") < 1 - m
    res = _surrogate(zero_net=True).price(m, T, r, S_IN)
    assert res.route == "payoff_floor"
    assert (res.price, res.delta, res.gamma) == (1 - m, -1.0, 0.0)


def test_negative_net_price_out_of_the_money_is_floored_at_zero():
    sur = _surrogate(zero_net=True)
    torch.nn.init.constant_(sur.model.net[-1].bias, -1.0)  # premium -1: the price would be negative
    res = sur.price(np.array([1.05, 1.2]), T_IN, R_IN, S_IN)
    assert (res.route == "payoff_floor").all()
    np.testing.assert_array_equal(res.price, 0.0)
    np.testing.assert_array_equal(res.delta, 0.0)
    np.testing.assert_array_equal(res.gamma, 0.0)


def test_payoff_floor_leaves_prices_above_the_payoff_untouched():
    sur = _surrogate(zero_net=True)
    torch.nn.init.constant_(sur.model.net[-1].bias, 0.05)  # a positive premium: always above the payoff
    res = sur.price(M_IN, T_IN, R_IN, S_IN)
    assert (res.route == "net").all()
    np.testing.assert_allclose(res.price, black_scholes_price(M_IN, 1.0, T_IN, R_IN, S_IN, "put") + 0.05, atol=1e-15)


def test_net_route_delta_and_gamma_match_finite_differences_of_the_price():
    sur = _surrogate()
    h = 1e-4
    res = sur.price(M_IN, T_IN, R_IN, S_IN)
    up, down = sur.price(M_IN + h, T_IN, R_IN, S_IN).price, sur.price(M_IN - h, T_IN, R_IN, S_IN).price
    # central differences: truncation O(h^2) ~ 1e-8 times the third/fourth
    # derivatives (O(10) here), rounding 1e-16 / h^2 = 1e-8 for gamma
    np.testing.assert_allclose(res.delta, (up - down) / (2 * h), atol=1e-7)
    np.testing.assert_allclose(res.gamma, (up - 2 * res.price + down) / h**2, atol=1e-5)


def test_float32_net_gives_float64_outputs_close_to_its_float64_copy():
    sur32 = _surrogate(dtype=torch.float32)
    sur64 = PutSurrogate(MLP(**sur32.model.config).double(), sur32.stats, DOMAIN)
    sur64.model.load_state_dict({k: v.double() for k, v in sur32.model.state_dict().items()})
    r32, r64 = sur32.price(M_IN, T_IN, R_IN, S_IN), sur64.price(M_IN, T_IN, R_IN, S_IN)
    assert r32.price.dtype == np.float64
    # float32 rounding (~1e-7 relative) on O(1) activations
    np.testing.assert_allclose(r32.price, r64.price, atol=1e-5)
    np.testing.assert_allclose(r32.gamma, r64.gamma, atol=1e-3)


# --- routing outside the domain ---


def test_far_out_of_the_money_goes_to_black_scholes():
    m = np.exp(3.2 * S_IN * np.sqrt(T_IN))  # z = 3.2 > 3, m = 1.25 inside the cap
    res = _surrogate().price(m, T_IN, R_IN, S_IN)
    assert res.route == "black_scholes"
    args = (m, 1.0, T_IN, R_IN, S_IN)
    assert res.price == black_scholes_price(*args, "put")
    assert res.delta == black_scholes_delta(*args, "put")
    assert res.gamma == black_scholes_gamma(*args)


@pytest.mark.parametrize("m", [np.exp(-3.6 * S_IN * np.sqrt(T_IN)), 0.3])  # z < -3.5, inside and below the m cap
def test_far_in_the_money_with_a_positive_rate_is_exercised_today(m):
    res = _surrogate().price(m, T_IN, R_IN, S_IN)
    assert res.route == "exercised"
    assert (res.price, res.delta, res.gamma) == (1.0 - m, -1.0, 0.0)


@pytest.mark.parametrize(
    "m, T, r, sigma",
    [
        (np.exp(-3.6 * S_IN * np.sqrt(T_IN)), T_IN, 0.001, S_IN),  # deep ITM but r < 0.005: may not be exercised
        (2.6, 1.0, R_IN, 0.5),  # m above the cap with z = 1.9 inside its range
        (1.0, 6.0, R_IN, S_IN),  # T beyond 5 years
        (1.0, 1 / 104, R_IN, S_IN),  # T below 1 week
        (1.0, T_IN, R_IN, 0.7),  # sigma above 0.6
        (1.0, T_IN, -0.01, S_IN),  # negative rate
        (2.0, 6.0, R_IN, 0.05),  # z > 3 but T outside: the closed form is not trusted there
    ],
)
def test_other_queries_outside_the_domain_go_to_the_tree(m, T, r, sigma):
    res = _surrogate().price(m, T, r, sigma)
    assert res.route == "tree"
    tree = binomial_put_batch(m, T, r, sigma)
    assert (res.price, res.delta, res.gamma) == (tree.price, tree.delta, tree.gamma)


@pytest.mark.parametrize(
    "m, T, r, sigma",
    [
        (1.0, DOMAIN["T_range"][0], R_IN, S_IN),
        (1.0, DOMAIN["T_range"][1], R_IN, S_IN),
        (1.0, T_IN, R_IN, DOMAIN["sigma_range"][0]),
        (1.0, T_IN, R_IN, DOMAIN["sigma_range"][1]),
        (1.0, T_IN, DOMAIN["r_range"][0], S_IN),
        (1.0, T_IN, DOMAIN["r_range"][1], S_IN),
        (DOMAIN["m_range"][0], 1.0, R_IN, 0.5),  # z = ln 0.4 / 0.5 = -1.8, inside z's range
        (DOMAIN["m_range"][1], 1.0, R_IN, 0.5),  # z = +1.8
    ],
)
def test_parameter_and_moneyness_bounds_are_inclusive(m, T, r, sigma):
    assert _surrogate().price(m, T, r, sigma).route == "net"


@pytest.mark.parametrize("m, beyond", [(1.2, "black_scholes"), (0.8, "exercised")])
def test_z_bounds_are_inclusive(m, beyond):
    # a z exactly on the bound can't be built from m (exp then log rounds),
    # so the bound is set to the query's own z, computed as price() does
    z = np.log(m) / (S_IN * np.sqrt(T_IN))
    side = 1 if beyond == "black_scholes" else 0
    z_range = [-3.5, 3.0]
    z_range[side] = z
    sur = PutSurrogate(_surrogate().model, _stats(), {**DOMAIN, "z_range": tuple(z_range)})
    assert sur.price(m, T_IN, R_IN, S_IN).route == "net"
    assert sur.price(m * (1.01 if side else 0.99), T_IN, R_IN, S_IN).route == beyond


def test_exercised_rule_includes_the_minimum_rate():
    m = np.exp(-3.6 * S_IN * np.sqrt(T_IN))
    assert _surrogate().price(m, T_IN, EXERCISE_MIN_RATE, S_IN).route == "exercised"
    assert _surrogate().price(m, T_IN, EXERCISE_MIN_RATE * 0.99, S_IN).route == "tree"


def test_empty_query_gives_empty_outputs():
    res = _surrogate().price(np.array([]), T_IN, R_IN, S_IN)
    assert res.price.shape == res.route.shape == (0,)


def test_mixed_batch_routes_each_query_and_keeps_the_broadcast_shape():
    m = np.array([[1.0, 1.0], [np.exp(3.2 * S_IN * np.sqrt(T_IN)), np.exp(-3.6 * S_IN * np.sqrt(T_IN))]])
    sigma = np.array([[S_IN, 0.7], [S_IN, S_IN]])  # 0.7: above the domain's sigma, so the tree
    res = _surrogate().price(m, T_IN, R_IN, sigma)
    assert res.price.shape == res.delta.shape == res.gamma.shape == res.route.shape == (2, 2)
    assert res.route.tolist() == [["net", "tree"], ["black_scholes", "exercised"]]
    alone = _surrogate().price(m[0, 0], T_IN, R_IN, S_IN)  # a scalar query gives 0-d arrays
    assert alone.price.shape == ()
    assert alone.price == res.price[0, 0]


@pytest.mark.parametrize(
    "m, T, r, sigma",
    [
        (np.nan, T_IN, R_IN, S_IN),
        (-1.0, T_IN, R_IN, S_IN),
        # m = 0 and m = inf were answered silently, as exercised (z = -inf)
        # and as a NaN Black-Scholes price (z = +inf), before m was checked
        (0.0, T_IN, R_IN, S_IN),
        (np.inf, T_IN, R_IN, S_IN),
        (1.0, np.nan, R_IN, S_IN),
        (1.0, np.inf, R_IN, S_IN),
        (1.0, 0.0, R_IN, S_IN),
        (1.0, T_IN, np.nan, S_IN),
        (1.0, T_IN, np.inf, S_IN),
        (1.0, T_IN, R_IN, np.nan),
        (1.0, T_IN, R_IN, 0.0),
    ],
)
def test_invalid_queries_reach_the_tree_and_raise(m, T, r, sigma):
    with pytest.raises(ValueError):
        _surrogate().price(m, T, r, sigma)


# --- construction, save and load ---


def test_missing_domain_key_raises():
    domain = {k: v for k, v in DOMAIN.items() if k != "z_range"}
    with pytest.raises(ValueError, match="z_range"):
        PutSurrogate(MLP(), _stats(), domain)


def test_save_then_load_gives_the_same_prices_and_architecture(tmp_path):
    sur = _surrogate(dtype=torch.float32, activation="tanh", output="softplus")
    path = tmp_path / "sub" / "surrogate.pth"
    sur.save(path)
    loaded = PutSurrogate.load(path)
    assert loaded.model.config == sur.model.config
    assert loaded.domain == sur.domain
    m = np.array([0.5, 1.0, 1.3, 3.0])
    for a, b in zip(sur.price(m, T_IN, R_IN, S_IN)[:3], loaded.price(m, T_IN, R_IN, S_IN)[:3]):
        np.testing.assert_array_equal(a, b)


def test_load_refuses_a_legacy_state_dict(tmp_path):
    path = tmp_path / "legacy.pth"
    torch.save(LegacyMLP().state_dict(), path)
    with pytest.raises(ValueError, match="not a PutSurrogate checkpoint"):
        PutSurrogate.load(path)


def test_load_refuses_stats_for_other_features(tmp_path):
    path = tmp_path / "s.pth"
    _surrogate(dtype=torch.float32).save(path)
    ckpt = torch.load(path, weights_only=True)
    ckpt["feature_stats"]["names"] = ["log_m"] * len(ckpt["feature_stats"]["names"])
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="feature statistics"):
        PutSurrogate.load(path)
