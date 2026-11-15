"""Tests for optionpricing.ml.hedging.hedge_model: the band network, its inputs and its feature statistics."""

import math

import pytest
import torch
import torch.nn.functional as F

from optionpricing.ml.hedging import hedge_model as hm

STRIKE = 100.0


def make_batch(n=3, n_days=4, seed=0):
    """A batch shaped like `HedgeDataset` items, with distinct seeded values so a swapped row shows."""
    g = torch.Generator().manual_seed(seed)
    spot = 80 + 40 * torch.rand(n, n_days, generator=g, dtype=torch.float64)
    return {
        "spot": spot,
        "delta": -torch.rand(n, n_days, generator=g, dtype=torch.float64),
        "gamma": 0.05 * torch.rand(n, n_days, generator=g, dtype=torch.float64),
        "boundary": 60 + 10 * torch.rand(n, n_days, generator=g, dtype=torch.float64),
        "time_left": (n_days - torch.arange(n_days, dtype=torch.float64)).expand(n, n_days) / 365,
        "sigma": 0.15 + 0.15 * torch.rand(n, generator=g, dtype=torch.float64),
        "rate": 0.02 + 0.04 * torch.rand(n, generator=g, dtype=torch.float64),
        "tau": torch.tensor([n_days - 1, 1, 0][:n] + [n_days - 1] * max(0, n - 3)),
    }


class FakeBank:
    """The part of `HedgeDataset` that `fit_feature_stats` reads: `strike`, `len` and batch indexing."""

    def __init__(self, batch, strike=STRIKE):
        self.batch, self.strike = batch, strike

    def __len__(self):
        return len(self.batch["tau"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.batch.items()}


def make_network(n_features=6, **kwargs):
    """A network with neutral stats (mean 0, std 1); 6 features is `FEATURES`, 7 adds the boundary."""
    return hm.BandNetwork(torch.zeros(n_features), torch.ones(n_features), **kwargs)


def one_date(n=5, delta=-0.4, seed=0):
    """One hedge date of inputs for `n` scenarios, with the given delta in its row."""
    x = torch.rand(n, len(hm.FEATURES), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    x[:, hm.DELTA] = delta
    return x


# hedge_inputs


def test_hedge_inputs_puts_each_feature_in_its_named_row():
    batch = make_batch()
    out = hm.hedge_inputs(batch, STRIKE, boundary_input=True)
    names = hm.feature_names(True)
    assert out.shape == (3, len(names), 4)
    row = {name: out[:, i] for i, name in enumerate(names)}
    torch.testing.assert_close(row["log_moneyness"], torch.log(batch["spot"] / STRIKE))
    torch.testing.assert_close(row["time_left"], batch["time_left"])
    torch.testing.assert_close(row["sigma"], batch["sigma"][:, None].expand(3, 4))
    torch.testing.assert_close(row["rate"], batch["rate"][:, None].expand(3, 4))
    torch.testing.assert_close(row["delta"], batch["delta"])
    torch.testing.assert_close(row["gamma"], batch["gamma"])
    torch.testing.assert_close(row["boundary_distance"], torch.log(batch["spot"] / batch["boundary"]))


def test_hedge_inputs_leaves_the_boundary_out_unless_asked():
    out = hm.hedge_inputs(make_batch(), STRIKE)
    assert out.shape[1] == len(hm.FEATURES) == 6
    assert hm.feature_names(True)[-1] == hm.BOUNDARY_FEATURE


@pytest.mark.parametrize("strike", [0.0, -1.0, math.inf, math.nan])
def test_hedge_inputs_rejects_a_strike_that_is_not_positive_and_finite(strike):
    with pytest.raises(ValueError, match="strike must be positive"):
        hm.hedge_inputs(make_batch(), strike)


# fit_feature_stats


def test_feature_stats_count_only_the_hedge_dates_before_each_tau():
    batch = make_batch()  # tau = 3, 1, 0: days 0-2 of scenario 0 and day 0 of scenario 1
    batch["spot"][2] = 1e6  # scenario 2 is exercised at once: if counted, it would wreck the mean
    mean, std = hm.fit_feature_stats(FakeBank(batch), chunk_size=2)  # 2 chunks: the sums must add up across them
    inputs = hm.hedge_inputs(batch, STRIKE)
    rows = torch.cat([inputs[0, :, :3], inputs[1, :, :1]], dim=1)  # (n_features, 4 hedge dates)
    torch.testing.assert_close(mean, rows.mean(dim=1))
    torch.testing.assert_close(std, rows.std(dim=1, correction=0))


def test_a_constant_feature_gets_std_one_so_standardizing_only_centres_it():
    batch = make_batch()
    batch["sigma"][:] = 0.2  # one volatility for every contract
    mean, std = hm.fit_feature_stats(FakeBank(batch))
    i = hm.FEATURES.index("sigma")
    assert mean[i].item() == pytest.approx(0.2) and std[i].item() == 1.0


def test_feature_stats_refuse_a_bank_with_no_hedge_date():
    batch = make_batch()
    batch["tau"][:] = 0
    with pytest.raises(ValueError, match="no hedge date"):
        hm.fit_feature_stats(FakeBank(batch))


def test_feature_stats_reject_a_chunk_size_below_one():
    with pytest.raises(ValueError, match="chunk_size"):
        hm.fit_feature_stats(FakeBank(make_batch()), chunk_size=0)


# BandNetwork


@pytest.mark.parametrize("activation", sorted(hm.ACTIVATIONS))
def test_untrained_network_is_the_delta_hedge_to_within_its_initial_half_width(activation):
    # The zero output weights cut the hidden layers off at the start, so this holds for every activation.
    torch.manual_seed(0)
    model = make_network(init_logit=-5.0, activation=activation)
    half_width = F.softplus(torch.tensor(-5.0, dtype=torch.float64))
    position = torch.tensor([-2.0, -0.4, 0.0, 1.0, -0.41], dtype=torch.float64)
    out = model(one_date(delta=-0.4), position)
    # Far from delta, the position is clamped to the band's edge on its side; inside, it is held.
    expected = torch.tensor([-0.4 - half_width, -0.4, -0.4 + half_width, -0.4 + half_width, -0.4 - half_width])
    torch.testing.assert_close(out, expected.to(torch.float64))


def test_a_position_inside_the_band_is_held_exactly_so_no_cost_is_paid():
    model = make_network(init_logit=5.0)  # half-width softplus(5) ~ 5 shares: every position below is inside
    position = torch.tensor([-1.3, -0.4, 0.0, 0.7, -0.123456789], dtype=torch.float64)
    out = model(one_date(), position)
    assert torch.equal(out, position)  # bit for bit, or hedge_pnl would charge a fixed fee on a tiny trade


def test_the_band_always_contains_delta_whatever_the_weights():
    torch.manual_seed(0)
    model = make_network(depth=2, width=8)
    for p in model.parameters():  # random output weights too, so the logits differ per input
        torch.nn.init.normal_(p, std=3.0)
    x = one_date(n=200, seed=1)
    x[:, hm.DELTA] = -torch.rand(200, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    lower, upper = model.band(x, torch.zeros(200, dtype=torch.float64))
    assert torch.all(lower <= x[:, hm.DELTA]) and torch.all(x[:, hm.DELTA] <= upper)
    assert torch.any(upper - x[:, hm.DELTA] != x[:, hm.DELTA] - lower)  # the band can be asymmetric


def test_a_clamped_position_carries_a_gradient_to_the_network():
    # Training learns only through the edges: a held position does not depend on the weights.
    model = make_network()
    position = torch.full((5,), 0.5, dtype=torch.float64)  # above every band: clamped to the upper edge
    model(one_date(), position).sum().backward()
    assert model.net[-1].bias.grad[1] != 0  # the upper logit
    assert model.net[-1].bias.grad[0] == 0  # the lower edge played no part


def test_the_network_standardizes_its_inputs_with_the_stored_stats():
    # A network whose stats match a shift of the inputs gives the same band as one fed the unshifted inputs.
    torch.manual_seed(0)
    plain = make_network(depth=1, width=4)
    torch.nn.init.normal_(plain.net[-1].weight)
    shift, scale = torch.arange(6.0), torch.arange(1.0, 7.0)
    shifted = hm.BandNetwork(shift, scale, depth=1, width=4)
    shifted.net.load_state_dict(plain.net.state_dict())
    x, position = one_date(), torch.zeros(5, dtype=torch.float64)
    x_shifted = x * scale.double() + shift.double()
    x_shifted[:, hm.DELTA] = x[:, hm.DELTA]  # the band is centred on the raw delta, so keep it the same...
    plain_x = x.clone()
    plain_x[:, hm.DELTA] = (x[:, hm.DELTA] - shift[hm.DELTA]) / scale[hm.DELTA]  # ...and give plain its standard
    lower_p, upper_p = plain.band(plain_x, position)
    lower_s, upper_s = shifted.band(x_shifted, position)
    # same logits, so the same half-widths around each one's delta (float32 network: ~1e-6 relative)
    torch.testing.assert_close(upper_s - x_shifted[:, hm.DELTA], upper_p - plain_x[:, hm.DELTA], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(x_shifted[:, hm.DELTA] - lower_s, plain_x[:, hm.DELTA] - lower_p, rtol=1e-5, atol=1e-6)


def test_save_and_load_give_the_same_policy_and_extras(tmp_path):
    torch.manual_seed(0)
    model = hm.BandNetwork(
        torch.arange(7.0), torch.arange(1.0, 8.0), depth=2, width=8, init_logit=-3.0, activation="tanh"
    )
    for p in model.net.parameters():
        torch.nn.init.normal_(p)
    model.save(tmp_path / "p.pth", config={"lr": 1e-3}, bank_settings={"seed": 1, "range": (0.1, 0.2)})
    loaded, extra = hm.BandNetwork.load(tmp_path / "p.pth")
    x = torch.rand(4, 7, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    position = torch.zeros(4, dtype=torch.float64)
    torch.testing.assert_close(loaded(x, position), model(x, position), rtol=0, atol=0)
    assert loaded.hparams == model.hparams and loaded.n_features == 7
    assert isinstance(loaded.net[1], torch.nn.Tanh)  # the activation is rebuilt, not just recorded
    assert extra == {"config": {"lr": 1e-3}, "bank_settings": {"seed": 1, "range": (0.1, 0.2)}}


def test_load_refuses_a_file_that_is_not_a_hedge_policy(tmp_path):
    torch.save({"state_dict": {}}, tmp_path / "other.pth")
    with pytest.raises(ValueError, match="not a hedge policy"):
        hm.BandNetwork.load(tmp_path / "other.pth")


@pytest.mark.parametrize(
    ("mean", "std", "kwargs", "match"),
    [
        (torch.zeros(6), torch.ones(5), {}, "same length"),
        (torch.zeros(4), torch.ones(4), {}, "same length"),  # too short to hold the delta row
        (torch.zeros(6), torch.zeros(6), {}, "std must be > 0"),
        (torch.zeros(6), torch.ones(6), {"depth": 0}, "depth and width"),
        (torch.zeros(6), torch.ones(6), {"width": 0}, "depth and width"),
        (torch.zeros(6), torch.ones(6), {"init_logit": math.inf}, "init_logit"),
        (torch.zeros(6), torch.ones(6), {"activation": "sigmoid"}, "activation must be one of"),
    ],
)
def test_band_network_rejects_invalid_arguments(mean, std, kwargs, match):
    with pytest.raises(ValueError, match=match):
        hm.BandNetwork(mean, std, **kwargs)


def test_band_rejects_inputs_with_the_wrong_number_of_features():
    model = make_network(n_features=7)  # built with the boundary input
    with pytest.raises(ValueError, match=r"\(nb_scenarios, 7\)"):
        model(one_date(), torch.zeros(5, dtype=torch.float64))
