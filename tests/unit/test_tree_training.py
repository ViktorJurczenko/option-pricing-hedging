import math

import numpy as np
import pytest
import torch

import optionpricing.ml.surrogate.tree.tree_training as training
from optionpricing.ml.surrogate.tree.generate_tree_dataset import DOMAIN
from optionpricing.ml.surrogate.tree.tree_dataset import TreePutDataset, build_features
from optionpricing.ml.surrogate.tree.tree_model import MLP
from optionpricing.ml.surrogate.tree.tree_pricer import PutSurrogate
from optionpricing.ml.surrogate.tree.tree_training import (
    LossScales,
    ShapePoints,
    TrainConfig,
    evaluate_labeled,
    initial_model,
    labeled_loss,
    premium_and_delta,
    shape_violations,
    train_surrogate,
)
from optionpricing.pricing.european.black_scholes import black_scholes_dT, black_scholes_vega
from tests.helpers import write_tree_dataset

# tiny net and run: these tests check the loop's mechanics, not accuracy
TINY = dict(depth=1, width=8, batch_size=16, epochs=4, seed=0)


@pytest.fixture(scope="module")
def tree_file(tmp_path_factory):
    return write_tree_dataset(tmp_path_factory.mktemp("tree"))


@pytest.fixture(scope="module")
def data(tree_file):
    return TreePutDataset(tree_file)


def _net(scale=1.0):
    """A seeded float64 net; `scale` multiplies the output layer, so its derivatives can dominate BS's."""
    torch.manual_seed(0)
    model = MLP(depth=2, width=16).double()
    with torch.no_grad():
        model.net[-1].weight.mul_(scale)
    return model


# --- TrainConfig ---


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"epochs": 0},
        {"patience": 0},
        {"shape_points": 0},
        {"lr": 0.0},
        {"lr_min": -1e-6},
        {"lr": 1e-4, "lr_min": 1e-3},
        {"valid_ratio": 0.0},
        {"valid_ratio": 1.0},
        {"lambda_delta": -1.0},
        {"lambda_shape": -0.1},
        {"lbfgs_steps": -1},
    ],
)
def test_train_config_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_train_config_defaults_follow_the_plan():
    c = TrainConfig()
    assert (c.depth, c.width, c.activation, c.output) == (4, 128, "silu", "linear")
    assert (c.lr, c.lr_min, c.batch_size, c.epochs) == (1e-3, 1e-5, 1024, 400)
    assert (c.lambda_delta, c.lambda_shape, c.lbfgs_steps) == (1.0, 0.0, 0)


# --- loss pieces ---


def test_loss_scales_are_the_targets_population_variances_and_never_zero():
    s = LossScales.fit(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 0.0, 6.0]))
    assert s.premium == pytest.approx(1.0) and s.delta == pytest.approx(8.0)
    assert LossScales.fit(torch.tensor([2.0]), torch.tensor([5.0, 5.0])) == (1.0, 1.0)


def test_net_delta_is_the_derivative_in_m_through_the_dataset_features(data):
    # premium_and_delta's chain rule with the dataset's features_dm must equal
    # a finite difference of the net composed with build_features and the stats
    model = _net()
    x, x_dm = data.features[:20].double(), data.features_dm[:20].double()
    _, delta = premium_and_delta(model, x, x_dm, create_graph=False)
    p = {k: v[:20] for k, v in data.params.items()}
    h = 1e-6
    f = lambda m: model(data.stats.apply(build_features(m, p["T"], p["r"], p["sigma"]))).squeeze(-1)
    fd = (f(p["m"] + h) - f(p["m"] - h)) / (2 * h)
    # x is the dataset's float32 features: ~1e-7 relative input rounding
    torch.testing.assert_close(delta.detach(), fd.detach(), atol=1e-5, rtol=1e-5)


def test_labeled_loss_is_the_variance_scaled_sum_of_the_two_mses(data):
    model = _net().float()
    batch = (data.features[:32], data.features_dm[:32], data.premium[:32], data.premium_dm[:32])
    scales = LossScales(premium=2e-4, delta=3e-2)
    pred, pred_delta = premium_and_delta(model, batch[0], batch[1])
    premium_mse = float(torch.mean((pred - batch[2]) ** 2).detach())
    delta_mse = float(torch.mean((pred_delta - batch[3]) ** 2).detach())
    loss, p_mse, d_mse = labeled_loss(model, batch, scales, lambda_delta=2.5)
    assert float(loss.detach()) == pytest.approx(premium_mse / 2e-4 + 2.5 * delta_mse / 3e-2, rel=1e-5)
    assert (float(p_mse), float(d_mse)) == pytest.approx((premium_mse, delta_mse), rel=1e-5)
    loss0, _, d0 = labeled_loss(model, batch, scales, lambda_delta=0.0)
    assert float(loss0.detach()) == pytest.approx(premium_mse / 2e-4, rel=1e-5) and math.isnan(float(d0))


def test_evaluate_labeled_reports_the_two_terms_of_the_loss_apart(data):
    # the checkpoint goes by their sum, but one term can improve while the other worsens, so both are reported
    model = _net().float()
    rows = (data.features, data.features_dm, data.premium, data.premium_dm)
    scales = LossScales(premium=2e-4, delta=3e-2)
    m = evaluate_labeled(model, rows, scales, lambda_delta=2.5)
    assert m["premium_term"] == pytest.approx(m["premium_rmse"] ** 2 / 2e-4, rel=1e-6)
    assert m["delta_term"] == pytest.approx(2.5 * m["delta_rmse"] ** 2 / 3e-2, rel=1e-6)
    assert m["loss"] == pytest.approx(m["premium_term"] + m["delta_term"], rel=1e-12)
    assert evaluate_labeled(model, rows, scales, lambda_delta=0.0)["delta_term"] == 0.0


def test_delta_term_trains_the_weights_through_the_input_gradient(data):
    # create_graph=True must let the delta loss reach the parameters
    model = _net().float()
    batch = (data.features[:32], data.features_dm[:32], torch.zeros(32), data.premium_dm[:32])
    pred, _ = premium_and_delta(model, batch[0], batch[1])
    with torch.no_grad():
        batch = (batch[0], batch[1], pred.detach(), batch[3])  # premium term exactly 0
    loss, _, _ = labeled_loss(model, batch, LossScales(1.0, 1.0), lambda_delta=1.0)
    loss.backward()
    assert model.net[0].weight.grad.abs().sum() > 0


# --- shape penalties ---

# inside the domain (z from -1.8 to 1.9), so PutSurrogate prices them with the net
POINTS = dict(
    m=np.array([0.7, 0.9, 1.0, 1.1, 1.3]),
    T=np.array([1.0, 0.5, 1.0, 2.0, 0.3]),
    r=np.array([0.05, 0.03, 0.0, 0.07, 0.02]),
    sigma=np.array([0.2, 0.3, 0.15, 0.4, 0.25]),
)


def test_zero_premium_violates_only_the_maturity_condition_where_the_european_put_falls_with_T(data):
    model = _net()
    torch.nn.init.zeros_(model.net[-1].weight)
    torch.nn.init.zeros_(model.net[-1].bias)
    m, T, r, sigma = (np.array([0.5, 0.6, 1.0, 1.2]), np.full(4, 1.0), np.full(4, 0.06), np.full(4, 0.2))
    v = shape_violations(model, data.stats, ShapePoints.from_params(m, T, r, sigma)).detach().numpy()
    dT = black_scholes_dT(m, 1.0, T, r, sigma, "put")
    assert (dT[:2] < 0).all() and (dT[2:] > 0).all()  # deep ITM: the European put loses value with T
    np.testing.assert_allclose(v[:, 2], np.maximum(-T * dT, 0), atol=1e-15)
    np.testing.assert_array_equal(np.delete(v, 2, axis=1), 0)


def test_shape_violations_are_the_relu_of_the_full_price_derivatives(data):
    # a net scaled up 50x breaks every condition somewhere; its violations
    # must equal relu of finite differences of the PutSurrogate price, which
    # adds the BS put by its own closed form
    model = _net(scale=50.0)
    with torch.no_grad():  # a constant premium changes no derivative, and lifts every price above the payoff floor
        model.net[-1].bias.add_(10.0)
    sur = PutSurrogate(model, data.stats, DOMAIN)
    p = POINTS
    assert (sur.price(**p).route == "net").all()
    v = shape_violations(model, data.stats, ShapePoints.from_params(**p)).detach().numpy()
    price = lambda **kw: sur.price(**{**p, **kw}).price
    h = 1e-5
    d_m = (price(m=p["m"] + h) - price(m=p["m"] - h)) / (2 * h)
    d_mm = (price(m=p["m"] + 1e-3) - 2 * price() + price(m=p["m"] - 1e-3)) / 1e-6
    d_T = (price(T=p["T"] + h) - price(T=p["T"] - h)) / (2 * h)
    d_s = (price(sigma=p["sigma"] + h) - price(sigma=p["sigma"] - h)) / (2 * h)
    # the m terms are scaled by w = sigma sqrt(T): price changes over a one-sigma move (an unscaled
    # convexity term outweighed the labeled loss ~3000x, BUGS.md "Shape penalty's convexity term ...")
    w = p["sigma"] * np.sqrt(p["T"])
    expected = np.maximum(
        np.stack([w * p["m"] * d_m, -((w * p["m"]) ** 2) * d_mm, -p["T"] * d_T, -p["sigma"] * d_s], 1), 0
    )
    assert (expected > 0).any(0).all(), "every condition must be broken somewhere, or the check is vacuous"
    # step-size truncation: ~h^2 times third derivatives of a 50x net (O(1e3))
    np.testing.assert_allclose(v, expected, atol=1e-3 * np.abs(expected).max())


def test_shape_points_follow_the_domain_and_the_seed():
    a = ShapePoints.sample(np.random.default_rng(5), 256, DOMAIN)
    b = ShapePoints.sample(np.random.default_rng(5), 256, DOMAIN)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)
    for name, key in (("m", "m_range"), ("T", "T_range"), ("r", "r_range"), ("sigma", "sigma_range")):
        col = getattr(a, name).numpy()
        assert DOMAIN[key][0] <= col.min() and col.max() <= DOMAIN[key][1]
    np.testing.assert_allclose(
        a.bs_vega.numpy(), black_scholes_vega(a.m.numpy(), 1.0, a.T.numpy(), a.r.numpy(), a.sigma.numpy())
    )


# --- initial_model ---


@pytest.mark.parametrize("width", [8, 256])
@pytest.mark.parametrize("seed", [0, 1])
def test_softplus_output_starts_at_the_mean_premium(data, width, seed):
    # B-036: PyTorch's default bias term made the softplus net start at ~0.69 per unit of
    # strike against a mean premium of ~0.012 (premium term ~500); at width 256 the
    # overshoot pushed every output into the softplus's flat region and training stopped
    rows = (data.features, data.features_dm, data.premium, data.premium_dm)
    scales = LossScales.fit(data.premium, data.premium_dm)
    model = initial_model(TrainConfig(width=width, output="softplus", seed=seed), data.premium)
    assert model.net[-2].bias.item() == pytest.approx(math.log(math.expm1(float(data.premium.mean()))))
    _, premium_mse, _ = labeled_loss(model, rows, scales, lambda_delta=0.3)
    # a constant prediction at the mean has MSE = var(premium), i.e. a premium term of 1;
    # the small output weights add a varying part, 0.998-1.055 over widths 8-256 and seeds 0-2 here
    assert float(premium_mse) / scales.premium == pytest.approx(1.0, abs=0.1)


def test_default_bias_term_would_start_the_softplus_output_hundreds_of_times_too_high(data):
    # the size of B-036's bad start, which the bias term removes: a constant prediction c
    # has premium term 1 + (c - mean)^2 / var, and PyTorch's bias term gives c ~ softplus(0) = ln 2
    rows = (data.features, data.features_dm, data.premium, data.premium_dm)
    scales = LossScales.fit(data.premium, data.premium_dm)
    torch.manual_seed(0)
    default = MLP(width=256, output="softplus")
    _, premium_mse, _ = labeled_loss(default, rows, scales, lambda_delta=0.3)
    expected = 1 + (math.log(2) - float(data.premium.mean())) ** 2 / scales.premium
    # the random bias term moves c within softplus(+-1/16) = 0.66-0.73, so the term within ~15% of ln 2's
    assert float(premium_mse) / scales.premium == pytest.approx(expected, rel=0.15)
    assert float(premium_mse) / scales.premium > 100


def test_linear_output_keeps_pytorchs_default_initialization(data):
    # the linear output never had the problem, and its recorded ablation runs must reproduce
    config = TrainConfig(width=16, output="linear", seed=5)
    torch.manual_seed(5)
    default = MLP(depth=config.depth, width=16)
    for x, y in zip(initial_model(config, data.premium).state_dict().values(), default.state_dict().values()):
        assert torch.equal(x, y)


@pytest.mark.parametrize("premium", [torch.zeros(10), torch.full((10,), -1e-6), torch.zeros(0)])
def test_softplus_start_needs_a_positive_mean_premium(premium):
    # no bias term gives softplus(bias) <= 0, so there is no mean to start at
    with pytest.raises(ValueError, match="positive mean premium"):
        initial_model(TrainConfig(width=8, output="softplus"), premium)
    initial_model(TrainConfig(width=8, output="linear"), premium)  # the linear output needs no mean


def test_training_starts_the_softplus_output_at_the_trained_rows_mean_premium(tree_file):
    # a learning rate too small to move any weight leaves the starting bias term in place,
    # so this checks that train_surrogate starts from initial_model on its trained rows
    config = TrainConfig(**{**TINY, "output": "softplus", "epochs": 1, "lr": 1e-12, "lr_min": 1e-12})
    model = train_surrogate(tree_file, config).surrogate.model
    train, _ = _split(tree_file, config)
    expected = math.log(math.expm1(float(train[2].mean())))
    # float32 rounding of a value near -4.4 is ~5e-7; the few Adam steps move it by ~1e-12 each
    assert model.net[-2].bias.item() == pytest.approx(expected, abs=1e-6)


# --- train_surrogate ---


def _split(tree_file, config):
    """The rows train_surrogate trains and validates on, rebuilt independently."""
    data = TreePutDataset(tree_file, is_train=True, split_ratio=config.split_ratio, seed=config.seed)
    perm = torch.randperm(len(data), generator=torch.Generator().manual_seed(config.seed))
    n_valid = int(len(data) * config.valid_ratio)
    rows = (data.features, data.features_dm, data.premium, data.premium_dm)
    return tuple(t[perm[n_valid:]] for t in rows), tuple(t[perm[:n_valid]] for t in rows)


def test_training_is_reproducible_and_returns_the_best_validation_weights(tree_file, tmp_path):
    config = TrainConfig(**TINY)
    ckpt = tmp_path / "s.pth"
    a = train_surrogate(tree_file, config, checkpoint_path=ckpt)
    b = train_surrogate(tree_file, config)
    assert a.history == b.history
    for x, y in zip(a.surrogate.model.state_dict().values(), b.surrogate.model.state_dict().values()):
        assert torch.equal(x, y)
    train, valid = _split(tree_file, config)
    best = evaluate_labeled(a.surrogate.model, valid, LossScales.fit(train[2], train[3]), config.lambda_delta)
    assert best["loss"] == pytest.approx(min(h["valid_loss"] for h in a.history), rel=1e-6)
    loaded = PutSurrogate.load(ckpt)
    for x, y in zip(loaded.model.state_dict().values(), a.surrogate.model.state_dict().values()):
        assert torch.equal(x, y)


def test_training_lowers_the_validation_loss(tree_file):
    res = train_surrogate(tree_file, TrainConfig(**{**TINY, "epochs": 30, "lr": 1e-2}))
    assert res.history[-1]["valid_loss"] < 0.5 * res.history[0]["valid_loss"]


def test_learning_rate_follows_the_cosine_down_to_lr_min(tree_file):
    res = train_surrogate(tree_file, TrainConfig(**{**TINY, "lr": 1e-2, "lr_min": 1e-4, "patience": 100}))
    lrs = [h["lr"] for h in res.history]
    assert lrs == sorted(lrs, reverse=True)
    assert lrs[-1] == pytest.approx(1e-4)


def test_early_stopping_ends_a_run_that_stops_improving(tree_file):
    # lr = lr_min ~0: the weights never move, so epoch 2 does not beat epoch 1, and the rate is at
    # its minimum from the start, so that epoch already counts
    res = train_surrogate(tree_file, TrainConfig(**{**TINY, "epochs": 50, "lr": 1e-30, "lr_min": 1e-30, "patience": 1}))
    assert len(res.history) == 2


def test_early_stopping_does_not_count_epochs_while_the_rate_is_high(tree_file):
    # Regression: a fixed patience cut 400-epoch runs short while the cosine was still high and the
    # validation loss jittered (BUGS.md). Here no epoch after the first improves (lr ~0), yet the
    # run must go on until the first epoch at or below the patience rate, then stop (patience 1).
    config = TrainConfig(**{**TINY, "epochs": 50, "lr": 1e-30, "lr_min": 0.0, "patience": 1})
    res = train_surrogate(tree_file, config)
    patience_lr = config.lr_min + training.PATIENCE_LR_FRACTION * (config.lr - config.lr_min)
    lrs = [h["lr"] for h in res.history]
    first_counted = next(i for i, lr in enumerate(lrs) if lr <= patience_lr)
    assert len(res.history) == first_counted + 1 < config.epochs
    assert first_counted > 1  # the high-rate epochs before it did not count


def test_shape_penalty_is_trained_on_when_enabled(tree_file):
    off = train_surrogate(tree_file, TrainConfig(**TINY))
    on = train_surrogate(tree_file, TrainConfig(**{**TINY, "lambda_shape": 1.0, "shape_points": 32}))
    assert all(h["shape_penalty"] == 0 for h in off.history)
    assert any(h["shape_penalty"] > 0 for h in on.history)
    assert on.history[0]["train_loss"] != off.history[0]["train_loss"]


@pytest.mark.parametrize("lambda_shape", [0.0, 1.0])
def test_lbfgs_finish_is_kept_only_if_it_improves_validation(tree_file, lambda_shape):
    config = TrainConfig(**{**TINY, "lbfgs_steps": 5, "lambda_shape": lambda_shape, "shape_points": 32})
    res = train_surrogate(tree_file, config)
    last = res.history[-1]
    assert last["epoch"] == "lbfgs"
    best_adam = min(h["valid_loss"] for h in res.history[:-1])
    train, valid = _split(tree_file, config)
    final = evaluate_labeled(res.surrogate.model, valid, LossScales.fit(train[2], train[3]), config.lambda_delta)
    if last["kept"]:
        assert last["valid_loss"] < best_adam
        assert final["loss"] == pytest.approx(last["valid_loss"], rel=1e-6)
    else:
        assert final["loss"] == pytest.approx(best_adam, rel=1e-6)


def test_too_few_rows_to_validate_raises(tree_file):
    with pytest.raises(ValueError, match="valid_ratio"):
        train_surrogate(tree_file, TrainConfig(**{**TINY, "valid_ratio": 1e-4}))


def test_a_validation_loss_that_is_never_finite_raises(tree_file, monkeypatch):
    nan = dict.fromkeys(("loss", "premium_term", "delta_term", "premium_rmse", "delta_rmse"), float("nan"))
    monkeypatch.setattr(training, "evaluate_labeled", lambda *a, **k: nan)
    with pytest.raises(ValueError, match="diverged"):
        train_surrogate(tree_file, TrainConfig(**TINY))


def test_an_lbfgs_finish_that_makes_validation_worse_is_thrown_away(tree_file, monkeypatch):
    def scramble(model, *args):  # stands in for a finish that wrecks the weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(10.0)

    monkeypatch.setattr(training, "_lbfgs_finish", scramble)
    config = TrainConfig(**{**TINY, "lbfgs_steps": 5})
    res = train_surrogate(tree_file, config)
    assert res.history[-1]["kept"] is False
    train, valid = _split(tree_file, config)
    final = evaluate_labeled(res.surrogate.model, valid, LossScales.fit(train[2], train[3]), config.lambda_delta)
    assert final["loss"] == pytest.approx(min(h["valid_loss"] for h in res.history[:-1]), rel=1e-6)
