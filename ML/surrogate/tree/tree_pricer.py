import numpy as np
import torch

import os
from pathlib import Path
from typing import NamedTuple, Union
from numpy.typing import ArrayLike

from AmericanOptionPricing.BatchBinomialTree import binomial_put_batch
from EuropeanOptionPricing.BlackScholes import black_scholes_delta, black_scholes_gamma, black_scholes_price
from ML.surrogate.tree.tree_dataset import FeatureStats, build_features
from ML.surrogate.tree.tree_model import MLP

# The sampling ranges a surrogate is valid on, as generate_tree_dataset stores
# them (TreePutDataset.domain); each is an inclusive (low, high).
DOMAIN_KEYS = ("T_range", "sigma_range", "r_range", "z_range", "m_range")
# Below z_range's low end a put is exercised today only if r is not ~0: as
# r -> 0 early exercise disappears and the boundary goes to z = -inf. Every
# measured row below z = -3.3 with r >= 0.005 was exercised.
EXERCISE_MIN_RATE = 0.005
# Tag of the checkpoints PutSurrogate.save writes, so that load refuses
# anything else (e.g. a LegacyMLP state_dict) with a clear message.
CHECKPOINT_FORMAT = "tree_put_surrogate"
# Where each query's answer comes from (SurrogateResult.route).
# "payoff_floor": a net-route query whose net price fell below the payoff
# max(1 - m, 0), replaced by it (see PutSurrogate).
ROUTES = ("net", "payoff_floor", "black_scholes", "exercised", "tree")

class SurrogateResult(NamedTuple):
    """
    Per-option outputs of PutSurrogate.price, per unit of strike like
    binomial_put_batch's TreeResult: price = V / K, delta = dV/dS, gamma =
    K d2V/dS2 (scale back with K * price and gamma / K). `route` names the
    method that answered each option, one of ROUTES.
    """
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    route: np.ndarray

class PutSurrogate:
    """
    American put pricer built on a trained MLP: price = Black-Scholes
    European + the net's early-exercise premium, delta and gamma = the
    Black-Scholes ones + the net's derivatives in m by autograd. Every
    consumer (test-ml, a hedger, a calibration loop) reconstructs prices
    through `price`, so none can add the European part back differently.

    Queries outside the training domain are routed, not extrapolated
    (`price` says which way each went):
    - T, sigma, r inside, z > z_range high end: Black-Scholes (the premium
      there is < 4e-5 K);
    - T, sigma, r inside, z < z_range low end, r >= EXERCISE_MIN_RATE:
      exercised today, 1 - m, delta -1, gamma 0;
    - otherwise outside (m beyond its cap, T, sigma or r outside, small r
      deep in the money): binomial_put_batch at the dataset teacher's
      settings (BBSR, 1000 steps).
    The two closed forms are only used where T, sigma and r are inside the
    domain, since that is where they were measured to hold.

    Payoff floor: an American put can always be exercised, so it is never
    worth less than its payoff max(1 - m, 0). Where the net's price falls
    below it (route "payoff_floor"), the payoff replaces it, with its own
    delta (-1 in the money, 0 out of it) and gamma 0. The net is least
    accurate on puts exercised today, whose price is exactly 1 - m: the
    floor fixes the ones it under-prices (on a 32k-row trial run, the test
    price MAE fell 35%, 3.3e-4 -> 8.5e-5 on exercised rows). Where the net
    crosses the payoff at an angle the floor leaves a kink in delta; the
    true price meets it smoothly (smooth pasting).

    Args:
        model: the trained MLP (put in eval mode here).
        stats: the training split's FeatureStats the model was trained with.
        domain: the training data's sampling ranges, with every DOMAIN_KEYS
            entry (TreePutDataset.domain); other keys are ignored.

    Raises:
        ValueError: if `domain` misses a DOMAIN_KEYS entry.
    """
    def __init__(self, model: MLP, stats: FeatureStats, domain: dict):
        missing = [k for k in DOMAIN_KEYS if k not in domain]
        if missing:
            raise ValueError(f"domain is missing {missing}")
        self.model = model.eval()
        self.stats = stats
        self.domain = {k: (float(domain[k][0]), float(domain[k][1])) for k in DOMAIN_KEYS}

    def price(self, m: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike) -> SurrogateResult:
        """
        Prices American puts per unit of strike (K = 1, m = S / K).

        Args:
            m, T, r, sigma: arrays or scalars that broadcast together; the
                outputs have their broadcast shape.

        Returns:
            SurrogateResult(price, delta, gamma, route).

        Raises:
            ValueError: from binomial_put_batch, for a non-finite parameter
                or m, T or sigma <= 0: such queries are kept out of every
                other route, so they always reach the tree, which rejects them.
        """
        # work on flat 1-D arrays so one boolean mask per route indexes every
        # input and output alike; the caller's shape is restored at the end
        m, T, r, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=np.float64) for x in (m, T, r, sigma)))
        shape = m.shape
        m, T, r, sigma = (x.ravel() for x in (m, T, r, sigma))
        with np.errstate(divide="ignore", invalid="ignore"):  # m, T, sigma <= 0 end up in the tree, which rejects them
            z = np.log(m) / (sigma * np.sqrt(T))

        def inside(x, key):
            low, high = self.domain[key]
            return (x >= low) & (x <= high)

        # the closed forms were only measured for (T, sigma, r) in the domain,
        # hence params_inside; they don't need m's cap (beyond z's bounds,
        # further out of / into the money only makes them more exact). The
        # three masks are disjoint: z can't be both inside and outside its range.
        # The m check is needed because the z rules don't bound m: m = 0
        # (z = -inf) would pass as exercised and m = inf (z = +inf) as
        # Black-Scholes; T, sigma and r are already bounded by their ranges.
        m_valid = np.isfinite(m) & (m > 0)
        params_inside = m_valid & inside(T, "T_range") & inside(sigma, "sigma_range") & inside(r, "r_range")
        z_low, z_high = self.domain["z_range"]
        routes = {
            "net": params_inside & inside(z, "z_range") & inside(m, "m_range"),
            "black_scholes": params_inside & (z > z_high),
            "exercised": params_inside & (z < z_low) & (r >= EXERCISE_MIN_RATE),
        }
        # written as "none of the others" so invalid input (NaN, which fails
        # every comparison above, or m excluded by m_valid) lands here and
        # the tree raises on it
        routes["tree"] = ~(routes["net"] | routes["black_scholes"] | routes["exercised"])

        # np.empty is safe: the masks cover every row exactly once, so every
        # slot is written below. The route dtype must fit the longest name,
        # or numpy silently truncates it ("black_scholes" -> "black").
        price, delta, gamma = np.empty((3, len(m)))
        route = np.empty(len(m), dtype=f"<U{max(map(len, ROUTES))}")
        for name, mask in routes.items():
            route[mask] = name

        # net rows start from Black-Scholes too (the net only adds the
        # premium on top), so both routes share one closed-form call
        european = routes["net"] | routes["black_scholes"]
        if european.any():
            args = (m[european], 1.0, T[european], r[european], sigma[european])
            price[european] = black_scholes_price(*args, "put")
            delta[european] = black_scholes_delta(*args, "put")
            gamma[european] = black_scholes_gamma(*args)
        if routes["net"].any():
            net = routes["net"]
            # (premium, d premium/dm, d2 premium/dm2) added onto (price, delta, gamma)
            premium = self._premium(m[net], T[net], r[net], sigma[net])
            for out, add in zip((price, delta, gamma), premium):
                out[net] += add
            # payoff floor: never below what exercising now pays
            payoff = np.maximum(1.0 - m, 0.0)
            floored = net & (price < payoff)
            price[floored], gamma[floored] = payoff[floored], 0.0
            delta[floored] = np.where(m[floored] < 1.0, -1.0, 0.0)
            route[floored] = "payoff_floor"
        exercised = routes["exercised"]
        price[exercised], delta[exercised], gamma[exercised] = 1.0 - m[exercised], -1.0, 0.0
        tree = routes["tree"]
        if tree.any():
            # chunks of 32 (results don't depend on it): stepping the whole
            # batch at once makes each level's arrays too big for the CPU
            # cache, which the generator's timing found slower
            res = binomial_put_batch(m[tree], T[tree], r[tree], sigma[tree], chunk_size=32)
            price[tree], delta[tree], gamma[tree] = res.price, res.delta, res.gamma

        return SurrogateResult(*(x.reshape(shape) for x in (price, delta, gamma, route)))

    def _premium(self, m: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray):
        """
        The net's premium and its first two derivatives in m, by autograd
        through build_features and the stats, as (n,) float64 arrays. Each
        row's output depends on its own m only (no BatchNorm), so the
        gradient of the sum is each row's derivative.
        """
        param = next(self.model.parameters())
        # features built in float64 from m, then cast to the net's dtype and
        # device; the cast is differentiable, so gradients still reach m
        m_t = torch.tensor(m, dtype=torch.float64, requires_grad=True)
        x = self.stats.apply(build_features(m_t, T, r, sigma))
        premium = self.model(x.to(param)).squeeze(-1)
        # create_graph: keep delta's own graph, so it can be differentiated again for gamma
        dm, = torch.autograd.grad(premium.sum(), m_t, create_graph=True)
        # allow_unused: a net whose delta does not depend on m has no graph to differentiate
        dm2, = torch.autograd.grad(dm.sum(), m_t, allow_unused=True)
        dm2 = torch.zeros_like(m_t) if dm2 is None else dm2
        return tuple(v.detach().double().cpu().numpy() for v in (premium, dm, dm2))

    def save(self, path: Union[str, os.PathLike]) -> None:
        """
        Saves everything `load` needs in one file: the net's constructor
        arguments and weights, the feature stats and the domain, so a
        consumer can't pair the weights with other stats or another
        architecture.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": CHECKPOINT_FORMAT,
                    "model_config": self.model.config,
                    "state_dict": self.model.state_dict(),
                    "feature_stats": self.stats.to_dict(),
                    "domain": {k: list(v) for k, v in self.domain.items()}}, path)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "PutSurrogate":
        """
        Rebuilds a surrogate saved by `save`, on the CPU.

        Raises:
            ValueError: if the file is not a PutSurrogate checkpoint (e.g. a
                LegacyMLP state_dict from the LSM pipeline), or its feature
                stats are for other features (FeatureStats.from_dict).
        """
        # weights_only: the checkpoint is plain dicts, lists, strings and tensors
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict) or ckpt.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"'{path}' is not a PutSurrogate checkpoint (format '{CHECKPOINT_FORMAT}')")
        model = MLP(**ckpt["model_config"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, FeatureStats.from_dict(ckpt["feature_stats"]), ckpt["domain"])
