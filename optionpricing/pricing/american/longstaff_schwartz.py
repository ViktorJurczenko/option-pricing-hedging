import numpy as np

from optionpricing.core.base import OptionPricing
from optionpricing.pricing.american.basis_functions import (
    laguerre_deriv_k,
    laguerre_poly_gen,
    simple_poly,
    simple_poly_deriv_k,
)
from optionpricing.pricing.european.black_scholes import BlackScholes


def least_squares(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Least-squares weights of `target` on `features` via the normal equations.

    Solves the small (n_features x n_features) system `X^T X w = X^T y` instead
    of taking an SVD of the tall (n_paths x n_features) matrix like
    `np.linalg.lstsq(X, y)`, which is several times cheaper for the 10k-path
    regressions here. Forming `X^T X` squares the condition number, which costs
    about 4-5 digits on the Laguerre basis (deltas differ by ~1e-5), so
    LongstaffSchwartz uses it only when it isn't fitting Greeks. The Gram matrix is still handed to `lstsq`, not `solve`,
    so a degenerate system (every path at the same spot at step 0, or too few
    ITM paths) gets the same minimum-norm solution as lstsq on `X` itself.
    """
    return np.linalg.lstsq(features.T @ features, features.T @ target, rcond=None)[0]


class LongstaffSchwartz(OptionPricing):
    """
    Class for Longstaff-Schwartz option pricing.
    Extends OptionPricing.
    """

    def __init__(
        self,
        simulation: np.ndarray,
        strike: float = 100.0,
        risk_free_rate: float = 0.05,
        maturity: float = 1.0,
        option_type: str = "call",
        reg_type: str = "simple",
        max_degree: int = 3,
        sigma: float | None = None,
        control_variate: bool = False,
        compute_greeks: bool = True,
    ):
        """
        Initialize the Longstaff-Schwartz model.

        Args:
            simulation: Simulated prices, shape (n_paths, n_steps + 1), time last as `GBM.simulate_paths`
                returns them.
            strike: Strike price of the option.
            risk_free_rate: Risk-free interest rate.
            maturity: Time to maturity in years (defaults to 1).
            option_type: Type of option ("call" or "put").
            reg_type: The type of polynomial used for regression
            max_degree: The maximum degree of the features of the regression.
            sigma: Volatility the simulation was generated with. Optional: when
                given, the Black-Scholes delta of the European option is added
                as a feature of the delta regression (see fit()). Without it,
                a cubic basis can't follow delta's steepening near expiry.
                It must be the simulation's own volatility; any other value
                makes that feature a biased guide.
            control_variate: use the European option as a control variate in
                the continuation regression (requires sigma): regress the
                realized cashflow minus the realized discounted European payoff,
                then add back the exact Black-Scholes price. Paths that aren't
                exercised early contribute exactly 0 to that target, which
                removes most of the regression noise. For an American call
                (never optimally exercised early) this eliminates the spurious
                early exercise plain LSM makes near expiry; for a put the plain
                rule is as good or better out of sample, so this is opt-in
                (.claude/BUGS.md, B-006). Also applied to calculate_option_price().
            compute_greeks: fit the per-path delta and gamma (the hedging
                environment needs them). Turn it off when only the price is
                wanted (dataset generation): it skips the delta regression and
                the Black-Scholes/basis derivatives, and `delta`/`gamma` stay
                None. The price and the exercise rule are unaffected.
        """
        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        if reg_type not in ["simple", "laguerre"]:
            raise ValueError("reg_type must be 'simple' or 'laguerre'")
        if max_degree < 0:
            raise ValueError("max_degree must be >= 0")
        if sigma is not None and sigma <= 0:
            raise ValueError("sigma must be > 0")
        if control_variate and sigma is None:
            raise ValueError("control_variate requires sigma")
        if simulation.ndim != 2 or simulation.shape[1] < 2:
            raise ValueError(
                "simulation must have shape (n_paths, n_steps + 1) with at least 2 steps (start and expiry)"
            )

        self.simulation = simulation
        # The backward induction reads and writes one step of every path at a time, so it works on step-major
        # copies, where a step is contiguous in memory; the public arrays below are time-last views of them.
        # The copy of the simulation doubles its memory (~150 MB at 50,000 paths x 366 steps).
        self._by_step = np.ascontiguousarray(simulation.T)
        n_steps, n_paths = self._by_step.shape
        self._continue_val = np.zeros((n_steps, n_paths))
        self.exercise_time = np.full(n_paths, n_steps - 1)  # Default = maturity (i.e. not early exercised)
        self.compute_greeks = compute_greeks
        # No delta at the end. None (not zeros) without compute_greeks, so using them fails loudly.
        self._delta = np.zeros((n_steps - 1, n_paths)) if compute_greeks else None
        self._gamma = np.zeros((n_steps - 1, n_paths)) if compute_greeks else None
        self._payoff = None  # Allocated by predict_payoff(); None marks an unfitted model
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.maturity = maturity
        # Step 0 is today and the last step is expiry, so the steps are n_steps - 1 intervals apart.
        self.dt = maturity / (n_steps - 1)
        self.option_type = option_type
        self.reg_type = reg_type
        self.max_degree = max_degree
        self.sigma = sigma
        self.control_variate = control_variate

    @property
    def payoff(self) -> np.ndarray | None:
        """Each path's cashflow under the fitted rule, discounted to each step, shape (n_paths, n_steps + 1)."""
        return None if self._payoff is None else self._payoff.T

    @property
    def continue_val(self) -> np.ndarray:
        """Fitted continuation value per path and step, shape (n_paths, n_steps + 1)."""
        return self._continue_val.T

    @property
    def delta(self) -> np.ndarray | None:
        """Fitted delta per path and step, shape (n_paths, n_steps) (none at expiry), or None without Greeks."""
        return None if self._delta is None else self._delta.T

    @property
    def gamma(self) -> np.ndarray | None:
        """Fitted gamma per path and step, shape (n_paths, n_steps) (none at expiry), or None without Greeks."""
        return None if self._gamma is None else self._gamma.T

    def _basis(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        """k-th derivative (0 = the values) of the regression basis at x = S/K."""
        if self.reg_type == "laguerre":
            return laguerre_poly_gen(x, self.max_degree) if k == 0 else laguerre_deriv_k(x, self.max_degree, k)
        return simple_poly(x, self.max_degree) if k == 0 else simple_poly_deriv_k(x, self.max_degree, k)

    def _regress(self, features: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Least-squares weights: exact SVD `lstsq` when fitting Greeks (the hedging
        environment trains on them), the faster normal equations for price-only runs."""
        if self.compute_greeks:
            return np.linalg.lstsq(features, target, rcond=None)[0]
        return least_squares(features, target)

    def _intrinsic(self, S: np.ndarray) -> np.ndarray:
        if self.option_type == "call":
            return np.maximum(S - self.strike, 0)
        return np.maximum(self.strike - S, 0)

    def _european(self, S: np.ndarray) -> BlackScholes:
        """The European option on the same strike at spots S (callers pass the step's remaining maturity)."""
        return BlackScholes(self.sigma, S, self.strike, self.risk_free_rate, self.option_type)

    def _realized_european_payoff(self, index: int) -> np.ndarray:
        """Each path's realized European payoff, discounted from expiry to step `index`."""
        n_steps = self._by_step.shape[0]
        return np.exp(-self.risk_free_rate * (n_steps - 1 - index) * self.dt) * self._intrinsic(self._by_step[-1])

    def fit(self, payoff_next: np.ndarray, index: int):
        """Regressions at step `index`, given each path's realized cashflow
        `payoff_next` (discounted to step index + 1) under the exercise rule
        already fitted for the later steps.

        - Continuation value: regressed on in-the-money paths only, as in the
          original method. It drives the exercise decision in predict_payoff().
          With control_variate, the regression target is the realized
          cashflow minus the realized European payoff, and the exact European
          price is added back to the fit.
        - Delta (only with compute_greeks): its own regression, over all paths, of the pathwise delta
          (the derivative of each path's discounted cashflow with respect to
          S_index: ±S_stop/S_index discounted, if the path pays off). Gamma is
          the derivative of that fit. Differentiating the continuation fit
          instead extrapolates the ITM-only polynomial to out-of-the-money
          paths, where it gave deltas in the tens (.claude/BUGS.md, B-003).

        At step 0 every path shares S0, so both regressions are degenerate:
        lstsq returns the minimum-norm fit, whose prediction is the mean of the
        targets (the plain Monte-Carlo estimate) at that single spot.
        """
        n_steps = self._by_step.shape[0]
        dt = self.dt
        S = self._by_step[index]
        x = S / self.strike

        realized_continuation = np.exp(-self.risk_free_rate * dt) * payoff_next
        features = self._basis(x)
        tau = (n_steps - 1 - index) * dt
        european = (
            self._european(S) if self.sigma is not None and (self.control_variate or self.compute_greeks) else None
        )

        # Continuation value, on ITM paths only
        itm_mask = self._intrinsic(S) > 0
        if not np.any(itm_mask):
            # No ITM paths → no exercise decision to make at this step
            self._continue_val[index] = realized_continuation
        elif self.control_variate:
            target = realized_continuation - self._realized_european_payoff(index)
            continuation_weights = self._regress(features[itm_mask], target[itm_mask])
            self._continue_val[index] = features @ continuation_weights + european.calculate_option_price(tau)
        else:
            continuation_weights = self._regress(features[itm_mask], realized_continuation[itm_mask])
            self._continue_val[index] = features @ continuation_weights

        if not self.compute_greeks:
            return

        # Pathwise delta. At this point exercise_time holds each path's
        # cashflow step under the rule for steps index + 1 onward.
        stop = self.exercise_time
        stop_price = self._by_step[stop, np.arange(S.shape[0])]
        sign = 1.0 if self.option_type == "call" else -1.0
        pathwise_delta = np.where(
            payoff_next > 0,
            sign * stop_price / S * np.exp(-self.risk_free_rate * (stop - index) * dt),
            0.0,
        )

        delta_features = features
        gamma_features = self._basis(x, 1) / self.strike
        if self.sigma is not None:
            delta_features = np.column_stack([delta_features, european.calculate_delta(tau)])
            gamma_features = np.column_stack([gamma_features, european.calculate_gamma(tau)])

        greeks_weights = self._regress(delta_features, pathwise_delta)
        self._delta[index] = delta_features @ greeks_weights
        self._gamma[index] = gamma_features @ greeks_weights

    def predict_payoff(self) -> None:
        """
        Predicts the payoff of each paths at all times
        """
        dt = self.dt

        self._payoff = np.zeros(self._by_step.shape)
        self._payoff[-1] = self._intrinsic(self._by_step[-1])

        for i in range(
            self._by_step.shape[0] - 2, -1, -1
        ):  # Traverses prediction backward starting from the second last element
            self.fit(self._payoff[i + 1], i)

            exercice_payoff = self._intrinsic(self._by_step[i])

            exercice_mask = (exercice_payoff > 0) & (exercice_payoff >= self._continue_val[i])
            # It's important to make sure exercice payoff is positive as it could be null but still selected since
            # the continue payoff which is found via regression may be negative

            self._payoff[i, exercice_mask] = exercice_payoff[exercice_mask]
            self._payoff[i, ~exercice_mask] = np.exp(-self.risk_free_rate * dt) * self._payoff[i + 1, ~exercice_mask]

            self.exercise_time[exercice_mask] = i

    def calculate_option_price(self, T: float = 1) -> float:
        # It's important to note that we don't use T here because we are evaluating the price of an american option which can be exercised
        # at any time so the time to maturity is pointless in american option pricing.
        if self._payoff is None:
            # Same error as BinomialTree, which has no payoff until predict_payoff().
            raise AttributeError("call predict_payoff() before calculate_option_price()")
        if self.control_variate:
            # Same control variate as the regressions: exact European price plus
            # the simulated early-exercise premium (cashflow minus European payoff).
            european_price = self._european(self._by_step[0]).calculate_option_price(self.maturity)
            return np.mean(european_price + self._payoff[0] - self._realized_european_payoff(0), dtype=float)
        return np.mean(self._payoff[0], dtype=float)
