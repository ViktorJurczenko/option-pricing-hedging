import numpy as np

from core.base import OptionPricing
from EuropeanOptionPricing.BlackScholes import BlackScholes
from AmericanOptionPricing.basis_functions import (
    simple_poly,
    simple_poly_deriv_k,
    laguerre_poly_gen,
    laguerre_deriv_k,
)

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
            control_variate: bool = False
        ):
        """
        Initialize the Longstaff-Schwartz model.

        Args:
            simulation: The price prediciton obtained with Geometric Brownian Motion (GBM).
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
                (see BUGS.md). Also applied to calculate_option_price().
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

        self.simulation = simulation
        self.continue_val = np.zeros(simulation.shape)
        self.exercise_time = np.full(simulation.shape[1], simulation.shape[0] - 1) # Default = maturity (i.e. not early exercised)
        self.delta = np.zeros((simulation.shape[0] - 1, simulation.shape[1])) # No delta at the end
        self.gamma = np.zeros((simulation.shape[0] - 1, simulation.shape[1]))
        self.payoff = np.zeros(simulation.shape)
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.maturity = maturity
        self.option_type = option_type
        self.reg_type = reg_type
        self.max_degree = max_degree
        self.sigma = sigma
        self.control_variate = control_variate

    def _basis(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        """k-th derivative (0 = the values) of the regression basis at x = S/K."""
        if self.reg_type == "laguerre":
            return laguerre_poly_gen(x, self.max_degree) if k == 0 else laguerre_deriv_k(x, self.max_degree, k)
        return simple_poly(x, self.max_degree) if k == 0 else simple_poly_deriv_k(x, self.max_degree, k)

    def _intrinsic(self, S: np.ndarray) -> np.ndarray:
        if self.option_type == "call":
            return np.maximum(S - self.strike, 0)
        return np.maximum(self.strike - S, 0)

    def _european(self, S: np.ndarray) -> BlackScholes:
        """The European option on the same strike at spots S (callers pass the row's remaining maturity)."""
        return BlackScholes(self.sigma, S, self.strike, self.risk_free_rate, self.option_type)

    def _realized_european_payoff(self, index: int) -> np.ndarray:
        """Each path's realized European payoff, discounted from expiry to row `index`."""
        n_steps = self.simulation.shape[0]
        dt = self.maturity / n_steps
        return np.exp(-self.risk_free_rate * (n_steps - 1 - index) * dt) * self._intrinsic(self.simulation[-1])

    def fit(self, payoff_next: np.ndarray, index: int):
        """Regressions at row `index`, given each path's realized cashflow
        `payoff_next` (discounted to row index + 1) under the exercise rule
        already fitted for the later rows.

        - Continuation value: regressed on in-the-money paths only, as in the
          original method. It drives the exercise decision in predict_payoff().
          With control_variate, the regression target is the realized
          cashflow minus the realized European payoff, and the exact European
          price is added back to the fit.
        - Delta: its own regression, over all paths, of the pathwise delta
          (the derivative of each path's discounted cashflow with respect to
          S_index: +-S_stop/S_index discounted, if the path pays off). Gamma is
          the derivative of that fit. Differentiating the continuation fit
          instead extrapolates the ITM-only polynomial to out-of-the-money
          paths, where it gave deltas in the tens (see BUGS.md).

        At row 0 every path shares S0, so both regressions are degenerate:
        lstsq returns the minimum-norm fit, whose prediction is the mean of the
        targets (the plain Monte-Carlo estimate) at that single spot.
        """
        n_steps = self.simulation.shape[0]
        dt = self.maturity / n_steps
        S = self.simulation[index]
        x = S / self.strike

        realized_continuation = np.exp(-self.risk_free_rate * dt) * payoff_next
        features = self._basis(x)
        tau = (n_steps - 1 - index) * dt
        if self.sigma is not None:
            european = self._european(S)

        # Continuation value, on ITM paths only
        itm_mask = self._intrinsic(S) > 0
        if not np.any(itm_mask):
            # No ITM paths → no exercise decision to make at this row
            self.continue_val[index] = realized_continuation
        elif self.control_variate:
            target = realized_continuation - self._realized_european_payoff(index)
            continuation_weights = np.linalg.lstsq(features[itm_mask], target[itm_mask], rcond=None)[0]
            self.continue_val[index] = features @ continuation_weights + european.calculate_option_price(tau)
        else:
            continuation_weights = np.linalg.lstsq(features[itm_mask], realized_continuation[itm_mask], rcond=None)[0]
            self.continue_val[index] = features @ continuation_weights

        # Pathwise delta. At this point exercise_time holds each path's
        # cashflow row under the rule for rows index + 1 onward.
        stop = self.exercise_time
        stop_price = self.simulation[stop, np.arange(S.shape[0])]
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

        greeks_weights = np.linalg.lstsq(delta_features, pathwise_delta, rcond=None)[0]
        self.delta[index] = delta_features @ greeks_weights
        self.gamma[index] = gamma_features @ greeks_weights

    def predict_payoff(self) -> None:
        """
        Predicts the payoff of each paths at all times
        """
        dt = self.maturity / self.simulation.shape[0]

        self.payoff[-1] = self._intrinsic(self.simulation[-1])

        for i in range(self.simulation.shape[0] - 2, -1, -1): # Traverses prediction backward starting from the second last element
            self.fit(self.payoff[i + 1], i)

            exercice_payoff = self._intrinsic(self.simulation[i])

            exercice_mask = (exercice_payoff > 0) & (exercice_payoff >= self.continue_val[i])
            # It's important to make sure exercice payoff is positive as it could be null but still selected since
            # the continue payoff which is found via regression may be negative

            self.payoff[i, exercice_mask] = exercice_payoff[exercice_mask]
            self.payoff[i, ~exercice_mask] = np.exp(-self.risk_free_rate * dt) * self.payoff[i + 1, ~exercice_mask]

            self.exercise_time[exercice_mask] = i

    def calculate_option_price(self, T: float = 1) -> float:
        # It's important to note that we don't use T here because we are evaluating the price of an american option which can be exercised
        # at any time so the time to maturity is pointless in american option pricing.
        if self.control_variate:
            # Same control variate as the regressions: exact European price plus
            # the simulated early-exercise premium (cashflow minus European payoff).
            tau = self.maturity * (self.simulation.shape[0] - 1) / self.simulation.shape[0]
            european_price = self._european(self.simulation[0]).calculate_option_price(tau)
            return np.mean(european_price + self.payoff[0] - self._realized_european_payoff(0), dtype=float)
        return np.mean(self.payoff[0], dtype=float)
