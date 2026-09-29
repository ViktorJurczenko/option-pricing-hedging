import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from core.base import OptionPricing
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
            ridge_lambda: float = 0.0,
            moneyness_bins: int = 1,
            moneyness_bins_window: float = 1.0
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
            ridge_lambda: L2 regularization strength for the per-timestep regression.
                0.0 (default) fits an unregularized LinearRegression, matching prior
                behavior exactly; > 0 switches to Ridge, to discourage the exploding
                coefficients that cause delta/gamma to overshoot near maturity.
            moneyness_bins: number of quantile bins (by in-the-money S_t) to split
                each timestep's regression into, fitting one model per bin instead of
                a single global fit. 1 (default) reduces to exactly the prior
                single-global-fit behavior. A local/binned fit only has to bend
                sharply where paths in that bin actually sit, rather than
                compromising its shape across the whole in-the-money range to
                accommodate the near-maturity kink - but splitting a timestep's
                paths into more, smaller fits also makes each fit less stable, so
                this should only be applied where the kink actually needs it (see
                moneyness_bins_window).
            moneyness_bins_window: fraction of the timeline, counted backward from
                maturity, over which moneyness_bins is actually applied; outside
                that window a single global fit is used regardless of
                moneyness_bins. 1.0 (default) applies it across the whole timeline.
                The value function's kink only sharpens close to maturity - binning
                timesteps where it hasn't (most of the timeline) trades away
                accuracy for no benefit, since a smaller per-bin sample makes an
                already-well-behaved global fit worse, not better.
        """
        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        if reg_type not in ["simple", "laguerre"]:
            raise ValueError("reg_type must be 'simple' or 'laguerre'")
        if max_degree < 0:
            raise ValueError("max_degree must be >= 0")
        if ridge_lambda < 0:
            raise ValueError("ridge_lambda must be >= 0")
        if moneyness_bins < 1:
            raise ValueError("moneyness_bins must be >= 1")
        if not 0.0 <= moneyness_bins_window <= 1.0:
            raise ValueError("moneyness_bins_window must be in [0, 1]")

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
        self.ridge_lambda = ridge_lambda
        self.moneyness_bins = moneyness_bins
        self.moneyness_bins_window = moneyness_bins_window

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        if self.ridge_lambda > 0:
            return Ridge(alpha=self.ridge_lambda, fit_intercept=False).fit(X, y)
        return LinearRegression(fit_intercept=False).fit(X, y)

    def fit(self, payoff_next: np.ndarray, index: int):
        dt = self.maturity / self.simulation.shape[0]
        S_t = self.simulation[index] / self.strike

        if self.option_type == "put":
            itm_mask = self.simulation[index] < self.strike   # S < K → put is ITM
        else:
            itm_mask = self.simulation[index] > self.strike   # S > K → call is ITM

        true_continue = np.exp(-self.risk_free_rate * dt) * payoff_next

        # Only fit on ITM paths
        S_t_itm = S_t[itm_mask]
        true_continue_itm = true_continue[itm_mask]

        if len(S_t_itm) == 0:
            # No ITM paths → discount next payoff forward for all paths, no exercise
            self.continue_val[index] = np.exp(-self.risk_free_rate * dt) * payoff_next
            return

        if self.reg_type == "laguerre":
            features_itm = laguerre_poly_gen(S_t_itm, self.max_degree)
            features_all = laguerre_poly_gen(S_t, self.max_degree)
            deriv_1 = laguerre_deriv_k(S_t, self.max_degree, 1)
            deriv_2 = laguerre_deriv_k(S_t, self.max_degree, 2)
        else:
            features_itm = simple_poly(S_t_itm, self.max_degree)
            features_all = simple_poly(S_t, self.max_degree)
            deriv_1 = simple_poly_deriv_k(S_t, self.max_degree, 1)
            deriv_2 = simple_poly_deriv_k(S_t, self.max_degree, 2)

        # Moneyness-binned regression: split ITM paths into `moneyness_bins`
        # quantile bins by S_t and fit one model per bin, instead of a single
        # global fit across every ITM path. moneyness_bins=1 reduces to
        # exactly the prior single-global-fit behavior (interior_edges is
        # then empty, so every path maps to bin 0 - see np.digitize below).
        n_steps = self.simulation.shape[0]
        t_frac_to_maturity = (index + 1) / n_steps  # 0 at the first step, ~1 at the step before maturity
        near_maturity = t_frac_to_maturity >= 1 - self.moneyness_bins_window

        n_bins = self.moneyness_bins if near_maturity else 1
        min_points_per_bin = max(self.max_degree + 2, 10)
        if len(S_t_itm) < n_bins * min_points_per_bin:
            # Not enough ITM paths to support this many bins reliably at
            # this timestep - fall back to a single bin just for it.
            n_bins = 1

        bin_edges = np.quantile(S_t_itm, np.linspace(0, 1, n_bins + 1))
        interior_edges = bin_edges[1:-1]
        # Paths outside the ITM range (e.g. OTM paths, for which we still
        # need a continuation-value/Greeks estimate) are clipped to the
        # nearest bin rather than left unassigned.
        bin_idx_itm = np.clip(np.digitize(S_t_itm, interior_edges), 0, n_bins - 1)
        bin_idx_all = np.clip(np.digitize(S_t, interior_edges), 0, n_bins - 1)

        for b in range(n_bins):
            train_mask = bin_idx_itm == b
            if not np.any(train_mask):
                continue

            model = self._fit_model(features_itm[train_mask], true_continue_itm[train_mask])

            pred_mask = bin_idx_all == b
            self.continue_val[index, pred_mask] = model.predict(features_all[pred_mask])
            self.delta[index, pred_mask] = model.predict(deriv_1[pred_mask]) / self.strike
            self.gamma[index, pred_mask] = model.predict(deriv_2[pred_mask]) / self.strike ** 2

    def predict_payoff(self) -> None:
        """
        Predicts the payoff of each paths at all times
        """
        dt = self.maturity / self.simulation.shape[0]

        if self.option_type == "call":
            self.payoff[-1] = np.maximum(self.simulation[-1] - self.strike, 0)
        else:
            self.payoff[-1] = np.maximum(self.strike - self.simulation[-1], 0)

        for i in range(self.simulation.shape[0] - 2, -1, -1): # Traverses prediction backward starting from the second last element
            self.fit(self.payoff[i + 1], i)

            if self.option_type == "call":
                exercice_payoff = np.maximum(self.simulation[i] - self.strike, 0)
            else:
                exercice_payoff = np.maximum(self.strike - self.simulation[i], 0)

            exercice_mask = (exercice_payoff > 0) & (exercice_payoff >= self.continue_val[i])
            # Note : it's important to make sure exercice payoff is positive as it could be null but still selected since
            # the continue payoff which is found via regression may be negative

            self.payoff[i, exercice_mask] = exercice_payoff[exercice_mask]
            self.payoff[i, ~exercice_mask] = np.exp(-self.risk_free_rate * dt) * self.payoff[i + 1, ~exercice_mask]

            self.exercise_time[exercice_mask] = i

    def calculate_option_price(self, T: float = 1) -> float:
        # It's important to note that we don't use T here because we are evaluating the price of an american option which can be exercised
        # at any time so the time to maturity is pointless in american option pricing.
        return np.mean(self.payoff[0], dtype=float)
