import numpy as np
from sklearn.linear_model import LinearRegression

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
            max_degree: int = 3
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
        """
        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        if reg_type not in ["simple", "laguerre"]:
            raise ValueError("reg_type must be 'simple' or 'laguerre'")
        if max_degree < 0:
            raise ValueError("max_degree must be >= 0")

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
        else:
            features_itm = simple_poly(S_t_itm, self.max_degree)

        # Train only with ITM paths
        model = LinearRegression(fit_intercept=False).fit(features_itm, true_continue_itm)

        # Predict continuation for ALL paths (including OTM)
        # so that predict() can use continue_val[index] for all paths
        if self.reg_type == "laguerre":
            features_all = laguerre_poly_gen(S_t, self.max_degree)
            deriv_1 = laguerre_deriv_k(S_t, self.max_degree, 1)
            deriv_2 = laguerre_deriv_k(S_t, self.max_degree, 2)
        else:
            features_all = simple_poly(S_t, self.max_degree)
            deriv_1 = simple_poly_deriv_k(S_t, self.max_degree, 1)
            deriv_2 = simple_poly_deriv_k(S_t, self.max_degree, 2)

        self.continue_val[index] = model.predict(features_all)
        self.delta[index] = model.predict(deriv_1) / self.strike
        self.gamma[index] = model.predict(deriv_2) / self.strike ** 2

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
