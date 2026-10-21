import numpy as np
from scipy.special import ndtr

from core.base import OptionPricing

def _pdf(x):
    """Standard normal density (scipy's `norm.pdf` without its distribution overhead)."""
    return np.exp(-0.5 * np.square(x)) / np.sqrt(2 * np.pi)

class BlackScholes(OptionPricing):
    """
    Class for Black and Scholes option pricing.
    Extends OptionPricing.
    """
    def __init__(
        self,
        sigma: float = 0.2,
        start_price: float = 100.0,
        strike: float = 100.0,
        risk_free_rate: float = 0.05,
        option_type: str = "call",
    ):
        """
        Initialize the Black and Scholes model.

        Args:
            sigma: Volatility (standard deviation of returns).
            start_price: Initial stock price.
            strike: Strike price of the option.
            risk_free_rate: Risk-free interest rate.
            option_type: Type of option ("call" or "put").

        Raises:
            ValueError: if option_type is not "call"/"put" or sigma <= 0. The
                pricing methods raise it too for a time to maturity T <= 0.
        """
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        self.sigma = sigma
        self.start_price = start_price
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.option_type = option_type

    def _d1(self, T: float):
        if np.any(np.asarray(T) <= 0):
            raise ValueError("T (time to maturity) must be > 0")
        return (np.log(self.start_price / self.strike) + (self.risk_free_rate + (self.sigma ** 2) / 2) * T) / (self.sigma * np.sqrt(T))

    def calculate_option_price(self, T: float = 1) -> float:
        """
        Calculate the price of the option using Black and Scholes formula.

        Args:
            T: Time to maturity in T (defaults to 1).

        Returns:
            option_price: Estimated option price.
        """
        d1 = self._d1(T)
        d2 = d1 - (self.sigma * np.sqrt(T))

        if self.option_type == 'call':
            option_price = self.start_price * ndtr(d1) - self.strike * np.exp(-self.risk_free_rate * T) * ndtr(d2)
        else:
            option_price = self.strike * np.exp(-self.risk_free_rate * T) * ndtr(-d2) - self.start_price * ndtr(-d1)
        return option_price

    def calculate_delta(self, T: float = 1) -> float:
        """Derivative of the price with respect to the spot: N(d1) for a call,
        N(d1) - 1 for a put."""
        delta = ndtr(self._d1(T))
        return delta if self.option_type == "call" else delta - 1

    def calculate_gamma(self, T: float = 1) -> float:
        """Second derivative of the price with respect to the spot, the same
        for a call and a put: n(d1) / (S * sigma * sqrt(T))."""
        return _pdf(self._d1(T)) / (self.start_price * self.sigma * np.sqrt(T))
