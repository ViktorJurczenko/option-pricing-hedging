import numpy as np
from scipy.special import ndtr

from optionpricing.core.base import OptionPricing

# Black-Scholes formulas for European options, as plain functions: every
# parameter can be an array (they broadcast together), so a batch of options
# with different spots, strikes, maturities, rates and volatilities is priced
# in one call. The `BlackScholes` class below is a thin wrapper for code that
# fixes the option once and prices it at several times to maturity.


def _pdf(x):
    """Standard normal density (scipy's `norm.pdf` without its distribution overhead)."""
    return np.exp(-0.5 * np.square(x)) / np.sqrt(2 * np.pi)


def _check_option_type(option_type: str):
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")


def _d1(start_price, strike, maturity, risk_free_rate, sigma):
    """d1 of the Black-Scholes formula; d2 is d1 - sigma * sqrt(maturity).

    Raises:
        ValueError: if any maturity or sigma is <= 0.
    """
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (start_price, strike, maturity, risk_free_rate, sigma))
    if np.any(T <= 0):
        raise ValueError("T (time to maturity) must be > 0")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be > 0")
    return (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))


def black_scholes_price(start_price, strike, maturity, risk_free_rate, sigma, option_type: str = "call"):
    """European option price.

    The price is homogeneous of degree 1 in (S, K): with strike=1 and
    start_price=S/K it is the price per unit of strike.

    Raises:
        ValueError: for an option_type other than "call"/"put", or any
            maturity or sigma <= 0.
    """
    _check_option_type(option_type)
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (start_price, strike, maturity, risk_free_rate, sigma))
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    discounted_strike = K * np.exp(-r * T)
    if option_type == "call":
        return S * ndtr(d1) - discounted_strike * ndtr(d2)
    return discounted_strike * ndtr(-d2) - S * ndtr(-d1)


def black_scholes_delta(start_price, strike, maturity, risk_free_rate, sigma, option_type: str = "call"):
    """Derivative of the price with respect to the spot: N(d1) for a call,
    N(d1) - 1 for a put. Raises like `black_scholes_price`."""
    _check_option_type(option_type)
    delta = ndtr(_d1(start_price, strike, maturity, risk_free_rate, sigma))
    return delta if option_type == "call" else delta - 1


def black_scholes_gamma(start_price, strike, maturity, risk_free_rate, sigma):
    """Second derivative of the price with respect to the spot, the same for
    a call and a put: n(d1) / (S * sigma * sqrt(T)). Raises like
    `black_scholes_price` (for maturity and sigma)."""
    S, T, sigma = (np.asarray(x, dtype=float) for x in (start_price, maturity, sigma))
    return _pdf(_d1(S, strike, T, risk_free_rate, sigma)) / (S * sigma * np.sqrt(T))


def black_scholes_vega(start_price, strike, maturity, risk_free_rate, sigma):
    """Derivative of the price with respect to sigma, the same for a call
    and a put: S * n(d1) * sqrt(T). Raises like `black_scholes_price` (for
    maturity and sigma)."""
    S, T = (np.asarray(x, dtype=float) for x in (start_price, maturity))
    return S * _pdf(_d1(S, strike, T, risk_free_rate, sigma)) * np.sqrt(T)


def black_scholes_dT(start_price, strike, maturity, risk_free_rate, sigma, option_type: str = "call"):
    """Derivative of the price with respect to the time to maturity T, i.e.
    minus the theta dV/dt: S n(d1) sigma / (2 sqrt(T)) + r K e^(-rT) N(d2)
    for a call, S n(d1) sigma / (2 sqrt(T)) - r K e^(-rT) N(-d2) for a put
    (negative for a deep in-the-money European put with r > 0, which an
    American put never is). Raises like `black_scholes_price`."""
    _check_option_type(option_type)
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (start_price, strike, maturity, risk_free_rate, sigma))
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    time_value = S * _pdf(d1) * sigma / (2 * np.sqrt(T))
    discounted_strike = K * np.exp(-r * T)
    if option_type == "call":
        return time_value + r * discounted_strike * ndtr(d2)
    return time_value - r * discounted_strike * ndtr(-d2)


class BlackScholes(OptionPricing):
    """
    Class for Black and Scholes option pricing: the option's spot, strike,
    rate and volatility are fixed at construction, its time to maturity is
    given to each method. Extends OptionPricing; the formulas are the
    module's `black_scholes_*` functions.
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
        _check_option_type(option_type)
        if np.any(np.asarray(sigma) <= 0):
            raise ValueError("sigma must be > 0")

        self.sigma = sigma
        self.start_price = start_price
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.option_type = option_type

    def _d1(self, T: float):
        return _d1(self.start_price, self.strike, T, self.risk_free_rate, self.sigma)

    def calculate_option_price(self, T: float = 1) -> float:
        """
        Calculate the price of the option using Black and Scholes formula.

        Args:
            T: Time to maturity in T (defaults to 1).

        Returns:
            option_price: Estimated option price.
        """
        return black_scholes_price(self.start_price, self.strike, T, self.risk_free_rate, self.sigma, self.option_type)

    def calculate_delta(self, T: float = 1) -> float:
        """Derivative of the price with respect to the spot: N(d1) for a call,
        N(d1) - 1 for a put."""
        return black_scholes_delta(self.start_price, self.strike, T, self.risk_free_rate, self.sigma, self.option_type)

    def calculate_gamma(self, T: float = 1) -> float:
        """Second derivative of the price with respect to the spot, the same
        for a call and a put: n(d1) / (S * sigma * sqrt(T))."""
        return black_scholes_gamma(self.start_price, self.strike, T, self.risk_free_rate, self.sigma)
