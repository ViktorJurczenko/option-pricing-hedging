import numpy as np

from core.base import OptionPricing

# --- Monte-Carlo Class ---
class MonteCarlo(OptionPricing):
    """
    Class for Monte-Carlo simulation and option pricing.
    Extends OptionPricing.
    """
    def __init__(
        self,
        predictions: np.ndarray,
        strike: float = 100.0,
        risk_free_rate: float = 0.05,
        option_type: str = "call",
    ):
        """
        Initialize the Monte-Carlo simulator.

        Args:
            predictions: The price prediciton obtained with Geometric Brownian Motion (GBM).
            strike: Strike price of the option.
            risk_free_rate: Risk-free interest rate.
            option_type: Type of option ("call" or "put").
        """
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

        self.predictions = predictions
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.option_type = option_type

    def calculate_payoffs(self) -> np.ndarray:
        """
        Calculate payoffs for all simulated paths.

        Returns:
            payoffs: Array of payoffs for each path.
        """

        final_prices = self.predictions[-1]
        if self.option_type == "call":
            payoffs = np.maximum(final_prices - self.strike, 0)
        else:
            payoffs = np.maximum(self.strike - final_prices, 0)

        return payoffs

    def calculate_option_price(self, T: float = 1) -> float:
        """
        Calculate the price of the option using Monte-Carlo simulation.

        Args:
            T: Time to maturity in T (defaults to 1).

        Returns:
            option_price: Estimated option price.
        """

        payoffs = self.calculate_payoffs()
        # Discount the average payoff to present value
        option_price = np.mean(payoffs) * np.exp(-self.risk_free_rate * T)
        return option_price
