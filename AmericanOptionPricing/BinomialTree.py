import numpy as np

from core.base import OptionPricing

class BinomialTree(OptionPricing):
    """
    Class for Binomial tree option pricing.
    Extends OptionPricing.
    """

    def __init__(
            self,
            sigma: float = 0.2,
            start_price: float = 100.0,
            strike: float = 100.0,
            risk_free_rate: float = 0.05,
            maturity: float = 1.0,
            nb_steps: int = 1000,
            option_type: str = "call"
        ):
        """
        Initialize the Binomial tree model.

        Args:
            sigma: Volatility (standard deviation of returns).
            start_price: Initial stock price.
            strike: Strike price of the option.
            risk_free_rate: Risk-free interest rate.
            maturity: Time to maturity in years (defaults to 1).
            nb_steps: Number of tree steps.
            option_type: Type of option ("call" or "put").

        Raises:
            ValueError: if option_type is not "call"/"put", sigma <= 0,
                maturity <= 0 or nb_steps < 1.
        """
        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if maturity <= 0:
            raise ValueError("maturity must be > 0")
        if nb_steps < 1:
            raise ValueError("nb_steps must be >= 1")

        self.sigma = sigma
        self.start_price = start_price
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.maturity = maturity
        self.nb_steps = nb_steps
        self.option_type = option_type

    def predict_payoff(self):
        """
        Predicts the payoff at all times
        """
        dt = self.maturity / self.nb_steps
        discount = np.exp(-self.risk_free_rate * dt)

        u = np.exp(self.sigma * np.sqrt(dt)) # Up factor
        d = 1 / u # Down factor

        p = (np.exp(self.risk_free_rate * dt) - d) / (u - d) # Risk-neutral probability of an up move

        j = np.arange(self.nb_steps + 1)
        end_price = self.start_price * u ** j * d ** (self.nb_steps - j)

        if self.option_type == "call":
            self.payoff = np.maximum(end_price - self.strike, 0)
        else:
            self.payoff = np.maximum(self.strike - end_price, 0)

        # Backpropagation of payoff
        for i in range(self.nb_steps - 1, -1, -1):
            j = np.arange(i + 1)
            price = self.start_price * u ** j * d ** (i - j)
            if self.option_type == "call":
                exercise_payoff = np.maximum(price - self.strike, 0)
            else:
                exercise_payoff = np.maximum(self.strike - price, 0)

            continuation_payoff = discount * (p * self.payoff[1:i+2] + (1 -p) * self.payoff[0:i+1])

            self.payoff = np.maximum(exercise_payoff[:i+1], continuation_payoff)


    def calculate_option_price(self, T: float = 1) -> float:
        # It's important to note that we don't use T here because we are evaluating the price of an american option which can be exercised
        # at any time so the time to maturity is pointless in american option pricing.
        return self.payoff[0]
