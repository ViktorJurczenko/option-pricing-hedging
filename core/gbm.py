import numpy as np
import matplotlib.pyplot as plt

from typing import Optional

class GBM():

    def __init__(
        self,
        sigma: float = 0.2,
        start_price: float = 100.0,
        risk_free_rate: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Monte-Carlo simulator.

        Args:
            sigma: Volatility (standard deviation of returns).
            start_price: Initial stock price.
            risk_free_rate: Risk-free interest rate.
            seed: Random seed for reproducibility.
        """
        self.sigma = sigma
        self.start_price = start_price
        self.risk_free_rate = risk_free_rate
        self.seed = seed
        self.simulations: Optional[np.ndarray] = None

    def simulate_paths(self, nb_predictions: int = 10000, maturity: float = 1.0, sampling: int = 365) -> np.ndarray:
        """
        Simulate stock price paths using Geometric Brownian Motion (GBM).

        Args:
            maturity: Time to maturity, in years, to simulate.
            sampling: Number of time steps per year (e.g., 365 for daily).
            nb_predictions: Number of simulation paths.

        Returns:
            self.simulations: 2D array of shape (maturity * sampling, nb_predictions).
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        size = int(maturity * sampling)
        dt = maturity / size

        self.simulations = np.zeros((size, nb_predictions))

        self.simulations[0] = self.start_price

        for i in range(1, size):
            Z = np.random.standard_normal(nb_predictions)
            self.simulations[i] = self.simulations[i - 1] * np.exp((self.risk_free_rate - 0.5 * self.sigma**2) * dt +(self.sigma * np.sqrt(dt) * Z))

        return self.simulations

    def plot_paths(self) -> None:
        """
        Plot the first 50 simulated paths.
        """
        if self.simulations is None:
            self.simulate_paths()

        plt.figure(figsize=(10, 6))
        for i in range(min(50, self.simulations.shape[1])):
            plt.plot(self.simulations[:, i], lw=1, alpha=0.5)

        plt.xlabel("Time Step")
        plt.ylabel("Price")
        plt.title("Monte-Carlo Simulation of Stock Price")
        plt.grid(True)
        plt.show()
