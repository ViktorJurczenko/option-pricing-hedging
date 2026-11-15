"""Geometric Brownian Motion share-price paths under the risk-neutral drift, for one contract or many at once."""

import matplotlib.pyplot as plt
import numpy as np

PARAMETERS = ("sigma", "start_price", "risk_free_rate")


class GBM:
    """Simulates `S_k = S_{k-1} exp((r - sigma^2 / 2) dt + sigma sqrt(dt) Z)` on a regular grid.

    Each parameter is one number (one contract) or an array of one value per contract, shape `(n_contracts,)`.
    `simulate_paths` then gives `nb_predictions` paths for each contract.
    """

    def __init__(
        self,
        sigma: float | np.ndarray = 0.2,
        start_price: float | np.ndarray = 100.0,
        risk_free_rate: float | np.ndarray = 0.05,
        seed: int | np.random.Generator | None = None,
    ):
        """
        Initialize the Monte-Carlo simulator.

        Args:
            sigma: Volatility (standard deviation of returns), a number or one value per contract.
            start_price: Initial stock price, a number or one value per contract.
            risk_free_rate: Risk-free interest rate, a number or one value per contract.
            seed: An int reseeds the global `np.random` before each simulation, as the existing results were
                produced. A `np.random.Generator` is drawn from instead and leaves the global state alone.
        """
        self.sigma = sigma
        self.start_price = start_price
        self.risk_free_rate = risk_free_rate
        self.seed = seed
        self.simulations: np.ndarray | None = None

    def _contract_shape(self) -> tuple[int, ...]:
        """Return `()` if every parameter is a number, else `(n_contracts,)`, or raise ValueError."""
        shapes = {np.shape(getattr(self, name)) for name in PARAMETERS} - {()}
        if any(len(shape) != 1 for shape in shapes):
            raise ValueError(f"each parameter must be a number or a 1-D array, got shapes {sorted(shapes)}")
        if len(shapes) > 1:
            raise ValueError(f"per-contract parameters must all have the same length, got shapes {sorted(shapes)}")
        return shapes.pop() if shapes else ()

    def simulate_paths(self, nb_predictions: int = 10000, maturity: float = 1.0, sampling: int = 365) -> np.ndarray:
        """
        Simulate stock price paths using Geometric Brownian Motion (GBM).

        Time is the last axis, so `paths[c, p, k]` reads "contract `c`, path `p`, step `k`".

        Args:
            nb_predictions: Number of paths per contract.
            maturity: Time to maturity, in years, to simulate.
            sampling: Number of time steps per year (e.g., 365 for daily).

        Returns:
            self.simulations: shape `(nb_predictions, n_steps + 1)` for one contract, or
                `(n_contracts, nb_predictions, n_steps + 1)` with per-contract parameters, where
                `n_steps = int(maturity * sampling)`. Step 0 is the start price and the last step sits exactly
                at `maturity` (`n_steps` intervals of `maturity / n_steps`).

        Raises:
            ValueError: if `maturity * sampling < 1`, or the parameters are not numbers or 1-D arrays of one
                common length.
        """
        if isinstance(self.seed, np.random.Generator):
            standard_normal = self.seed.standard_normal
        else:
            if self.seed is not None:
                np.random.seed(self.seed)
            standard_normal = np.random.standard_normal

        # Rounded before truncating: 3/365 * 365 is 2.9999999999999996 in floating point and must give 3.
        size = int(round(maturity * sampling, 9))  # number of time intervals
        if size < 1:
            raise ValueError("maturity * sampling must be >= 1")
        dt = maturity / size
        contract_shape = self._contract_shape()
        # One column per contract, (n_contracts, 1), so broadcasting applies it to every path of that contract.
        sigma, start_price, rate = (
            np.broadcast_to(np.asarray(getattr(self, name), dtype=float), contract_shape)[..., None]
            for name in PARAMETERS
        )

        # Filled one step at a time along a leading time axis, where each step is contiguous in memory, then
        # moved to the last axis. For one contract the draws are the same calls as before the move, so seeded
        # results are unchanged.
        batch_shape = contract_shape + (nb_predictions,)
        paths = np.empty((size + 1, *batch_shape))
        paths[0] = start_price
        for i in range(1, size + 1):
            Z = standard_normal(batch_shape)
            paths[i] = paths[i - 1] * np.exp((rate - 0.5 * sigma**2) * dt + (sigma * np.sqrt(dt) * Z))

        self.simulations = np.ascontiguousarray(np.moveaxis(paths, 0, -1))
        return self.simulations

    def plot_paths(self) -> None:
        """
        Plot the first 50 simulated paths.
        """
        if self.simulations is None:
            self.simulate_paths()

        paths = self.simulations.reshape(-1, self.simulations.shape[-1])
        plt.figure(figsize=(10, 6))
        for path in paths[:50]:
            plt.plot(path, lw=1, alpha=0.5)

        plt.xlabel("Time Step")
        plt.ylabel("Price")
        plt.title("Monte-Carlo Simulation of Stock Price")
        plt.grid(True)
        plt.show()
