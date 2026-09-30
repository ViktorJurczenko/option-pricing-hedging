from typing import Callable

import numpy as np
from AmericanOptionPricing import LongstaffSchwartz

class HedgingEnv:
    # Trades smaller than this are treated as "no trade" so a fixed cost isn't
    # charged for floating-point noise around holding the same position.
    _TRADE_EPS = 1e-8

    def __init__(self, lsm_model: LongstaffSchwartz, transaction_cost: float = 0.001, fixed_cost: float = 0.0,
                 reward_mode: str = "terminal", value_fn: Callable[[int, np.ndarray], np.ndarray] | None = None):
        """
        reward_mode:
            "terminal" (default): per-step reward is hedge P&L minus costs, and
                the option payoff is subtracted on the exercise step only. The
                episode sum is the hedger's total P&L (what test-hedge reports).
            "accounting": each step additionally subtracts the change in the
                option's value, -(V_{t+1} - V_t), with V replaced by the payoff
                on the exercise step. Each step's reward is then the local
                hedging error, which is what makes a per-step variance penalty
                (train()'s risk_lambda) a real mean-variance objective instead
                of a penalty on holding any position. The episode sum telescopes
                to the terminal-mode sum plus the constant V_0/K.
        value_fn: value_fn(t, S) -> option value at simulation row t for spots S
            (see AmericanHedging/option_value.py). Required for "accounting".
        """
        if reward_mode not in ["terminal", "accounting"]:
            raise ValueError("reward_mode must be 'terminal' or 'accounting'")
        if reward_mode == "accounting" and value_fn is None:
            raise ValueError("reward_mode='accounting' requires a value_fn")

        self.lsm         = lsm_model
        self.option_type = lsm_model.option_type
        self.tc          = transaction_cost
        self.fixed_cost  = fixed_cost
        self.reward_mode = reward_mode
        self.n_steps     = lsm_model.simulation.shape[0]
        self.n_paths     = lsm_model.simulation.shape[1]

        if reward_mode == "accounting":
            # Precomputed once for every (row, path) so step() stays a lookup.
            self.option_values = np.stack([
                value_fn(t, lsm_model.simulation[t]) for t in range(self.n_steps)
            ])

    def reset(self, path_idx: int | None = None):
        """Start an episode on path `path_idx`, or on a random path if None."""
        self.path_idx   = np.random.randint(self.n_paths) if path_idx is None else path_idx
        self.t          = 0
        self.position   = 0.0
        self.T_exercise = self.lsm.exercise_time[self.path_idx]
        return self._get_state()

    def _get_state(self):
        return self._build_states(self.t, self.path_idx, self.position)

    def _build_states(self, t, path_idx, position):
        """State(s) at row t for path index/indices `path_idx` holding `position`.
        Scalar path_idx/position give shape (5,); arrays give shape (n, 5)."""
        t_clamped = min(t, self.lsm.delta.shape[0] - 1)  # greeks undefined at maturity
        spot      = self.lsm.simulation[t, path_idx]
        return np.stack(np.broadcast_arrays(
            spot / self.lsm.strike,             # normalized spot
            t / (self.n_steps - 1),             # normalized time (1 at expiry)
            self.lsm.delta[t_clamped, path_idx],  # L-S delta
            self.lsm.gamma[t_clamped, path_idx],  # L-S gamma
            position,                           # current hedge held
        ), axis=-1).astype(np.float64)

    def _trading_cost(self, trade: float | np.ndarray, S_t: float | np.ndarray) -> float | np.ndarray:
        """Proportional cost plus the fixed cost on any real trade (scalar or array).
        Shared by step() and rollout_pnl() so the two can't drift apart."""
        return self.tc * np.abs(trade) * S_t + self.fixed_cost * (np.abs(trade) > self._TRADE_EPS)

    def _payoff(self, S: float | np.ndarray) -> float | np.ndarray:
        """Option payoff at spot(s) S. Shared by step() and rollout_pnl()."""
        if self.option_type == "put":
            return np.maximum(self.lsm.strike - S, 0)
        return np.maximum(S - self.lsm.strike, 0)

    def rollout_pnl(self, hedge_fn: Callable[[np.ndarray], np.ndarray], path_idxs: np.ndarray) -> np.ndarray:
        """Terminal-mode P&L (units of K) of one episode per path in `path_idxs`,
        all stepped together: hedge_fn maps a batch of states (n, 5) to hedge
        ratios (n,), so a policy is called once per time step instead of once
        per path and step.

        Must match (up to float rounding) summing step()'s terminal-mode
        rewards over an episode on each path, whatever the env's reward_mode:
        it shares _trading_cost/_payoff with step() and replicates the rest of
        its arithmetic in vectorized form, including step()'s
        convention that the P&L over [t, t+1] is earned by the position held
        *before* the trade at t. Doesn't touch the env's episode state or
        np.random.
        """
        path_idxs  = np.asarray(path_idxs)
        sim        = self.lsm.simulation
        T_exercise = self.lsm.exercise_time[path_idxs]
        position   = np.zeros(len(path_idxs))
        pnl        = np.zeros(len(path_idxs))

        for t in range(int(T_exercise.max())):
            active = t < T_exercise
            idx    = path_idxs[active]
            states = self._build_states(t, idx, position[active])
            action = np.clip(np.asarray(hedge_fn(states), dtype=np.float64).reshape(-1), -1, 1)

            S_t, S_t1 = sim[t, idx], sim[t + 1, idx]
            step_pnl  = position[active] * (S_t1 - S_t) - self._trading_cost(action - position[active], S_t)

            done = t + 1 >= T_exercise[active]
            step_pnl -= np.where(done, self._payoff(S_t1), 0.0)

            pnl[active]      += step_pnl
            position[active]  = action

        return pnl / self.lsm.strike

    def step(self, action: float):
        action = np.clip(action, -1, 1) # enforce valid hedge ratio

        i    = self.path_idx
        S_t  = self.lsm.simulation[self.t,     i]
        S_t1 = self.lsm.simulation[self.t + 1, i]

        trade     = action - self.position
        hedge_pnl = self.position * (S_t1 - S_t)
        cost      = self._trading_cost(trade, S_t)

        t             = self.t
        self.position = action
        self.t       += 1
        done          = self.t >= self.T_exercise

        if done:
            option_payoff = self._payoff(S_t1)

        if self.reward_mode == "accounting":
            V_t  = self.option_values[t, i]
            # option_payoff is only defined when done; the conditional
            # expression never evaluates it otherwise. On the exercise step the
            # option is worth its payoff, not its continuation value V.
            V_t1 = option_payoff if done else self.option_values[t + 1, i]
            reward = (hedge_pnl - cost - (V_t1 - V_t)) / self.lsm.strike
        elif done:
            reward = (hedge_pnl - cost - option_payoff) / self.lsm.strike
        else:
            reward = (hedge_pnl - cost) / self.lsm.strike

        return self._get_state(), reward, done
