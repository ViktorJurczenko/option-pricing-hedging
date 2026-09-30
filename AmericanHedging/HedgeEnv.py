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

    def reset(self):
        self.path_idx   = np.random.randint(self.n_paths)
        self.t          = 0
        self.position   = 0.0
        self.T_exercise = self.lsm.exercise_time[self.path_idx]
        return self._get_state()

    def _get_state(self):
        t, i      = self.t, self.path_idx
        t_clamped = min(t, self.lsm.delta.shape[0] - 1)  # greeks undefined at maturity
        return np.array([
            self.lsm.simulation[t, i] / self.lsm.strike,  # normalized spot
            t / (self.n_steps - 1),                       # normalized time (1 at expiry)
            self.lsm.delta[t_clamped, i],                 # L-S delta
            self.lsm.gamma[t_clamped, i],                 # L-S gamma
            self.position,                                # current hedge held
        ])

    def step(self, action: float):
        action = np.clip(action, -1, 1) # enforce valid hedge ratio

        i    = self.path_idx
        S_t  = self.lsm.simulation[self.t,     i]
        S_t1 = self.lsm.simulation[self.t + 1, i]

        trade     = action - self.position
        hedge_pnl = self.position * (S_t1 - S_t)
        cost      = self.tc * abs(trade) * S_t
        if abs(trade) > self._TRADE_EPS:
            cost += self.fixed_cost

        t             = self.t
        self.position = action
        self.t       += 1
        done          = self.t >= self.T_exercise

        if done:
            if self.option_type == "put":
                option_payoff = max(self.lsm.strike - S_t1, 0)
            else:
                option_payoff = max(S_t1 - self.lsm.strike, 0)

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
