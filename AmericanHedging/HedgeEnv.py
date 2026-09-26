import numpy as np
from AmericanOptionPricing import LongstaffSchwartz

class HedgingEnv:
    def __init__(self, lsm_model: LongstaffSchwartz, transaction_cost: float = 0.001):
        self.lsm         = lsm_model
        self.option_type = lsm_model.option_type
        self.tc          = transaction_cost
        self.n_steps     = lsm_model.simulation.shape[0]
        self.n_paths     = lsm_model.simulation.shape[1]

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
            t / self.n_steps,                             # normalized time
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

        self.position = action
        self.t       += 1
        done          = self.t >= self.T_exercise

        if done:
            if self.option_type == "put":
                option_payoff = max(self.lsm.strike - S_t1, 0)
            else:
                option_payoff = max(S_t1 - self.lsm.strike, 0)
            reward = (hedge_pnl - cost - option_payoff) / self.lsm.strike
        else:
            reward = (hedge_pnl - cost) / self.lsm.strike

        return self._get_state(), reward, done
