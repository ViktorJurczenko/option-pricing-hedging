import numpy as np

from tqdm import tqdm

def cvar(pnls: np.ndarray | list[float], level: float = 0.95) -> float:
    """Conditional value at risk (expected shortfall) of a P&L sample: the mean
    of its worst `1 - level` fraction, at least one outcome. Always <= the VaR
    (the `1 - level` percentile), which ignores how bad the tail is past it."""
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    pnls  = np.sort(np.asarray(pnls, dtype=np.float64))
    if pnls.size == 0:
        raise ValueError("cvar of an empty P&L sample is undefined")
    # round first: (1 - 0.95) * 100 is 5.000000000000004 in floating point, so a
    # bare ceil would count 6 outcomes instead of 5
    n_bad = max(1, int(np.ceil(round((1 - level) * len(pnls), 9))))
    return float(pnls[:n_bad].mean())

def evaluate(action_fn, env, n_episodes=1000):
    """
    action_fn: state (np.ndarray) -> hedge position (float). For a trained RL
    policy, wrap `ML.rl.sac.deterministic_hedge(policy, s).item()`, as
    scripts/test_hedge.py does; for a baseline, pass
    one of Hedging.baselines directly - they already match this
    signature.

    Returns (mean_pnl, std_pnl, cvar_95); the 95% VaR is printed too.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    pnls = []
    for ep in tqdm(range(n_episodes), desc="Evaluating hedge policy", unit="episode"):
        state, done, total_reward = env.reset(), False, 0
        while not done:
            action = action_fn(state)
            state, reward, done = env.step(action)
            total_reward += reward
        pnls.append(total_reward)

    mean_pnl = np.mean(pnls)
    std_pnl  = np.std(pnls)
    var_95   = np.percentile(pnls, 5)
    cvar_95  = cvar(pnls)

    print(f"Mean P&L:          {mean_pnl:.4f}")
    print(f"Std  P&L (error):  {std_pnl:.4f}")   # your hedging error
    print(f"VaR  95%:          {var_95:.4f}")
    print(f"CVaR 95%:          {cvar_95:.4f}")

    return mean_pnl, std_pnl, cvar_95
