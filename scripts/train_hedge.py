import argparse
import math

from core.gbm import GBM
from AmericanOptionPricing.LongstaffSchwartz import LongstaffSchwartz
from AmericanHedging.Critic import Critic
from AmericanHedging.HedgeEnv import HedgingEnv
from AmericanHedging.ReplayBuffer import ReplayBuffer
from AmericanHedging.RLPolicy import RLPolicy
from AmericanHedging.Hedge import train
from AmericanHedging.option_value import make_option_value_fn

def _positive_float(value: str) -> float:
    """argparse type for a finite float that must be > 0 (argparse turns
    the ValueError into a usage error naming the flag)."""
    number = float(value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{value} is not a finite number > 0")
    return number

def add_arguments(parser):
    parser.add_argument("--sigma", default=0.2, type=float, help="Volatility (standard deviation of returns).")
    parser.add_argument("--start_price", default=100.0, type=float, help="Initial stock price.")
    parser.add_argument("--maturity", default=1.0, type=float, help="Time to maturity, in years.")
    parser.add_argument("--sampling", default=365, type=int, help="Number of time steps per year (e.g., 365 for daily).")
    parser.add_argument("--nb_predictions", default=10000, type=int, help="Number of simulation paths.")
    parser.add_argument("--strike", default=100.0, type=float, help="Strike price of the option.")
    parser.add_argument("--risk_free_rate", default=0.05, type=float, help="Risk-free interest rate.")
    parser.add_argument("--option_type", default="call", type=str, choices=["call", "put"], help="Type of option ('call' or 'put').")
    parser.add_argument("--max_degree", default=3, type=int, help="Maximum degree of the polynomial used in Longstaff-Schwartz regression.")
    parser.add_argument("--reg_type", default="simple", type=str, choices=["simple", "laguerre"], help="polynomials used for regression.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument("--n_episodes", default=1000, type=int, help="Number of training episodes.")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="Number of random-action steps collected before learning starts.")
    parser.add_argument("--update_every", default=10, type=int, help="Number of environment steps between gradient updates.")
    parser.add_argument("--grad_steps_per_update", default=1, type=int, help="Number of critic/actor gradient steps taken each time an update is triggered.")
    parser.add_argument("--lr", default=1e-4, type=float, help="Adam learning rate for both the actor and the twin critics.")
    parser.add_argument("--alpha", default=0.2, type=float, help="SAC entropy coefficient: weight of the entropy bonus (-alpha * log_prob) in the actor loss.")
    parser.add_argument("--replay_buffer_size", default=100_000, type=int, help="Max number of transitions kept in the replay buffer.")
    parser.add_argument("--prop_tc", default=0.001, type=float, help="Proportional trading cost: tc * |trade| * S_t is charged on every rebalance (training and validation envs).")
    parser.add_argument("--fixed_tc", default=0.0, type=float, help="Flat cost charged whenever the hedge position is changed, regardless of trade size.")
    parser.add_argument("--risk_lambda", default=1.0, type=_positive_float, help="Mean-variance penalty coefficient: the critic/actor train on x - risk_lambda * x**2 with x = reward_scale * reward, while logged/checkpointed reward stays raw P&L. Must be > 0 for the policy to have any incentive to hedge (every hedge has ~the same expected P&L; hedging only reduces its variance).")
    parser.add_argument("--reward_mode", default="accounting", type=str, choices=["terminal", "accounting"], help="'accounting' subtracts the option's value change every step, so each step's reward is the local hedging error (what makes --risk_lambda a real mean-variance penalty); 'terminal' only subtracts the payoff at exercise.")
    parser.add_argument("--gamma", default=1.0, type=float, help="SAC discount factor. Episodes are finite and time is in the state, so no discounting is needed by default.")
    parser.add_argument("--reward_scale", default=100.0, type=float, help="Multiplier on the env reward before the risk penalty, so per-step rewards (~1e-2) aren't drowned out by the entropy term.")
    parser.add_argument("--auto_alpha", default=True, action=argparse.BooleanOptionalAction, help="Learn the SAC entropy coefficient toward a target entropy of -action_dim, i.e. -1, or -2 with --action_mode band (--alpha is then its initial value). --no-auto_alpha keeps alpha fixed.")
    parser.add_argument("--alpha_lr", default=None, type=float, help="Adam learning rate for the learned entropy coefficient (with --auto_alpha; ignored with --no-auto_alpha). Defaults to --lr.")
    parser.add_argument("--residual_scale", default=0.1, type=float, help="The policy learns a correction to the LSM delta: hedge = clip(delta + residual_scale * tanh(output), -1, 1). Saved in the checkpoint, so test-hedge picks it up. 0 makes the policy output the hedge ratio directly.")
    parser.add_argument("--action_mode", default="residual", type=str, choices=["residual", "gap_fraction", "band"], help="How the policy's output becomes a hedge. residual: delta + residual_scale * tanh(output). gap_fraction: position + w * (delta - position) with w in (0, 1), the fraction of the gap to delta closed each step. band: a learned no-trade band - a 2-D action picks the band half-width (up to --band_max) and how far into the band to rebalance (edge to delta); inside the band the position is held exactly, so no fixed cost is paid. Saved in the checkpoint.")
    parser.add_argument("--band_max", default=0.2, type=float, help="--action_mode band only (ignored in the other modes): the widest no-trade band half-width the policy can pick. Saved in the checkpoint.")
    parser.add_argument("--val_paths", default=1000, type=int, help="Number of held-out validation paths the checkpoint is selected on (P&L std or CVaR, see --val_metric; the untrained policy is a candidate too). 0 falls back to the best trailing 50-episode mean training reward.")
    parser.add_argument("--val_every", default=50, type=int, help="Episodes between validation scorings (the last episode is always scored).")
    parser.add_argument("--val_metric", default="std", type=str, choices=["std", "cvar"], help="Checkpoint criterion on the validation P&L: lowest std (hedging error) or highest 95%% CVaR.")
    parser.add_argument("--val_seed", default=None, type=int, help="Seed of the validation paths. Defaults to --seed + 1000; keep it away from the seed you evaluate on with test-hedge.")
    parser.add_argument("--checkpoint_path", default="checkpoints/best_policy_model.pth", type=str, help="Where to save the best policy.")

def _simulate_and_fit_lsm(args, seed):
    gbm = GBM(
        sigma=args.sigma,
        start_price=args.start_price,
        risk_free_rate=args.risk_free_rate,
        seed=seed,
    )
    simulation = gbm.simulate_paths(args.nb_predictions, args.maturity, args.sampling)
    lsm = LongstaffSchwartz(
        simulation, args.strike, args.risk_free_rate, args.maturity, args.option_type, args.reg_type, args.max_degree,
        sigma=args.sigma,
        # European control variate only for calls: it removes plain LSM's spurious
        # early exercise there; for puts the plain rule is as good or better (BUGS.md).
        control_variate=args.option_type == "call",
    )
    lsm.predict_payoff()
    return simulation, lsm

def run(args):
    # Built first: GBM reseeds the global np.random, and the training
    # simulation's reseed below must be the last one, so the training stream
    # (env.reset, warmup actions) stays the one --seed gives.
    val_env = None
    if args.val_paths > 0:
        val_seed = args.seed + 1000 if args.val_seed is None else args.val_seed
        _, val_lsm = _simulate_and_fit_lsm(args, val_seed)
        # Terminal mode: the validation metric is then test-hedge's P&L.
        val_env = HedgingEnv(val_lsm, prop_tc=args.prop_tc, fixed_tc=args.fixed_tc)
        if args.val_paths > val_env.n_paths:
            print(f"Warning: --val_paths {args.val_paths} exceeds the {val_env.n_paths} simulated paths "
                  f"(--nb_predictions); validating on all {val_env.n_paths}.")

    simulation, lsm = _simulate_and_fit_lsm(args, args.seed)

    # State has shape [S_t/K, t/T, delta, gamma, current position]
    # band mode's action is (band width, rebalancing target); the others are 1-D.
    action_dim = 2 if args.action_mode == "band" else 1
    policy_model = RLPolicy(5, action_dim, residual_scale=args.residual_scale, action_mode=args.action_mode,
                            band_max=args.band_max)

    value_fn = None
    if args.reward_mode == "accounting":
        value_fn = make_option_value_fn(
            args.option_type, args.start_price, args.strike, args.risk_free_rate,
            args.sigma, args.maturity, simulation.shape[0],
        )
    env = HedgingEnv(lsm, prop_tc=args.prop_tc, fixed_tc=args.fixed_tc, reward_mode=args.reward_mode, value_fn=value_fn)

    rb = ReplayBuffer(max_size=args.replay_buffer_size)
    critic1 = Critic(5, action_dim)
    critic2 = Critic(5, action_dim)

    train(
        policy_model, env, rb, critic1, critic2,
        n_episodes=args.n_episodes, warmup_steps=args.warmup_steps, update_every=args.update_every,
        grad_steps_per_update=args.grad_steps_per_update, risk_lambda=args.risk_lambda,
        lr=args.lr, alpha=args.alpha, gamma=args.gamma,
        reward_scale=args.reward_scale, auto_alpha=args.auto_alpha, alpha_lr=args.alpha_lr,
        val_env=val_env, val_paths=args.val_paths, val_every=args.val_every, val_metric=args.val_metric,
        checkpoint_path=args.checkpoint_path,
    )
