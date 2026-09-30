import numpy as np
import torch

from core.gbm import GBM
from AmericanOptionPricing.LongstaffSchwartz import LongstaffSchwartz
from AmericanHedging.HedgeEnv import HedgingEnv
from AmericanHedging.RLPolicy import RLPolicy
from AmericanHedging.Hedge import deterministic_hedge, evaluate
from AmericanHedging.baselines import delta_hedge, no_hedge

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
    parser.add_argument("--n_episodes", default=1000, type=int, help="Number of evaluation episodes.")
    parser.add_argument("--fixed_cost", default=0.0, type=float, help="Flat cost charged whenever the hedge position is changed, regardless of trade size. Should match the value used for --policy rl's training run for a fair comparison.")
    parser.add_argument("--checkpoint_path", default="checkpoints/best_policy_model.pth", type=str, help="Path to the trained policy's weights (only used with --policy rl).")
    parser.add_argument("--policy", default="rl", type=str, choices=["rl", "delta", "no_hedge"], help="Hedging policy to evaluate: the trained RL policy, an analytic delta-hedge baseline, or a no-hedge baseline.")

def run(args):
    gbm = GBM(
        sigma=args.sigma,
        start_price=args.start_price,
        risk_free_rate=args.risk_free_rate,
        seed=args.seed,
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
    env = HedgingEnv(lsm, fixed_cost=args.fixed_cost)

    if args.policy == "rl":
        # residual_scale comes from the checkpoint, so it always matches training.
        policy_model = RLPolicy(5, 1)
        policy_model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))
        def action_fn(state):
            # Same deterministic action train()'s validation checkpointing scores.
            return deterministic_hedge(policy_model, state).item()
    elif args.policy == "delta":
        action_fn = delta_hedge
    else:
        action_fn = no_hedge

    # Reseed right before evaluation so --policy rl/delta/no_hedge runs with the
    # same --seed draw identical simulated paths, making the comparison fair.
    np.random.seed(args.seed)
    evaluate(action_fn, env, n_episodes=args.n_episodes)
