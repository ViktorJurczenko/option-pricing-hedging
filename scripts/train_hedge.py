import argparse

from core.gbm import GBM
from AmericanOptionPricing.LongstaffSchwartz import LongstaffSchwartz
from AmericanHedging.Critic import Critic
from AmericanHedging.HedgeEnv import HedgingEnv
from AmericanHedging.ReplayBuffer import ReplayBuffer
from AmericanHedging.RLPolicy import RLPolicy
from AmericanHedging.Hedge import train
from AmericanHedging.option_value import make_option_value_fn

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
    parser.add_argument("--ridge_lambda", default=0.06, type=float, help="L2 regularization strength for the Longstaff-Schwartz regression. Default validated to reduce near-maturity delta/gamma overshoot (see BUGS.md) while keeping price drift vs. a Binomial Tree benchmark within the unregularized baseline's own natural error.")
    parser.add_argument("--moneyness_bins", default=30, type=int, help="Number of quantile bins (by in-the-money S_t) to split each near-maturity timestep's regression into, instead of one global fit. See moneyness_bins_window for which timesteps this applies to.")
    parser.add_argument("--moneyness_bins_window", default=0.3, type=float, help="Fraction of the timeline, counted backward from maturity, over which moneyness_bins is applied; outside that window a single global fit is used regardless of moneyness_bins.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument("--n_episodes", default=1000, type=int, help="Number of training episodes.")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="Number of random-action steps collected before learning starts.")
    parser.add_argument("--update_every", default=10, type=int, help="Number of environment steps between gradient updates.")
    parser.add_argument("--grad_steps_per_update", default=1, type=int, help="Number of critic/actor gradient steps taken each time an update is triggered.")
    parser.add_argument("--lr", default=1e-4, type=float, help="Adam learning rate for both the actor and the twin critics.")
    parser.add_argument("--alpha", default=0.2, type=float, help="SAC entropy coefficient: weight of the entropy bonus (-alpha * log_prob) in the actor loss.")
    parser.add_argument("--replay_buffer_size", default=100_000, type=int, help="Max number of transitions kept in the replay buffer.")
    parser.add_argument("--fixed_cost", default=0.0, type=float, help="Flat cost charged whenever the hedge position is changed, regardless of trade size.")
    parser.add_argument("--risk_lambda", default=1.0, type=float, help="Mean-variance penalty coefficient: the critic/actor train on x - risk_lambda * x**2 with x = reward_scale * reward, while logged/checkpointed reward stays raw P&L. Must be > 0 for the policy to have any incentive to hedge (every hedge has ~the same expected P&L; hedging only reduces its variance).")
    parser.add_argument("--reward_mode", default="accounting", type=str, choices=["terminal", "accounting"], help="'accounting' subtracts the option's value change every step, so each step's reward is the local hedging error (what makes --risk_lambda a real mean-variance penalty); 'terminal' only subtracts the payoff at exercise.")
    parser.add_argument("--gamma", default=1.0, type=float, help="SAC discount factor. Episodes are finite and time is in the state, so no discounting is needed by default.")
    parser.add_argument("--reward_scale", default=100.0, type=float, help="Multiplier on the env reward before the risk penalty, so per-step rewards (~1e-2) aren't drowned out by the entropy term.")
    parser.add_argument("--auto_alpha", default=True, action=argparse.BooleanOptionalAction, help="Learn the SAC entropy coefficient toward a target entropy of -1 (--alpha is then its initial value). --no-auto_alpha keeps alpha fixed.")
    parser.add_argument("--checkpoint_path", default="checkpoints/best_policy_model.pth", type=str, help="Where to save the best policy.")

def run(args):
    gbm = GBM(
        sigma=args.sigma,
        start_price=args.start_price,
        risk_free_rate=args.risk_free_rate,
        seed=args.seed,
    )
    simulation = gbm.simulate_paths(args.nb_predictions, args.maturity, args.sampling)

    policy_model = RLPolicy(5, 1)  # State has shape [S_t/K, t/T, delta, gamma, current position]

    lsm = LongstaffSchwartz(
        simulation, args.strike, args.risk_free_rate, args.maturity, args.option_type, args.reg_type, args.max_degree,
        ridge_lambda=args.ridge_lambda, moneyness_bins=args.moneyness_bins, moneyness_bins_window=args.moneyness_bins_window,
    )
    lsm.predict_payoff()
    value_fn = None
    if args.reward_mode == "accounting":
        value_fn = make_option_value_fn(
            args.option_type, args.start_price, args.strike, args.risk_free_rate,
            args.sigma, args.maturity, simulation.shape[0],
        )
    env = HedgingEnv(lsm, fixed_cost=args.fixed_cost, reward_mode=args.reward_mode, value_fn=value_fn)

    rb = ReplayBuffer(max_size=args.replay_buffer_size)
    critic1 = Critic(5, 1)
    critic2 = Critic(5, 1)

    train(
        policy_model, env, rb, critic1, critic2,
        n_episodes=args.n_episodes, warmup_steps=args.warmup_steps, update_every=args.update_every,
        grad_steps_per_update=args.grad_steps_per_update, risk_lambda=args.risk_lambda,
        lr=args.lr, alpha=args.alpha, gamma=args.gamma,
        reward_scale=args.reward_scale, auto_alpha=args.auto_alpha,
        checkpoint_path=args.checkpoint_path,
    )
