from core.gbm import GBM
from AmericanOptionPricing.LongstaffSchwartz import LongstaffSchwartz
from AmericanHedging.Critic import Critic
from AmericanHedging.HedgeEnv import HedgingEnv
from AmericanHedging.ReplayBuffer import ReplayBuffer
from AmericanHedging.RLPolicy import RLPolicy
from AmericanHedging.Hedge import train

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
    parser.add_argument("--replay_buffer_size", default=100_000, type=int, help="Max number of transitions kept in the replay buffer.")
    parser.add_argument("--fixed_cost", default=0.0, type=float, help="Flat cost charged whenever the hedge position is changed, regardless of trade size.")
    parser.add_argument("--risk_lambda", default=0.0, type=float, help="Mean-variance penalty coefficient: the critic/actor train on reward - risk_lambda * reward**2, while logged/checkpointed reward stays raw P&L.")
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

    lsm = LongstaffSchwartz(simulation, args.strike, args.risk_free_rate, args.maturity, args.option_type, args.reg_type, args.max_degree)
    lsm.predict_payoff()
    env = HedgingEnv(lsm, fixed_cost=args.fixed_cost)

    rb = ReplayBuffer(max_size=args.replay_buffer_size)
    critic1 = Critic(5, 1)
    critic2 = Critic(5, 1)

    train(
        policy_model, env, rb, critic1, critic2,
        n_episodes=args.n_episodes, warmup_steps=args.warmup_steps, update_every=args.update_every,
        grad_steps_per_update=args.grad_steps_per_update, risk_lambda=args.risk_lambda,
        checkpoint_path=args.checkpoint_path,
    )
