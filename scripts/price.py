from optionpricing.core.gbm import GBM
from optionpricing.pricing.american.binomial_tree import BinomialTree
from optionpricing.pricing.american.longstaff_schwartz import LongstaffSchwartz
from optionpricing.pricing.european.black_scholes import BlackScholes
from optionpricing.pricing.european.monte_carlo import MonteCarlo


def add_arguments(parser):
    parser.add_argument("--sigma", default=0.2, type=float, help="Volatility (standard deviation of returns).")
    parser.add_argument("--start_price", default=100.0, type=float, help="Initial stock price.")
    parser.add_argument("--maturity", default=1.0, type=float, help="Time to maturity, in years.")
    parser.add_argument(
        "--sampling", default=365, type=int, help="Number of time steps per year (e.g., 365 for daily)."
    )
    parser.add_argument("--nb_predictions", default=10000, type=int, help="Number of simulation paths.")
    parser.add_argument("--nb_steps", default=1000, type=int, help="Number of steps for Binomial tree.")
    parser.add_argument("--strike", default=100.0, type=float, help="Strike price of the option.")
    parser.add_argument("--risk_free_rate", default=0.05, type=float, help="Risk-free interest rate.")
    parser.add_argument(
        "--option_type", default="call", type=str, choices=["call", "put"], help="Type of option ('call' or 'put')."
    )
    parser.add_argument(
        "--option_class", default="EU", type=str, choices=["EU", "US", "both"], help="Class of option ('EU' or 'US')."
    )
    parser.add_argument(
        "--max_degree",
        default=3,
        type=int,
        help="Maximum degree of the polynomial used in Longstaff-Schwartz regression.",
    )
    parser.add_argument(
        "--reg_type",
        default="simple",
        type=str,
        choices=["simple", "laguerre"],
        help="polynomials used for regression.",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument(
        "--plot", action="store_true", help="Plot the first 50 simulated paths of Monte Carlo simulation."
    )


def run(args):
    gbm = GBM(
        sigma=args.sigma,
        start_price=args.start_price,
        risk_free_rate=args.risk_free_rate,
        seed=args.seed,
    )
    simulation = gbm.simulate_paths(args.nb_predictions, args.maturity, args.sampling)

    if args.plot:
        gbm.plot_paths()

    if args.option_class in ("EU", "both"):
        mc = MonteCarlo(
            simulation,
            strike=args.strike,
            risk_free_rate=args.risk_free_rate,
            option_type=args.option_type,
        )
        bs = BlackScholes(
            sigma=args.sigma,
            start_price=args.start_price,
            strike=args.strike,
            risk_free_rate=args.risk_free_rate,
            option_type=args.option_type,
        )

        option_price_mc = mc.calculate_option_price(args.maturity)
        print(f"Estimated {args.option_type} option price with Monte-Carlo : {option_price_mc:.2f}")
        option_price_bs = bs.calculate_option_price(args.maturity)
        print(f"Estimated {args.option_type} option price with Black and Scholes : {option_price_bs:.2f}")

    if args.option_class in ("US", "both"):
        bin_tree = BinomialTree(
            sigma=args.sigma,
            start_price=args.start_price,
            strike=args.strike,
            risk_free_rate=args.risk_free_rate,
            maturity=args.maturity,
            nb_steps=args.nb_steps,
            option_type=args.option_type,
        )
        bin_tree.predict_payoff()
        option_price_bin = bin_tree.calculate_option_price(args.maturity)
        print(f"Estimated {args.option_type} option price with Binomial tree : {option_price_bin:.2f}")

        lsm = LongstaffSchwartz(
            simulation=simulation,
            strike=args.strike,
            risk_free_rate=args.risk_free_rate,
            maturity=args.maturity,
            option_type=args.option_type,
            reg_type=args.reg_type,
            max_degree=args.max_degree,
        )
        lsm.predict_payoff()
        option_price_lsm = lsm.calculate_option_price()
        print(f"Estimated {args.option_type} option price with Longstaff-Schwartz : {option_price_lsm:.2f}")
