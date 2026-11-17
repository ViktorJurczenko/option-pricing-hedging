import argparse

from optionpricing.core.keep_awake import keep_awake
from scripts import price, test_hedge, test_hedge_legacy, test_ml, train_hedge, train_hedge_legacy, train_ml

COMMANDS = {
    "price": price,
    "train-ml": train_ml,
    "test-ml": test_ml,
    "train-hedge": train_hedge,
    "test-hedge": test_hedge,
    "train-hedge-legacy": train_hedge_legacy,
    "test-hedge-legacy": test_hedge_legacy,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Option Pricing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers_help = {
        "price": "Compare option pricing models (Black-Scholes, Monte-Carlo, Binomial tree, Longstaff-Schwartz).",
        "train-ml": "Train the MLP surrogate pricer for American options.",
        "test-ml": "Evaluate the MLP put surrogate against an independent binomial tree.",
        "train-hedge": "Train the Deep Hedging band network on a bank of hedging scenarios.",
        "test-hedge": "Score the Deep Hedging network and its baselines on a test bank.",
        "train-hedge-legacy": "Train the legacy SAC hedging policy (LSM exercise and deltas).",
        "test-hedge-legacy": "Evaluate the legacy SAC hedging policy or its baselines.",
    }

    for name, module in COMMANDS.items():
        module.add_arguments(subparsers.add_parser(name, help=subparsers_help[name]))

    return parser


def main(argv=None):
    """Parse the command line and run the chosen command, with the PC held awake until it ends.

    A training or a test can run overnight, and on this PC the screen turning off suspends it (`keep_awake`).

    Args:
        argv: the arguments after `main.py`. None reads them from the command line.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    with keep_awake():
        COMMANDS[args.command].run(args)


if __name__ == "__main__":
    main()
