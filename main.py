import argparse

from scripts import price, test_hedge_legacy, test_ml, train_hedge_legacy, train_ml

COMMANDS = {
    "price": price,
    "train-ml": train_ml,
    "test-ml": test_ml,
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
        "train-hedge-legacy": "Train the legacy SAC hedging policy (LSM exercise and deltas).",
        "test-hedge-legacy": "Evaluate the legacy SAC hedging policy or its baselines.",
    }

    for name, module in COMMANDS.items():
        module.add_arguments(subparsers.add_parser(name, help=subparsers_help[name]))

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    COMMANDS[args.command].run(args)


if __name__ == "__main__":
    main()
