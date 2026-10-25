import argparse

from scripts import price, train_ml, test_ml, train_hedge, test_hedge

COMMANDS = {
    "price": price,
    "train-ml": train_ml,
    "test-ml": test_ml,
    "train-hedge": train_hedge,
    "test-hedge": test_hedge,
}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Option Pricing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers_help = {
        "price": "Compare option pricing models (Black-Scholes, Monte-Carlo, Binomial tree, Longstaff-Schwartz).",
        "train-ml": "Train the MLP surrogate pricer for American options.",
        "test-ml": "Evaluate the MLP put surrogate against an independent binomial tree.",
        "train-hedge": "Train the RL delta-hedging policy.",
        "test-hedge": "Evaluate the RL delta-hedging policy.",
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
