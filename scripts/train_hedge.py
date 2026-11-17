"""The train-hedge command: train the Deep Hedging band network on a train bank, keeping its best epoch on a
validation bank.

One flag per field of `HedgeTrainConfig`, of its `HedgeSettings` and of its `RiskMeasure`, with the defaults read
from them, so the CLI and the library never drift apart. The banks come from
`python -m optionpricing.ml.hedging.generate_hedge_dataset` (`--split train` and `--split val`).
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from optionpricing.ml.hedging.hedge_model import ACTIVATIONS
from optionpricing.ml.hedging.hedge_training import HedgeTrainConfig, train_hedging_model
from scripts.hedge_flags import add_settings_arguments, describe_score, describe_settings, resolve_settings

# Where train-hedge saves the network and test-hedge loads it.
CHECKPOINT_PATH = "artifacts/checkpoints/hedge_network.pth"
TRAIN_PATH, VALID_PATH = "artifacts/data/hedge_train.npz", "artifacts/data/hedge_val.npz"
# How a missing bank is made, for the error message.
GENERATE = "python -m optionpricing.ml.hedging.generate_hedge_dataset --split {split}"


def add_arguments(parser):
    defaults = HedgeTrainConfig()
    parser.add_argument("--train_path", default=TRAIN_PATH, type=str, help="The training bank (--split train).")
    parser.add_argument(
        "--valid_path",
        default=VALID_PATH,
        type=str,
        help="The validation bank (--split val): it chooses the best epoch.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=CHECKPOINT_PATH,
        type=str,
        help="Where the best network is saved, with its config (test-hedge's default).",
    )
    parser.add_argument(
        "--history_path",
        default=None,
        type=str,
        help="A JSON file for the config and the per-epoch history (NaN where a value does not apply). "
        "Default: none written.",
    )
    group = parser.add_argument_group("network", "The band network (BandNetwork).")
    group.add_argument("--depth", default=defaults.depth, type=int, help="Hidden layers.")
    group.add_argument("--width", default=defaults.width, type=int, help="Units per hidden layer.")
    group.add_argument(
        "--init_logit",
        default=defaults.init_logit,
        type=float,
        help="Initial pre-softplus half-width of the band: the untrained network is the delta hedge within it.",
    )
    group.add_argument(
        "--activation", default=defaults.activation, choices=sorted(ACTIVATIONS), help="Hidden-layer activation."
    )
    group.add_argument(
        "--boundary_input",
        action="store_true",
        help="Add the distance to the exercise boundary as an 8th input.",
    )
    add_settings_arguments(parser, inherit=False)
    group = parser.add_argument_group("optimization")
    group.add_argument("--epochs", default=defaults.epochs, type=int, help="Full passes over the training bank.")
    group.add_argument(
        "--batch_size",
        default=defaults.batch_size,
        type=int,
        help="Scenarios per Adam step. CVaR is a property of the batch's tail, so batches are large.",
    )
    group.add_argument("--lr", default=defaults.lr, type=float, help="The network's Adam learning rate.")
    group.add_argument(
        "--w_lr", default=defaults.w_lr, type=float, help="The CVaR threshold's Adam learning rate (--risk cvar)."
    )
    group.add_argument(
        "--seed", default=defaults.seed, type=int, help="Seeds the weights' initialization and the batch order."
    )


def train_config(args) -> HedgeTrainConfig:
    """The `HedgeTrainConfig` of the parsed flags.

    Raises:
        ValueError: as `HedgeTrainConfig`, `HedgeSettings` and `RiskMeasure`, for an invalid value.
    """
    hedging, risk = resolve_settings(args)
    names = ("depth", "width", "init_logit", "activation", "boundary_input", "epochs", "batch_size", "lr", "w_lr")
    return HedgeTrainConfig(**{name: getattr(args, name) for name in names}, hedging=hedging, risk=risk, seed=args.seed)


def check_banks(paths: dict[str, tuple[str, str]]) -> None:
    """Refuse a bank flag that names no file, with the command that generates the bank.

    Args:
        paths: `{flag: (path, split)}`.

    Raises:
        FileNotFoundError: naming the flag, the path and the generator command.
    """
    for flag, (path, split) in paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"{flag}: no file at {path} (generate it with `{GENERATE.format(split=split)}`).")


def run(args):
    # before any work: a bad flag or a missing bank fails here, not after the training bank is loaded
    config = train_config(args)
    check_banks({"--train_path": (args.train_path, "train"), "--valid_path": (args.valid_path, "val")})
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Training on {describe_settings(config.hedging, config.risk)}")

    start_time = time.time()
    report = train_hedging_model(args.train_path, args.valid_path, config, args.checkpoint_path)
    best, start = report.history[report.best_epoch], report.history[0]
    print(
        f"Best epoch {report.best_epoch} of {config.epochs}: validation {describe_score(config.risk, best['valid_score'])}, "
        f"mean P&L {best['valid_mean_pnl']:+.4f} $, mean costs {best['valid_mean_costs']:.4f} $"
    )
    print(f"  untrained network (delta within its initial band): {describe_score(config.risk, start['valid_score'])}")
    if report.best_epoch == 0:
        print("  Training never beat the untrained network: the checkpoint is the untrained one.")
    if args.history_path is not None:
        Path(args.history_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.history_path, "w", encoding="utf-8") as file:
            # NaN is written as NaN, which json.load reads back
            json.dump({"config": asdict(config), "best_epoch": report.best_epoch, "history": report.history}, file)
        print(f"History was saved as : {args.history_path}")
    print(f"Model was trained in : {(time.time() - start_time):.2f}s")
    print(f"Model was saved as : {args.checkpoint_path}")
