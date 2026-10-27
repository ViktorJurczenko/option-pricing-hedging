import time
from dataclasses import fields
from pathlib import Path

import torch

from ML.surrogate.tree.tree_model import ACTIVATIONS, OUTPUTS
from ML.surrogate.tree.tree_training import TrainConfig, train_surrogate

# One flag per TrainConfig field, defaults read from it, so the CLI and the library never drift apart.
def add_arguments(parser):
    parser.add_argument("--dataset_path", default="data/tree_put_dataset.npz", type=str,
                        help="The tree put dataset (python -m ML.surrogate.tree.generate_tree_dataset).")
    parser.add_argument("--checkpoint_path", default="checkpoints/put_surrogate.pth", type=str,
                        help="Where the best PutSurrogate is saved (test-ml's default).")
    # network
    parser.add_argument("--depth", default=TrainConfig.depth, type=int, help="Hidden layers.")
    parser.add_argument("--width", default=TrainConfig.width, type=int, help="Units per hidden layer.")
    parser.add_argument("--activation", default=TrainConfig.activation, choices=sorted(ACTIVATIONS),
                        help="Hidden-layer activation (smooth ones give meaningful autograd Greeks).")
    parser.add_argument("--output", default=TrainConfig.output, choices=OUTPUTS,
                        help="Output layer: linear, or softplus (premium forced > 0).")
    # data
    parser.add_argument("--split_ratio", default=TrainConfig.split_ratio, type=float,
                        help="Train/test split of the dataset (test-ml must use the same).")
    parser.add_argument("--valid_ratio", default=TrainConfig.valid_ratio, type=float,
                        help="Share of the training split held out for early stopping and the checkpoint choice.")
    # optimization
    parser.add_argument("--batch_size", default=TrainConfig.batch_size, type=int, help="Rows per Adam step.")
    parser.add_argument("--epochs", default=TrainConfig.epochs, type=int,
                        help="Epochs, also the length of the cosine learning-rate schedule.")
    parser.add_argument("--lr", default=TrainConfig.lr, type=float, help="Initial Adam learning rate.")
    parser.add_argument("--lr_min", default=TrainConfig.lr_min, type=float,
                        help="Learning rate the cosine schedule ends at.")
    parser.add_argument("--patience", default=TrainConfig.patience, type=int,
                        help="Epochs without a validation improvement before stopping early, counted only once the learning rate is near --lr_min (the last ~20%% of the schedule).")
    parser.add_argument("--lbfgs_steps", default=TrainConfig.lbfgs_steps, type=int,
                        help="Full-batch L-BFGS iterations after Adam, kept only if validation improves (0 = off).")
    # loss
    parser.add_argument("--lambda_delta", default=TrainConfig.lambda_delta, type=float,
                        help="Weight of the delta loss term (0 = off).")
    parser.add_argument("--lambda_shape", default=TrainConfig.lambda_shape, type=float,
                        help="Weight of the no-arbitrage shape penalty (0 = off).")
    parser.add_argument("--shape_points", default=TrainConfig.shape_points, type=int,
                        help="Fresh random domain points per step for the shape penalty.")
    parser.add_argument("--seed", default=TrainConfig.seed, type=int,
                        help="Seeds the split, the initialization, the batch order and the shape points "
                             "(test-ml must use the same).")

def train_config(args) -> TrainConfig:
    """The TrainConfig of the parsed flags (validated by TrainConfig: ValueError on a bad value)."""
    return TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})

def run(args):
    if not Path(args.dataset_path).is_file():
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}; generate it with "
                                "`python -m ML.surrogate.tree.generate_tree_dataset` or pass --dataset_path.")
    config = train_config(args)  # before any work: a bad flag fails here, not after loading the data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    start_time = time.time()
    result = train_surrogate(args.dataset_path, config, checkpoint_path=args.checkpoint_path, device=device)
    # the checkpoint holds the entry with the lowest validation loss (an L-BFGS finish only if it was kept)
    best = min(result.history, key=lambda h: h["valid_loss"])
    # the combined loss chose it; its two halves apart, since one can improve while the other worsens
    print(f"Best validation loss {best['valid_loss']:.4e} (epoch {best['epoch']}) = premium term "
          f"{best['valid_premium_term']:.4e} + delta term {best['valid_delta_term']:.4e}")
    print(f"  premium RMSE {best['valid_premium_rmse']:.3e}, delta RMSE {best['valid_delta_rmse']:.3e} per unit of strike")
    print(f"Model was trained in : {(time.time() - start_time):.2f}s")
    print(f"Model was saved as : {args.checkpoint_path}")
