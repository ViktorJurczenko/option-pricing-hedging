import time
import torch

from torch.utils.data import DataLoader, random_split

from datasets.american_option_dataset import AmericanOptionDataset
from AmericanOptionPricing.ML import MLP

def add_arguments(parser):
    parser.add_argument("--dataset_path", default="data/american_option_dataset.npz", type=str, help="Path to the dataset for MLP.")
    parser.add_argument("--split_ratio", default=0.8, type=float, help="Split size between test and train data.")
    parser.add_argument("--batch_size", default=32, type=int, help="Size of data batches for the MLP.")
    parser.add_argument("--epochs", default=100, type=int, help="Max training epochs (early stopping via --patience decides the actual stopping point).")
    parser.add_argument("--patience", default=10, type=int, help="Epochs with no validation improvement before early stopping.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument("--checkpoint_path", default="checkpoints/best_model.pth", type=str, help="Where to save the best model.")

def run(args):
    dataset_train = AmericanOptionDataset(args.dataset_path, is_train=True, split_ratio=args.split_ratio, seed=args.seed)

    mlp = MLP()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    mlp.to(device)

    n_valid = int(0.2 * len(dataset_train))
    n_train = len(dataset_train) - n_valid

    dataset_train, dataset_valid = random_split(
        dataset_train,
        [n_train, n_valid],
        generator=torch.Generator().manual_seed(args.seed)  # For reproducability
    )

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print("Training model...")
    start_time = time.time()

    mlp.train_model(
        optimizer, scheduler, args.patience, dataloader_train, dataloader_valid, device,
        epochs=args.epochs, checkpoint_path=args.checkpoint_path,
    )

    end_time = time.time()
    print(f"Model was trained in : {(end_time - start_time):.2f}s")
    print(f"Model was saved as : {args.checkpoint_path}")
