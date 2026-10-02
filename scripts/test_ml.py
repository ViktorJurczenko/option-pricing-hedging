import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from core.gbm import GBM
from datasets.american_option_dataset import AmericanOptionDataset
from AmericanOptionPricing.ML import MLP
from AmericanOptionPricing.evaluation import regression_tests, test_option_price_lsm

def add_arguments(parser):
    parser.add_argument("--dataset_path", default="data/american_option_dataset.npz", type=str, help="Path to the dataset for MLP.")
    parser.add_argument("--features_norm_path", default="data/features_norm.npz", type=str, help="Path to the feature normalization stats.")
    parser.add_argument("--split_ratio", default=0.8, type=float, help="Split size between test and train data.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument("--checkpoint_path", default="checkpoints/best_model.pth", type=str, help="Path to the trained model's weights.")
    parser.add_argument("--max_degree", default=3, type=int, help="Must match the Longstaff-Schwartz config used to generate the training dataset's labels.")
    parser.add_argument("--reg_type", default="simple", type=str, choices=["simple", "laguerre"], help="Must match the Longstaff-Schwartz config used to generate the training dataset's labels.")

def run(args):
    dataset_test = AmericanOptionDataset(args.dataset_path, is_train=False, split_ratio=args.split_ratio, seed=args.seed)

    mlp = MLP()
    mlp.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))

    dataloader_test = DataLoader(dataset_test)

    print("Testing model...")
    regression_tests(mlp, dataloader_test)

    sigma_val = np.linspace(0.2, 0.4, 50)
    r_val     = np.linspace(0.01, 0.05, 50)
    T_val     = np.linspace(0.5, 1, 50)

    norms = np.load(args.features_norm_path)
    log_sigma_mean = norms["log_sigma_mean"]
    log_sigma_std  = norms["log_sigma_std"]
    r_mean         = norms["r_mean"]
    r_std          = norms["r_std"]
    log_T_mean     = norms["log_T_mean"]
    log_T_std      = norms["log_T_std"]

    # Normalize features with mean and std of dataset (Only for Z-scored features)
    sigma_val_norm  = (np.log(sigma_val) - log_sigma_mean) / log_sigma_std
    r_val_norm      = (r_val - r_mean) / r_std
    T_val_norm      = (np.log(T_val) - log_T_mean) / log_T_std

    # Sweeps the full range the model was trained on (moneyness_range=(0.4, 1.6)
    # in datasets/generate_dataset.py), staying inside its edges, so this actually
    # stress-tests deep ITM/OTM instead of the narrow band around ATM.
    moneyness = np.linspace(0.5, 1.5, 50)
    K_val     = np.linspace(90, 110, 50)

    for option_type in ("put", "call"):
        # Calls are benchmarked with the control variate, as the dataset labels them.
        is_call    = option_type == "call"
        errors     = np.zeros(50)
        abs_errors = np.zeros(50)

        for i in tqdm(range(50), desc=f"LSM benchmark ({option_type})", unit="sample"):
            # A large, fixed-seed path count keeps the LSM "ground truth" itself from
            # being a significant source of the reported error, especially near the
            # ITM exercise boundary where the regression is most sensitive to noise.
            # Both types are priced on the same paths.
            gbm = GBM(sigma_val[i], moneyness[i] * K_val[i], r_val[i], seed=args.seed + i)
            simulation = gbm.simulate_paths(20000, T_val[i], 52)

            errors[i], abs_errors[i] = test_option_price_lsm(
                mlp, simulation, moneyness[i], K_val[i], r_val[i], r_val_norm[i], T_val[i], T_val_norm[i], sigma_val_norm[i],
                option_type=option_type, reg_type=args.reg_type, max_degree=args.max_degree,
                sigma=sigma_val[i] if is_call else None, control_variate=is_call,
            )

        # moneyness = S/K: a put is in the money below 1, a call above it.
        itm = moneyness > 1.05 if is_call else moneyness < 0.95
        otm = moneyness < 0.95 if is_call else moneyness > 1.05
        atm = (moneyness >= 0.95) & (moneyness <= 1.05)

        print(f"\n{option_type.capitalize()}:")
        print(f"ITM avg error: {np.mean(errors[itm]):.2f}%")
        print(f"ATM avg error: {np.mean(errors[atm]):.2f}%")
        print(f"OTM avg error: {np.mean(errors[otm]):.2f}%")
        print(f"OTM avg abs error: {np.mean(abs_errors[otm]):.4f}")
