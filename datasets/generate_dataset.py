import numpy as np
from pathlib import Path
from contextlib import contextmanager
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import argparse

from AmericanOptionPricing.LongstaffSchwartz import LongstaffSchwartz
from AmericanOptionPricing.ML import smooth_intrinsic_value
from core.gbm import GBM

@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager that makes a tqdm bar advance on each completed joblib
    batch, instead of on each submitted argument (the default `tqdm(args_list)`
    wrapping reports submission order, not completion, so it doesn't reflect
    actual progress/ETA for parallel work).
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

def gen_rdm_params(
    n_samples: int,
    S0_range: tuple = (80, 120),
    moneyness_range: tuple = (0.5, 1.5),
    samples_per_param: int = 5,
    sigma_range: tuple = (0.05, 0.6),
    r_range: tuple = (0.0, 0.08),
    T_range: tuple = (0.25, 3),
) -> tuple:
    """
    Generate random parameter combinations for dataset generation.

    Args:
        n_samples: Number of unique (S0, sigma, r, T) combinations to generate.
        S0_range: Range for initial stock price (S0).
        moneyness_range: Range for moneyness (S0/K), sampled log-uniformly so
            training data covers it continuously. (Sampling from a handful of
            discrete moneyness values instead, e.g. {0.5, 0.7, 1.0, 1.2, 1.5},
            leaves large untrained gaps between them that the model then has to
            extrapolate across.)
        samples_per_param: Number of distinct strikes (moneyness draws) generated
            per (S0, sigma, r, T) combination.
        sigma_range: Range for volatility (sigma).
        r_range: Range for risk-free rate (r).
        T_range: Range for time to maturity (T).

    Returns:
        Tuple of arrays: (S0_val, K_val, sigma_val, r_val, T_val)
    """

    S0_val = np.random.uniform(S0_range[0], S0_range[1], n_samples)
    r_val  = np.random.uniform(r_range[0], r_range[1], n_samples)

    # Log-uniform for sigma and T ensures proportional coverage of small values
    sigma_val = np.exp(np.random.uniform(np.log(sigma_range[0]), np.log(sigma_range[1]), n_samples))
    T_val     = np.exp(np.random.uniform(np.log(T_range[0]), np.log(T_range[1]), n_samples))

    S0_repeated    = np.repeat(S0_val, samples_per_param)
    sigma_repeated = np.repeat(sigma_val, samples_per_param)
    r_repeated     = np.repeat(r_val, samples_per_param)
    T_repeated     = np.repeat(T_val, samples_per_param)

    # Log-uniform (moneyness is naturally multiplicative) and continuous across
    # the whole range, instead of clustered around a few discrete points.
    moneyness = np.exp(np.random.uniform(
        np.log(moneyness_range[0]), np.log(moneyness_range[1]), len(S0_repeated)
    ))
    K_val = S0_repeated / moneyness

    return  S0_repeated, K_val, sigma_repeated, r_repeated, T_repeated

def gen_data(simulation, S0, K, r, T, option_type, max_degree, reg_type) -> float:
    """
    Runs LSM and returns the time value (price minus a *smooth* approximation of
    intrinsic value - see AmericanOptionPricing.ML.smooth_intrinsic_value),
    scaled by strike. Subtracting the exact intrinsic value would stamp its hard
    kink at S == K onto this residual even though the pre-expiry price has no
    kink there, which empirically showed up as the MLP's error spiking right at
    the money; the smooth version keeps the target smooth everywhere while still
    shrinking its dynamic range (deep ITM prices are much larger than OTM ones).
    """
    lsm = LongstaffSchwartz(simulation, K, r, T, option_type, reg_type, max_degree)
    lsm.predict_payoff()
    price = lsm.calculate_option_price()
    return (price - smooth_intrinsic_value(S0, K, option_type)) / K

def process_sample(
    S0: float,
    K: float,
    sigma: float,
    log_sigma_mean: float,
    log_sigma_std: float,
    r: float,
    r_mean,
    r_std,
    T: float,
    log_T_mean: float,
    log_T_std: float,
    n_paths: int,
    sampling: int,
    option_type: str = "put",
    max_degree: int = 3,
    reg_type: str = "simple"
) -> tuple:
    """
    Process a single sample: simulate paths and compute LSM.

    Args:
        S0: Initial stock price.
        K: Strike price.
        sigma: Volatility.
        r: Risk-free rate.
        T: Time to maturity (in years).
        n_paths: Number of paths to simulate.
        option_type: Type of option ("call" or "put").
        max_degree: Maximum polynomial degree for LSM regression.
        reg_type: Basis used for the LSM regression ("simple" or "laguerre").

    Returns:
        Tuple of:
        - features
        - label
    """
    try:
        gbm        = GBM(sigma, S0, r)
        simulation = gbm.simulate_paths(n_paths, T, sampling)

        sigma_norm = (np.log(sigma) - log_sigma_mean) / log_sigma_std
        r_norm     = (r - r_mean) / r_std
        T_norm     = (np.log(T) - log_T_mean) / log_T_std

        features = [S0 / K, T_norm, r_norm, sigma_norm]
        label    = gen_data(simulation, S0, K, r, T, option_type, max_degree, reg_type)

        return features, label

    except Exception as e:
        print(f"Skipping sample (S0={S0}, K={K}, sigma={sigma}, r={r}, T={T}): {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="American Option pricing data set generation")

    parser.add_argument("--path_per_sim", default=10000, type=int, help="The number of path generated by the GBM with a given set of parameters.")
    parser.add_argument("--nb_param_comb", default=20000, type=int, help="The number of parameter combination each yielding a prediction.")
    parser.add_argument("--samples_per_param", default=5, type=int, help="Number of distinct strikes (moneyness draws) generated per parameter combination.")
    parser.add_argument("--nb_jobs", default=6, type=int, help="The number of CPU cores used to generate the data set.")
    parser.add_argument("--sampling", default=52, type=int, help="Number of time steps per year (e.g., 365 for daily).")
    parser.add_argument("--option_type", default="put", type=str, choices=["call", "put"], help="Type of option ('call' or 'put').")
    parser.add_argument("--max_degree", default=3, type=int, help="Maximum degree of the polynomial used in Longstaff-Schwartz regression.")
    parser.add_argument("--reg_type", default="simple", type=str, choices=["simple", "laguerre"], help="polynomials used for regression.")
    parser.add_argument("--output_dir", default="data", type=str, help="Directory to write the generated dataset and normalization stats to.")

    args = parser.parse_args()

    print("Generating data set...")
    start_time = time.time()

    moneyness_range = (0.4, 1.6)

    # Generate random parameters
    S0_val, K_val, sigma_val, r_val, T_val = gen_rdm_params(
        args.nb_param_comb, (50, 150), moneyness_range, args.samples_per_param, (0.05, 0.6), (0.0, 0.08), (0.25, 5)
    )

    # Normalize sigma and r values. The other parameters a normalized along data generation (see function gen_data)
    log_sigma_mean = np.mean(np.log(sigma_val))
    log_T_mean     = np.mean(np.log(T_val))
    r_mean         = np.mean(r_val)
    log_sigma_std  = np.std(np.log(sigma_val))
    log_T_std      = np.std(np.log(T_val))
    r_std          = np.std(r_val)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "features_norm.npz",
        log_sigma_mean = log_sigma_mean,
        log_sigma_std  = log_sigma_std,
        r_mean         = r_mean,
        r_std          = r_std,
        log_T_mean     = log_T_mean,
        log_T_std      = log_T_std,
    )

    n_total = args.nb_param_comb * args.samples_per_param

    # Prepare arguments for parallel processing
    args_list = [
        (S0_val[i], K_val[i], sigma_val[i], log_sigma_mean, log_sigma_std, r_val[i], r_mean, r_std,
         T_val[i], log_T_mean, log_T_std, args.path_per_sim, args.sampling, args.option_type, args.max_degree, args.reg_type)
        for i in range(n_total)
    ]

    # Parallel processing, with a progress bar driven by completed jobs (not
    # submitted args) so the ETA reflects actual remaining work.
    with tqdm_joblib(tqdm(total=n_total, desc="Generating dataset", unit="sample")):
        results = Parallel(n_jobs=args.nb_jobs, backend='loky', batch_size=10)(
            delayed(process_sample)(*a) for a in args_list)

    # Combine results
    features = np.zeros((n_total, 4))
    labels   = np.zeros(n_total)

    # Track which rows were actually filled
    filled = np.zeros(n_total, dtype=bool)
    for i, (feat, lab) in enumerate(results): # type: ignore
        if feat is None or lab is None:
            continue
        features[i] = feat
        labels[i]   = lab
        filled[i]   = True

    features = features[filled]
    labels   = labels[filled]

    print("Dataset generated successfully!")
    print(f"Features shape: {features.shape}")
    print(f"Time value (price - intrinsic, scaled by K) shape: {labels.shape}")

    np.savez(
        output_dir / "american_option_dataset.npz",
        features=features,
        label=labels
    )
    print(f"Dataset saved to '{output_dir / 'american_option_dataset.npz'}'.")

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
