"""
The legacy LSM dataset: Longstaff-Schwartz-labelled calls and puts for
lsm_model.LegacyMLP, written to `<output_dir>/american_option_dataset.npz`
with `features_norm.npz`. Run as a module from the project root:
`python -m optionpricing.ml.surrogate.legacy.generate_lsm_dataset`. The tree-trained
surrogate uses `python -m optionpricing.ml.surrogate.tree.generate_tree_dataset` instead.
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from optionpricing.core.gbm import GBM
from optionpricing.ml.surrogate.chunks import chunk_file, n_chunks_for, prepare_chunk_dir, save_chunk, tqdm_joblib
from optionpricing.ml.surrogate.legacy.lsm_model import option_type_feature, smooth_intrinsic_value
from optionpricing.pricing.american.longstaff_schwartz import LongstaffSchwartz


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
    r_val = np.random.uniform(r_range[0], r_range[1], n_samples)

    # Log-uniform for sigma and T ensures proportional coverage of small values
    sigma_val = np.exp(np.random.uniform(np.log(sigma_range[0]), np.log(sigma_range[1]), n_samples))
    T_val = np.exp(np.random.uniform(np.log(T_range[0]), np.log(T_range[1]), n_samples))

    S0_repeated = np.repeat(S0_val, samples_per_param)
    sigma_repeated = np.repeat(sigma_val, samples_per_param)
    r_repeated = np.repeat(r_val, samples_per_param)
    T_repeated = np.repeat(T_val, samples_per_param)

    # Log-uniform (moneyness is naturally multiplicative) and continuous across
    # the whole range, instead of clustered around a few discrete points.
    moneyness = np.exp(np.random.uniform(np.log(moneyness_range[0]), np.log(moneyness_range[1]), len(S0_repeated)))
    K_val = S0_repeated / moneyness

    return S0_repeated, K_val, sigma_repeated, r_repeated, T_repeated


def gen_data(simulation, S0, K, r, T, sigma, option_type, max_degree, reg_type) -> float:
    """
    Runs LSM and returns the time value (price minus a *smooth* approximation of
    intrinsic value - see optionpricing.ml.surrogate.legacy.lsm_model.smooth_intrinsic_value),
    scaled by strike. Subtracting the exact intrinsic value would stamp its hard
    kink at S == K onto this residual even though the pre-expiry price has no
    kink there, which empirically showed up as the MLP's error spiking right at
    the money; the smooth version keeps the target smooth everywhere while still
    shrinking its dynamic range (deep ITM prices are much larger than OTM ones).

    Calls use the European option as a control variate, which makes the
    label the exact Black-Scholes price (an American call without dividends
    is never exercised early); plain LSM exercises calls spuriously near
    expiry (see BUGS.md). Puts use plain LSM, which was worth more out of
    sample than the control-variate rule.
    """
    is_call = option_type == "call"
    lsm = LongstaffSchwartz(
        simulation,
        K,
        r,
        T,
        option_type,
        reg_type,
        max_degree,
        sigma=sigma if is_call else None,
        control_variate=is_call,
        compute_greeks=False,
    )
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
    option_types: tuple = ("call", "put"),
    max_degree: int = 3,
    reg_type: str = "simple",
    seed: int | None = None,
) -> list:
    """
    Process a single sample: simulate paths once and label them with LSM for
    each option type.

    Args:
        S0: Initial stock price.
        K: Strike price.
        sigma: Volatility.
        r: Risk-free rate.
        T: Time to maturity (in years).
        n_paths: Number of paths to simulate.
        option_types: Option types to label, each giving one row ("call", "put").
        max_degree: Maximum polynomial degree for LSM regression.
        reg_type: Basis used for the LSM regression ("simple" or "laguerre").
        seed: Seed of this sample's path simulation. `main` gives each sample its
            own, so a sample (hence a chunk redone on resume) is reproducible.

    Returns:
        List of (features, label) rows, one per option type, with features
        [S0/K, T_norm, r_norm, sigma_norm, is_call]. Empty if the sample failed.
    """
    try:
        gbm = GBM(sigma, S0, r, seed=seed)
        simulation = gbm.simulate_paths(n_paths, T, sampling)

        sigma_norm = (np.log(sigma) - log_sigma_mean) / log_sigma_std
        r_norm = (r - r_mean) / r_std
        T_norm = (np.log(T) - log_T_mean) / log_T_std

        rows = []
        for option_type in option_types:
            features = [S0 / K, T_norm, r_norm, sigma_norm, option_type_feature(option_type)]
            label = gen_data(simulation, S0, K, r, T, sigma, option_type, max_degree, reg_type)
            rows.append((features, label))

        return rows

    except Exception as e:
        print(f"Skipping sample (S0={S0}, K={K}, sigma={sigma}, r={r}, T={T}): {e}")
        return []


def generate_chunks(args_list: list, chunk_dir: Path, chunk_size: int, n_jobs: int) -> None:
    """
    Runs `process_sample` over `args_list` in chunks of `chunk_size`, saving each
    finished chunk to `chunk_dir` and skipping the chunks already there, so an
    interrupted run resumes instead of restarting. The progress bar starts at the
    samples already done and is driven by completed jobs (see tqdm_joblib).
    """
    n_total = len(args_list)
    todo = [i for i in range(n_chunks_for(n_total, chunk_size)) if not chunk_file(chunk_dir, i).exists()]
    done = n_total - sum(len(args_list[i * chunk_size : (i + 1) * chunk_size]) for i in todo)
    if done:
        print(f"Resuming: {done}/{n_total} samples already done.")

    with tqdm_joblib(tqdm(total=n_total, initial=done, desc="Generating dataset", unit="sample")):
        with Parallel(n_jobs=n_jobs, backend="loky", batch_size=10) as parallel:
            for i in todo:
                results = parallel(
                    delayed(process_sample)(*a) for a in args_list[i * chunk_size : (i + 1) * chunk_size]
                )
                # failed samples returned no rows
                rows = [row for sample_rows in results for row in sample_rows]  # type: ignore
                features = np.array([feat for feat, _ in rows], dtype=float).reshape(-1, 5)
                labels = np.array([lab for _, lab in rows], dtype=float)
                save_chunk(chunk_file(chunk_dir, i), features=features, label=labels)


def load_chunks(chunk_dir: Path, n_chunks: int) -> tuple:
    """Concatenates the chunks in order into (features, labels)."""
    parts = [np.load(chunk_file(chunk_dir, i)) for i in range(n_chunks)]
    return np.concatenate([p["features"] for p in parts]), np.concatenate([p["label"] for p in parts])


def main():
    parser = argparse.ArgumentParser(description="American Option pricing data set generation")

    parser.add_argument(
        "--path_per_sim",
        default=10000,
        type=int,
        help="The number of path generated by the GBM with a given set of parameters.",
    )
    parser.add_argument(
        "--nb_param_comb",
        default=20000,
        type=int,
        help="The number of parameter combination each yielding a prediction.",
    )
    parser.add_argument(
        "--samples_per_param",
        default=5,
        type=int,
        help="Number of distinct strikes (moneyness draws) generated per parameter combination.",
    )
    parser.add_argument("--nb_jobs", default=6, type=int, help="The number of CPU cores used to generate the data set.")
    parser.add_argument("--sampling", default=52, type=int, help="Number of time steps per year (e.g., 365 for daily).")
    parser.add_argument(
        "--option_type",
        default="both",
        type=str,
        choices=["call", "put", "both"],
        help="Option type(s) to label: 'both' prices a call and a put on each simulation, one row each.",
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
    parser.add_argument(
        "--output_dir",
        default="artifacts/data",
        type=str,
        help="Directory to write the generated dataset and normalization stats to.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed of the parameter draw and, offset by the sample index, of each sample's path simulation; a resumed run must use the same one.",
    )
    parser.add_argument(
        "--chunk_size",
        default=1000,
        type=int,
        help="Samples per saved chunk. Finished chunks are kept in <output_dir>/chunks, so an interrupted run (Ctrl-C) resumes by re-running the same command; at most one chunk of work is lost.",
    )

    args = parser.parse_args()

    print("Generating data set...")
    start_time = time.time()

    moneyness_range = (0.4, 1.6)

    # Seeded so a resumed run draws the same parameters (and normalization
    # stats) as the interrupted one.
    np.random.seed(args.seed)

    # Generate random parameters
    S0_val, K_val, sigma_val, r_val, T_val = gen_rdm_params(
        args.nb_param_comb, (50, 150), moneyness_range, args.samples_per_param, (0.05, 0.6), (0.0, 0.08), (0.25, 5)
    )

    # Normalize sigma and r values. The other parameters a normalized along data generation (see function gen_data)
    log_sigma_mean = np.mean(np.log(sigma_val))
    log_T_mean = np.mean(np.log(T_val))
    r_mean = np.mean(r_val)
    log_sigma_std = np.std(np.log(sigma_val))
    log_T_std = np.std(np.log(T_val))
    r_std = np.std(r_val)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "features_norm.npz",
        log_sigma_mean=log_sigma_mean,
        log_sigma_std=log_sigma_std,
        r_mean=r_mean,
        r_std=r_std,
        log_T_mean=log_T_mean,
        log_T_std=log_T_std,
    )

    n_total = args.nb_param_comb * args.samples_per_param
    option_types = ("call", "put") if args.option_type == "both" else (args.option_type,)

    # Prepare arguments for parallel processing
    args_list = [
        (
            S0_val[i],
            K_val[i],
            sigma_val[i],
            log_sigma_mean,
            log_sigma_std,
            r_val[i],
            r_mean,
            r_std,
            T_val[i],
            log_T_mean,
            log_T_std,
            args.path_per_sim,
            args.sampling,
            option_types,
            args.max_degree,
            args.reg_type,
            args.seed + i,
        )  # one seed per sample, so a redone chunk reproduces its labels
        for i in range(n_total)
    ]

    # Chunked, resumable parallel processing: finished chunks are kept on disk
    # and skipped when the same command is re-run after an interruption.
    chunk_dir = output_dir / "chunks"
    prepare_chunk_dir(
        chunk_dir,
        {
            "seed": args.seed,
            "nb_param_comb": args.nb_param_comb,
            "samples_per_param": args.samples_per_param,
            "path_per_sim": args.path_per_sim,
            "sampling": args.sampling,
            "option_type": args.option_type,
            "reg_type": args.reg_type,
            "max_degree": args.max_degree,
            "chunk_size": args.chunk_size,
        },
    )
    generate_chunks(args_list, chunk_dir, args.chunk_size, args.nb_jobs)
    features, labels = load_chunks(chunk_dir, n_chunks_for(n_total, args.chunk_size))

    print("Dataset generated successfully!")
    print(f"Features shape: {features.shape}")
    print(f"Time value (price - intrinsic, scaled by K) shape: {labels.shape}")

    np.savez(
        output_dir / "american_option_dataset.npz",
        features=features,
        label=labels,
        # The teacher's settings travel with the labels, so an evaluation
        # (lsm_evaluation.test_option_price_lsm) can benchmark against the
        # same LSM instead of its own, possibly drifted, settings.
        reg_type=args.reg_type,
        max_degree=args.max_degree,
        path_per_sim=args.path_per_sim,
        sampling=args.sampling,
    )
    print(f"Dataset saved to '{output_dir / 'american_option_dataset.npz'}'.")
    shutil.rmtree(chunk_dir)

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    main()
