import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import qmc
from tqdm import tqdm

from optionpricing.ml.surrogate.chunks import chunk_file, n_chunks_for, prepare_chunk_dir, save_chunk, tqdm_joblib
from optionpricing.pricing.american.batch_binomial_tree import binomial_put_batch
from optionpricing.pricing.european.black_scholes import black_scholes_price

# The sampling domain, stored in the dataset so consumers know where the
# labels are valid (the `price` helper routes queries outside it).
DOMAIN = {
    "T_range": (1 / 52, 5.0),  # log-uniform, 1 week to 5 years
    "sigma_range": (0.05, 0.6),  # log-uniform
    "r_range": (0.0, 0.08),  # uniform
    # z = ln(m) / (sigma sqrt(T)): beyond +3 the early-exercise premium is
    # < 4e-5 K; below -3.5 (with r >= 0.005) the put is exercised today
    "z_range": (-3.5, 3.0),
    "m_range": (0.4, 2.5),  # moneyness cap, symmetric in log
}
# The parameters drawn from the Sobol points, one dimension each, in column
# order; m is derived from z. A new sampled parameter (e.g. dividends q)
# needs an entry here, its range in DOMAIN and its mapping in map_to_domain.
SAMPLED = ("T", "sigma", "r", "z")
COLUMNS = ("m", "T", "r", "sigma", "z", "price", "european_price", "delta", "gamma", "exercised")


def is_power_of_two(n: int) -> bool:
    return n >= 1 and n & (n - 1) == 0


def sobol_points(n: int, n_dims: int, seed: int) -> np.ndarray:
    """
    The first `n` points (a power of 2) of a scrambled `n_dims`-d Sobol
    sequence, shape (n, n_dims), in [0, 1). The Sobol points fill the cube far more evenly
    than independent draws (no clumps or holes), and that balance holds
    exactly for power-of-2 counts, hence the restriction. Point i depends
    only on the seed, not on n, so a run's first 2^k rows are a balanced
    subset too.
    """
    if not is_power_of_two(n):
        raise ValueError(f"the number of Sobol points must be a power of 2, got {n}")
    return qmc.Sobol(n_dims, scramble=True, seed=seed).random_base2(int(np.log2(n)))


def map_to_domain(u: np.ndarray, domain: dict = DOMAIN) -> dict:
    """
    Maps (n, len(SAMPLED)) uniforms to put parameters, one column per entry
    of SAMPLED: T and sigma log-uniform, r uniform, z by its position in
    its interval.

    z is drawn uniformly on that draw's allowed interval
    [max(z_low, ln m_low / vol), min(z_high, ln m_high / vol)],
    vol = sigma sqrt(T), so m = exp(z vol) stays in the moneyness cap. Rejecting draws with m outside
    the cap instead would reject long, volatile options more often and skew
    the T and sigma distributions.

    Returns:
        dict of (n,) arrays: m, T, r, sigma, z.

    Raises:
        ValueError: if `u` does not have one column per sampled parameter.
    """
    if u.ndim != 2 or u.shape[1] != len(SAMPLED):
        raise ValueError(f"expected uniforms of shape (n, {len(SAMPLED)}) for {SAMPLED}, got {u.shape}")
    col = dict(zip(SAMPLED, u.T))

    def log_uniform(x, low_high):
        low, high = np.log(low_high)
        return np.exp(low + x * (high - low))

    T = log_uniform(col["T"], domain["T_range"])
    sigma = log_uniform(col["sigma"], domain["sigma_range"])
    r_low, r_high = domain["r_range"]
    r = r_low + col["r"] * (r_high - r_low)
    vol = sigma * np.sqrt(T)
    # Each end takes the tighter of the z range and the z where m = exp(z vol)
    # hits the moneyness cap (ln(m_cap) / vol): the z range binds for short,
    # low-vol options, the cap for long, volatile ones.
    z_low = np.maximum(domain["z_range"][0], np.log(domain["m_range"][0]) / vol)
    z_high = np.minimum(domain["z_range"][1], np.log(domain["m_range"][1]) / vol)
    z = z_low + col["z"] * (z_high - z_low)
    return {"m": np.exp(z * vol), "T": T, "r": r, "sigma": sigma, "z": z}


def check_teacher(nb_steps: int, method: str, domain: dict = DOMAIN) -> None:
    """
    Checks, before any work starts, that the teacher (`binomial_put_batch`
    with these `nb_steps` and `method`) can price every put of the
    sampling domain; raises the pricer's ValueError if not.

    What can go wrong:
    - the settings themselves: an unknown method, too few steps, an odd
      `nb_steps` for "bbsr" (its coarse tree has nb_steps / 2 steps);
    - the settings combined with the domain: a tree step dt = T / nb_steps
      so long that the risk-neutral probability of an up move leaves
      (0, 1), which the pricer rejects rather than return a wrong price;
    - the domain itself, if DOMAIN is edited: a T or sigma bound <= 0.

    With u = exp(sigma sqrt(dt)) and d = 1 / u, the probability
    p = (exp(r dt) - d) / (u - d) reaches 1 exactly when r sqrt(dt) >= sigma,
    a condition that worsens steadily with T and r and as sigma falls (p
    itself is not monotone in T, but the breach condition is). Reaching 0
    would need r < 0, which the domain excludes. So the worst case is a
    corner of the domain (longest T, lowest sigma, highest r), and pricing
    the 8 corner puts (each T, sigma, r bound, at the money) checks the
    whole domain in milliseconds.

    Why up front: `main` writes the chunk directory's config.json before
    the workers start. If a bad setting only failed later, inside a worker,
    that config would stay behind and a re-run with corrected flags would
    be refused as "other settings".
    """
    T, sigma, r = np.meshgrid(domain["T_range"], domain["sigma_range"], domain["r_range"])
    binomial_put_batch(1.0, T, r, sigma, nb_steps=nb_steps, method=method)


def label_puts(params: dict, nb_steps: int, method: str) -> dict:
    """
    Prices American puts (per unit of strike) with the tree and adds their
    European Black-Scholes prices, whose difference is the early-exercise
    premium the MLP learns.

    Returns:
        `params` plus price, european_price, delta, gamma and exercised.
    """
    # Each tree level touches a (chunk, level + 1) array several times: at 32
    # options and 1000 steps that is ~256 KB, which stays in the CPU cache,
    # while a batch of 1000 (8 MB) goes to main memory on every pass. 32 was
    # the fastest of 16-1000 for "bbsr" at 1000 steps; smaller chunks pay more
    # Python overhead per level.
    tree = binomial_put_batch(
        params["m"], params["T"], params["r"], params["sigma"], nb_steps=nb_steps, method=method, chunk_size=32
    )
    european = black_scholes_price(params["m"], 1.0, params["T"], params["r"], params["sigma"], "put")
    return {
        **params,
        "price": tree.price,
        "european_price": european,
        "delta": tree.delta,
        "gamma": tree.gamma,
        "exercised": tree.exercised,
    }


def generate_chunk(path: Path, u: np.ndarray, nb_steps: int, method: str) -> None:
    """Labels one chunk of Sobol points and saves it to `path`."""
    save_chunk(path, **label_puts(map_to_domain(u), nb_steps, method))


def generate_tree_chunks(
    u: np.ndarray, chunk_dir: Path, chunk_size: int, nb_steps: int, method: str, n_jobs: int
) -> None:
    """
    Labels the points `u` in chunks of `chunk_size`, one joblib job per chunk,
    each saved to `chunk_dir` as it finishes; chunks already there are skipped,
    so an interrupted run resumes. The progress bar counts finished chunks
    (shown in rows) and starts at the ones already done.
    """
    n_chunks = n_chunks_for(len(u), chunk_size)
    todo = [i for i in range(n_chunks) if not chunk_file(chunk_dir, i).exists()]
    if len(todo) < n_chunks:
        print(f"Resuming: {n_chunks - len(todo)}/{n_chunks} chunks already done.")
    bar = tqdm(
        total=n_chunks, initial=n_chunks - len(todo), desc="Generating tree dataset", unit="row", unit_scale=chunk_size
    )
    with tqdm_joblib(bar):
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(generate_chunk)(
                chunk_file(chunk_dir, i), u[i * chunk_size : (i + 1) * chunk_size], nb_steps, method
            )
            for i in todo
        )


def load_tree_chunks(chunk_dir: Path, n_chunks: int) -> dict:
    """Concatenates the chunks' columns in chunk order."""
    parts = [np.load(chunk_file(chunk_dir, i)) for i in range(n_chunks)]
    return {col: np.concatenate([p[col] for p in parts]) for col in COLUMNS}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="American put dataset labelled by the binomial tree, for the MLP surrogate"
    )
    parser.add_argument(
        "--n_rows",
        default=2**19,
        type=int,
        help="Rows (puts) to generate; a power of 2, so the Sobol points stay balanced.",
    )
    parser.add_argument(
        "--chunk_size",
        default=1024,
        type=int,
        help="Rows per saved chunk, a power of 2. Finished chunks are kept in <output_dir>/tree_chunks, so an interrupted run resumes by re-running the same command.",
    )
    parser.add_argument("--nb_steps", default=1000, type=int, help="Tree steps per put (of the finer tree for 'bbsr').")
    parser.add_argument(
        "--method", default="bbsr", choices=["crr", "bbs", "bbsr"], help="Tree method (see binomial_put_batch)."
    )
    parser.add_argument("--nb_jobs", default=6, type=int, help="CPU cores used; each job labels one chunk.")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed of the Sobol scrambling; a resumed run must use the same one."
    )
    parser.add_argument("--output_dir", default="artifacts/data", type=str, help="Directory to write the dataset to.")
    args = parser.parse_args(argv)

    for name in ("n_rows", "chunk_size"):
        if not is_power_of_two(getattr(args, name)):
            raise ValueError(f"--{name} must be a power of 2, got {getattr(args, name)}")
    check_teacher(args.nb_steps, args.method)
    start_time = time.time()

    output_dir = Path(args.output_dir)
    chunk_dir = output_dir / "tree_chunks"
    settings = {
        "teacher": "tree",
        "nb_steps": args.nb_steps,
        "method": args.method,
        "seed": args.seed,
        "sampler": "sobol",
        "n_rows": args.n_rows,
    }
    # lists, not tuples: the stored config comes back from JSON with lists
    domain = {k: list(v) for k, v in DOMAIN.items()}
    prepare_chunk_dir(chunk_dir, {**settings, "chunk_size": args.chunk_size, **domain})

    u = sobol_points(args.n_rows, len(SAMPLED), args.seed)
    generate_tree_chunks(u, chunk_dir, args.chunk_size, args.nb_steps, args.method, args.nb_jobs)
    data = load_tree_chunks(chunk_dir, n_chunks_for(args.n_rows, args.chunk_size))

    path = output_dir / "tree_put_dataset.npz"
    # raw columns only: features and their normalization are built at load time
    np.savez(path, **data, **settings, **domain)
    shutil.rmtree(chunk_dir)
    print(
        f"Saved {args.n_rows} puts to '{path}': {data['exercised'].mean():.0%} exercised today, "
        f"{np.mean(np.abs(data['z']) < 1):.0%} within |z| < 1."
    )
    print(f"Total runtime: {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    main()
