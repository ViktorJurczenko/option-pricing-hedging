"""Generate a bank of hedging scenarios for Deep Hedging: everything about a path that training does not choose.

A bank holds simulated put contracts and their daily price paths (`simulate_scenarios`), and what follows from
them alone:
- the premium of each contract, the surrogate's price at `t_0`: the hedger sells the put at its own model's price;
- the early-exercise boundary of each contract on every day (`daily_boundary`), and the day `τ` each holder
  exercises (`exercise_days`, with the method chosen here);
- the surrogate's delta and gamma on every hedge date of every path, the pricing model the hedger reads.

Rebalancing frequency, costs, settlement, risk measure and network are chosen at train or test time, so one bank
serves many experiments (`docs/results/hedging-training.md`, "The pipeline"). The settings that produced a bank
are saved in it, so training and testing can refuse banks that do not belong together (`hedge_dataset.py`).

The work is split into blocks of contracts, one joblib job each. A block's results are saved as it finishes, so
an interrupted run resumes by re-running the same command. Most of the time goes to the surrogate's tree route:
its domain starts at `T` = 1 week, so the last 7 days of every path not exercised before them are priced by the
tree.

Run it as `python -m optionpricing.ml.hedging.generate_hedge_dataset --split train`, and once per split with the
same flags apart from `--split` and the sizes.
"""

import argparse
import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from optionpricing.hedging.exercise import METHODS, check_method, daily_boundary, exercise_days
from optionpricing.hedging.simulator import DAYS_PER_YEAR, ScenarioBank, simulate_scenarios
from optionpricing.ml.surrogate.chunks import chunk_file, n_chunks_for, prepare_chunk_dir, save_chunk, tqdm_joblib
from optionpricing.ml.surrogate.tree.tree_pricer import PutSurrogate

# Tag of the files this module writes, so the loader refuses anything else.
FORMAT = "hedge_dataset"
# Each split gets its own default seed, so train, validation and test banks never share paths by accident.
SPLIT_SEEDS = {"train": 1, "val": 2, "test": 3}
# The contract family, `docs/results/hedging-training.md`, "The defaults". 92 days is the first whole day past
# 3 months.
DEFAULT_RANGES = {
    "days_range": (92, 365),
    "sigma_range": (0.15, 0.30),
    "rate_range": (0.02, 0.06),
    "moneyness_range": (0.9, 1.1),
}
# The arrays a block job returns. Per contract: boundary, premium. Per scenario: tau, delta, gamma.
CHUNK_COLUMNS = ("boundary", "premium", "tau", "delta", "gamma")


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes, so a bank records exactly which surrogate checkpoint produced its Greeks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def surrogate_greeks(surrogate: PutSurrogate, bank: ScenarioBank, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The surrogate's delta and gamma of each scenario's put on every day before its `τ`, the hedge dates.

    Day `k` of a scenario is priced at `m = S_k / K` with `T - t_k` = its days left over `DAYS_PER_YEAR`. Since
    `τ` is at most the expiry, every priced day has at least one day left. `PutSurrogate.price` routes each query
    itself, so the days left outside its domain (the last week) go to the tree.

    Args:
        surrogate: the hedger's pricing model.
        bank: the scenarios.
        tau: exercise day of each scenario, as `exercise_days` returns it.

    Returns:
        `(delta, gamma)`, each of shape `bank.spot.shape`. Delta is in shares per option, gamma in shares per
        dollar (the surrogate's gamma per unit of strike, divided by the strike), the units of
        `baselines.baseline_inputs`. Days from `τ` on are NaN: the hedge never reads them.
    """
    delta = np.full(bank.spot.shape, np.nan)
    gamma = np.full(bank.spot.shape, np.nan)
    rows, days = np.nonzero(np.arange(bank.spot.shape[1]) < tau[:, None])
    if len(rows) == 0:
        return delta, gamma
    contract = bank.contract[rows]
    result = surrogate.price(
        bank.spot[rows, days] / bank.strike,
        (bank.days[contract] - days) / DAYS_PER_YEAR,
        bank.risk_free_rate[contract],
        bank.sigma[contract],
    )
    delta[rows, days] = result.delta
    gamma[rows, days] = result.gamma / bank.strike
    return delta, gamma


def label_contracts(
    bank: ScenarioBank, surrogate: PutSurrogate, method: str, shift: float | None, tree_steps: int
) -> dict[str, np.ndarray]:
    """Everything a bank stores beyond its paths, for the contracts and scenarios of `bank`.

    Args:
        bank: the scenarios, or a block of them (`ScenarioBank.select_contracts`).
        surrogate: the hedger's pricing model, for the Greeks.
        method, shift: the holder, as in `exercise_days`.
        tree_steps: steps of the BBSR trees that give the boundary.

    Returns:
        A dict with the `CHUNK_COLUMNS`: `boundary` (`daily_boundary`, in dollars) and `premium` (the surrogate's price
        at `t_0`, in dollars) per contract, and `tau`, `delta` and `gamma` (`surrogate_greeks`) per scenario.
    """
    # `tree` and `late` read the boundary. It is stored for every method, as the optional boundary input.
    boundary = daily_boundary(bank, nb_steps=tree_steps)
    tau = exercise_days(bank, method, boundary=boundary if method in ("tree", "late") else None, shift=shift)
    # The hedger prices with the surrogate, so it also sells at the surrogate's price. A pricing error then shows
    # in the P&L, as it would for a desk quoting from its model.
    premium = surrogate.price(bank.moneyness, bank.maturity, bank.risk_free_rate, bank.sigma).price * bank.strike
    delta, gamma = surrogate_greeks(surrogate, bank, tau)
    return {"boundary": boundary, "premium": premium, "tau": tau, "delta": delta, "gamma": gamma}


def generate_chunk(
    path: Path, bank: ScenarioBank, surrogate_path: str, method: str, shift: float | None, tree_steps: int
) -> None:
    """Label one block of contracts and save it to `path`. The surrogate is loaded here, in the worker."""
    save_chunk(path, **label_contracts(bank, PutSurrogate.load(surrogate_path), method, shift, tree_steps))


def generate_hedge_chunks(
    bank: ScenarioBank,
    chunk_dir: Path,
    chunk_size: int,
    surrogate_path: str,
    method: str,
    shift: float | None,
    tree_steps: int,
    n_jobs: int,
) -> None:
    """Label `bank` in blocks of `chunk_size` contracts, one joblib job each, saved to `chunk_dir` as they finish.

    Blocks already in `chunk_dir` are skipped, so an interrupted run resumes. The progress bar counts finished
    blocks, in contracts, and starts at the ones already done.
    """
    n_contracts = len(bank.days)
    n_chunks = n_chunks_for(n_contracts, chunk_size)
    todo = [i for i in range(n_chunks) if not chunk_file(chunk_dir, i).exists()]
    if len(todo) < n_chunks:
        print(f"Resuming: {n_chunks - len(todo)}/{n_chunks} chunks already done.")
    bar = tqdm(
        total=n_chunks,
        initial=n_chunks - len(todo),
        desc="Generating hedge dataset",
        unit="contract",
        unit_scale=chunk_size,
    )
    with tqdm_joblib(bar):
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(generate_chunk)(
                chunk_file(chunk_dir, i),
                bank.select_contracts(i * chunk_size, min((i + 1) * chunk_size, n_contracts)),
                surrogate_path,
                method,
                shift,
                tree_steps,
            )
            for i in todo
        )


def load_hedge_chunks(chunk_dir: Path, n_chunks: int) -> dict[str, np.ndarray]:
    """Concatenate the blocks' arrays in block order, which is contract order and so scenario order."""
    parts = [np.load(chunk_file(chunk_dir, i)) for i in range(n_chunks)]
    return {col: np.concatenate([p[col] for p in parts]) for col in CHUNK_COLUMNS}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Bank of hedging scenarios for Deep Hedging: paths, contracts, premiums, surrogate Greeks, "
        "the exercise boundary and the exercise days"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=list(SPLIT_SEEDS),
        help="Which bank this is. Sets the file name and the default seed.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help=f"Seed of the contracts and paths (default per split: {SPLIT_SEEDS}). Banks meant to be used "
        "together must have distinct seeds.",
    )
    parser.add_argument("--n_contracts", default=50_000, type=int, help="Contracts drawn.")
    parser.add_argument(
        "--paths_per_contract",
        default=1,
        type=int,
        help="Paths simulated per contract. 1 draws a contract per path.",
    )
    parser.add_argument(
        "--days_range",
        default=DEFAULT_RANGES["days_range"],
        type=int,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Shortest and longest days to expiry.",
    )
    for name, help_text in (
        ("sigma_range", "Lowest and highest volatility."),
        ("rate_range", "Lowest and highest risk-free rate."),
        ("moneyness_range", "Lowest and highest S_0 / K."),
    ):
        parser.add_argument(
            f"--{name}", default=DEFAULT_RANGES[name], type=float, nargs=2, metavar=("LOW", "HIGH"), help=help_text
        )
    parser.add_argument("--strike", default=100.0, type=float, help="Strike of every contract, in dollars.")
    parser.add_argument("--exercise", default="tree", choices=list(METHODS), help="How the holder chooses tau.")
    parser.add_argument(
        "--shift",
        default=None,
        type=float,
        help="For --exercise late only: the fraction in (0, 1) the boundary is lowered by.",
    )
    parser.add_argument(
        "--tree_steps",
        default=1000,
        type=int,
        help="Steps of the BBSR trees that give the exercise boundary.",
    )
    parser.add_argument(
        "--surrogate",
        default="artifacts/checkpoints/put_surrogate.pth",
        type=str,
        help="PutSurrogate checkpoint whose delta and gamma the hedger reads.",
    )
    parser.add_argument(
        "--chunk_size",
        default=512,
        type=int,
        help="Contracts per saved block. Finished blocks are kept in <output_dir>/hedge_chunks_<split>, so an "
        "interrupted run resumes by re-running the same command.",
    )
    parser.add_argument("--nb_jobs", default=6, type=int, help="CPU cores used; each job labels one block.")
    parser.add_argument("--output_dir", default="artifacts/data", type=str, help="Directory to write the bank to.")
    args = parser.parse_args(argv)

    # Everything that can be refused is checked before the chunk directory's config is written, or the corrected
    # re-run would be refused as "other settings".
    if args.chunk_size < 1:
        raise ValueError(f"--chunk_size must be >= 1, got {args.chunk_size}")
    if args.tree_steps < 2 or args.tree_steps % 2:
        raise ValueError(f"--tree_steps must be an even number >= 2 (BBSR halves the tree), got {args.tree_steps}")
    check_method(args.exercise, args.shift)
    PutSurrogate.load(args.surrogate)  # fails now on a missing or foreign checkpoint
    seed = SPLIT_SEEDS[args.split] if args.seed is None else args.seed
    ranges = {name: tuple(getattr(args, name)) for name in DEFAULT_RANGES}
    start_time = time.time()
    bank = simulate_scenarios(args.n_contracts, args.paths_per_contract, seed, strike=args.strike, **ranges)

    settings = {
        "format": FORMAT,
        "split": args.split,
        "seed": seed,
        "n_contracts": args.n_contracts,
        "paths_per_contract": args.paths_per_contract,
        "strike": float(args.strike),
        "days_per_year": DAYS_PER_YEAR,
        "exercise": args.exercise,
        # NaN, not None: np.savez stores None as an object array, which np.load refuses without allow_pickle
        "shift": float("nan") if args.shift is None else float(args.shift),
        "tree_steps": args.tree_steps,
        "surrogate_path": args.surrogate,
        "surrogate_sha256": file_sha256(args.surrogate),
    }
    # lists, not tuples: the stored config comes back from JSON with lists
    ranges = {k: list(v) for k, v in ranges.items()}
    output_dir = Path(args.output_dir)
    chunk_dir = output_dir / f"hedge_chunks_{args.split}"
    # JSON has no NaN in its standard, so the config stores the shift as given
    prepare_chunk_dir(chunk_dir, {**settings, "shift": args.shift, "chunk_size": args.chunk_size, **ranges})

    generate_hedge_chunks(
        bank, chunk_dir, args.chunk_size, args.surrogate, args.exercise, args.shift, args.tree_steps, args.nb_jobs
    )
    labels = load_hedge_chunks(chunk_dir, n_chunks_for(args.n_contracts, args.chunk_size))

    path = output_dir / f"hedge_{args.split}.npz"
    # float32 for the (scenario, day) arrays, which are most of the file: a cent in $100 is 1e-4, far above
    # float32's 6e-8 relative precision
    np.savez(
        path,
        spot=bank.spot.astype(np.float32),
        delta=labels["delta"].astype(np.float32),
        gamma=labels["gamma"].astype(np.float32),
        boundary=labels["boundary"].astype(np.float32),
        tau=labels["tau"].astype(np.int32),
        contract=bank.contract.astype(np.int32),
        days=bank.days.astype(np.int32),
        sigma=bank.sigma,
        risk_free_rate=bank.risk_free_rate,
        moneyness=bank.moneyness,
        premium=labels["premium"],
        **settings,
        **ranges,
    )
    shutil.rmtree(chunk_dir)
    tau, expiry = labels["tau"], bank.days[bank.contract]
    print(
        f"Saved {len(tau)} scenarios to '{path}': {np.mean(tau < expiry):.0%} exercised early, "
        f"{np.mean(tau == 0):.1%} at once."
    )
    print(f"Total runtime: {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    main()
