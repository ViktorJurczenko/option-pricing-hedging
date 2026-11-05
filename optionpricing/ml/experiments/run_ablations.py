"""
Step 10's seeded ablations (`.claude/PLAN.md`): each variant is the default
TrainConfig with one change (or, in COMBINED, the union of two such
variants' changes), trained once per seed on the tree dataset and
scored like `test-ml` (test split and grid, at several violation
tolerances). A dataset variant (DATASET_VARIANTS) trains another
variant's config on its own dataset, to test the data's design. Run
from the project root:

    python -m optionpricing.ml.experiments.run_ablations [--variants ...] [--seeds ...] [--jobs 4 --threads 2]
    # models trained on different T ranges, scored on the same T >= 0.25 grid
    python -m optionpricing.ml.experiments.run_ablations --variants lambda_delta_0.3_softplus T_from_0.25 --grid_T_min 0.25 --out_dir artifacts/runs/ablations_T0.25

Runs go to `<out_dir>/<variant>/seed_<seed>/`: the model, its training
history, the full text report, the training progress (log.txt) and
metrics.json, written last: a run with a metrics.json is done and is
skipped by a rerun (one whose stored settings differ is refused). The grid's
reference tree is computed once into `<out_dir>/reference_grid.npz`.
At the end, `<out_dir>/summary.md` holds mean ± standard deviation over
the seeds of each variant's key numbers, for every finished run in
`<out_dir>`, not only the ones this invocation asked for.
"""

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from optionpricing.ml.surrogate.tree.generate_tree_dataset import COLUMNS, domain_from
from optionpricing.ml.surrogate.tree.tree_training import TrainConfig
from optionpricing.pricing.american.batch_binomial_tree import METHODS

# name -> the TrainConfig fields it changes; the baseline changes none, so
# every other row differs from it in exactly one setting (PLAN.md, step 10)
VARIANTS = {
    "baseline": {},
    "lambda_delta_0": {"lambda_delta": 0.0},
    "lambda_delta_0.3": {"lambda_delta": 0.3},
    "lambda_delta_3": {"lambda_delta": 3.0},
    "softplus_output": {"output": "softplus"},
    "relu": {"activation": "relu"},
    "lbfgs_200": {"lbfgs_steps": 200},
    "shape_0.1": {"lambda_shape": 0.1},
    "width_256": {"width": 256},
}
# name -> the one-change variants it joins: the winners on different metrics
# (lambda_delta 0.3 on price, softplus on violations), to check that their
# gains add up (docs/results/surrogate-training.md); and that pair at width
# 256, to check whether a 4x larger network helps the chosen config
COMBINED = {
    "lambda_delta_0.3_softplus": ("lambda_delta_0.3", "softplus_output"),
    "lambda_delta_0.3_softplus_width_256": ("lambda_delta_0.3", "softplus_output", "width_256"),
}
VARIANTS |= {name: {k: v for part in parts for k, v in VARIANTS[part].items()} for name, parts in COMBINED.items()}
# name -> (the variant whose config it trains, the directory next to
# --dataset_path's that holds its tree_put_dataset.npz): the chosen config
# on a dataset made another way, each 2^19 rows so only the data differs
# (.claude/PLAN.md, dataset ablations). Generated with generate_tree_dataset's
# --T_min 0.25 and --exercised_keep 0.25.
DATASET_VARIANTS = {
    "T_from_0.25": ("lambda_delta_0.3_softplus", "T0.25"),
    "exercised_0.25": ("lambda_delta_0.3_softplus", "exercised_0.25"),
}
ALL_VARIANTS = (*VARIANTS, *DATASET_VARIANTS)
# three seeds: enough for a spread to compare variants against
SEEDS = (42, 43, 44)
# the default threshold and a 10x stricter one (the 2^19 net's error is ~1.4e-5 K)
TOLS = (1e-4, 1e-5)


def run_config(variant: str, seed: int, epochs=None) -> dict:
    """
    The variant's TrainConfig fields (a dataset variant's base variant's),
    as a dict, with `seed` (and `epochs` if given) set.
    """
    if variant not in ALL_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {list(ALL_VARIANTS)}")
    base = DATASET_VARIANTS[variant][0] if variant in DATASET_VARIANTS else variant
    overrides = {**VARIANTS[base], "seed": seed}
    if epochs is not None:
        overrides["epochs"] = epochs
    return asdict(TrainConfig(**overrides))  # validated here, before any run starts


def run_cost(config: dict) -> int:
    """
    Relative cost rank, so the slowest runs start first and none is left
    running alone at the end: the shape penalty triples a step, a network
    wider than TrainConfig's default costs ~2.5x per step at width 256
    (docs/results/surrogate-training.md, "Training cost by network width"),
    and the L-BFGS finish adds a tail.

    Args:
        config: a run's TrainConfig fields, as `run_config` returns them.

    Returns:
        The rank, 0 for the default config; higher runs start first.
    """
    wider = config["width"] > TrainConfig().width
    return 2 * (config["lambda_shape"] > 0) + 2 * wider + (config["lbfgs_steps"] > 0)


def run_dir(out_dir, variant: str, seed: int) -> Path:
    return Path(out_dir) / variant / f"seed_{seed}"


def dataset_path_for(variant: str, dataset_path) -> Path:
    """The dataset `variant` trains on: a dataset variant's own, next to `dataset_path`, else `dataset_path`."""
    if variant in DATASET_VARIANTS:
        return Path(dataset_path).parent / DATASET_VARIANTS[variant][1] / "tree_put_dataset.npz"
    return Path(dataset_path)


def dataset_domain(dataset_path) -> dict:
    """The tree dataset's sampling domain: {"<param>_range": (low, high)}."""
    with np.load(dataset_path) as data:
        return {k: tuple(float(v) for v in data[k]) for k in data.files if k.endswith("_range")}


def dataset_info(dataset_path) -> dict:
    """The dataset's path, generator settings and domain, recorded with each run so its data can be traced."""
    with np.load(dataset_path) as data:
        settings = {k: data[k].item() for k in data.files if k not in COLUMNS and not k.endswith("_range")}
    return {"path": str(dataset_path), **settings, **dataset_domain(dataset_path)}


def check_dataset(variant: str, dataset_path, grid: dict, ref_method: str, ref_steps: int) -> None:
    """
    Checks, before any run, that `variant`'s dataset exists, can be scored
    against the reference tree, and covers the grid every run is scored on.

    Raises:
        FileNotFoundError: if the dataset is missing (naming the command that makes it).
        ValueError: if the reference has the dataset's teacher settings
            (check_reference), or the grid leaves the dataset's domain: its
            points there would be routed, not priced by the net.
    """
    from optionpricing.ml.surrogate.tree.tree_evaluation import check_reference, teacher_settings

    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no dataset for {variant} at {path} "
            f"(python -m optionpricing.ml.surrogate.tree.generate_tree_dataset ... --output_dir {path.parent})"
        )
    check_reference(teacher_settings(path), ref_method, ref_steps)
    domain = dataset_domain(path)
    for key, (low, high) in grid.items():
        if low < domain[key][0] or high > domain[key][1]:
            raise ValueError(
                f"{variant}'s dataset {path} has {key} {domain[key]}, narrower than the grid's {(low, high)}: "
                "score it on a grid inside its domain (--grid_T_min)"
            )


def is_done(directory: Path, config: dict, evaluation: dict) -> bool:
    """
    True if the run in `directory` finished with these settings (its
    metrics.json exists), False if it has not finished.

    Raises:
        ValueError: if it finished with other settings: mixing them would
            compare runs that differ in more than the variant's one change.
    """
    path = directory / "metrics.json"
    if not path.is_file():
        return False
    stored = json.loads(path.read_text())
    if stored["config"] != config or stored["evaluation"] != evaluation:
        raise ValueError(f"{directory} holds a run with other settings; use another --out_dir or delete it")
    return True


def _json_default(x):
    """numpy scalars as Python numbers, for json.dump."""
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"not JSON serializable: {type(x)}")


def _write_json(path: Path, obj) -> None:
    """Writes `obj` atomically: an interrupted write leaves no file a rerun would trust."""
    tmp = path.with_name(path.stem + ".tmp.json")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    os.replace(tmp, path)


def _init_worker(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)


def train_and_score(job: dict) -> dict:
    """
    One run, in a worker process: trains, saves the model and history,
    scores it on the test split and the grid, writes the text report and,
    last, metrics.json. Progress bars and messages go to the run's log.txt.
    """
    from optionpricing.ml.surrogate.tree.tree_evaluation import evaluate_grid, evaluate_test_split, print_report
    from optionpricing.ml.surrogate.tree.tree_training import train_surrogate

    directory, config, ev = Path(job["dir"]), job["config"], job["evaluation"]
    directory.mkdir(parents=True, exist_ok=True)
    with (
        open(directory / "log.txt", "w", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        start = time.perf_counter()
        result = train_surrogate(job["dataset_path"], TrainConfig(**config), checkpoint_path=directory / "model.pth")
        train_seconds = time.perf_counter() - start
        _write_json(directory / "history.json", result.history)

        tols = ev["tols"]
        test = evaluate_test_split(result.surrogate, job["dataset_path"], config["split_ratio"], config["seed"], tols)
        grid = evaluate_grid(
            result.surrogate,
            ev["ref_steps"],
            ev["ref_method"],
            ev["n_z"],
            ev["n_sigma"],
            ev["n_T"],
            ev["rates"],
            tols,
            job["reference_cache"],
            domain=job["grid_domain"],
        )
    with open(directory / "report.txt", "w", encoding="utf-8") as report, contextlib.redirect_stdout(report):
        print_report(grid, test)

    # the entry the checkpoint holds: the lowest validation loss (an L-BFGS finish only if kept)
    best = min(result.history, key=lambda h: h["valid_loss"])
    metrics = {
        "variant": job["variant"],
        "seed": config["seed"],
        "config": config,
        "evaluation": ev,
        "dataset": dataset_info(job["dataset_path"]),
        "train_seconds": train_seconds,
        "best": best,
        "test": test,
        "grid": grid,
    }
    _write_json(directory / "metrics.json", metrics)
    return {"variant": job["variant"], "seed": config["seed"], "train_seconds": train_seconds}


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


def _row(rows: list, name: str) -> dict:
    return next(r for r in rows if r["name"] == name)


def _violations(result: dict, tol: float) -> dict:
    """The violation rates at `tol` (NaN if the run was scored at other tolerances)."""
    # json turned the float keys into strings
    by_tol = {float(k): v for k, v in result["violations"].items()}
    return by_tol.get(tol, {"continuation": {"any": math.nan}})


# (header, value of one run's metrics); figures in the report's units: $ at K = 100, bp of vol, %, 1/$
SUMMARY_COLUMNS = (
    ("grid \\|z\\|<1 MAE $", lambda m: 100 * _row(m["grid"]["by_z"], "|z| < 1")["price_mae"], ".4f"),
    ("grid all MAE $", lambda m: 100 * _row(m["grid"]["by_z"], "all")["price_mae"], ".4f"),
    ("grid max $", lambda m: 100 * _row(m["grid"]["by_z"], "all")["price_max"], ".4f"),
    ("IV err bp", lambda m: 1e4 * _row(m["grid"]["by_z"], "all")["iv_mae"], ".2f"),
    ("delta MAE \\|z\\|<1", lambda m: _row(m["grid"]["by_z"], "|z| < 1")["delta_mae"], ".2e"),
    ("gamma MAE 1/$", lambda m: _row(m["grid"]["by_z"], "all")["gamma_mae"] / 100, ".2e"),
    ("exercised max $", lambda m: 100 * _row(m["grid"]["by_z"], "exercised")["price_max"], ".4f"),
    ("test all MAE $", lambda m: 100 * _row(m["test"]["by_z"], "all")["price_mae"], ".4f"),
    *(
        (f"viol. > {tol:g} %", lambda m, tol=tol: 100 * _violations(m["grid"], tol)["continuation"]["any"], ".2f")
        for tol in TOLS
    ),
    ("valid premium term", lambda m: m["best"]["valid_premium_term"], ".2e"),
    ("valid delta term", lambda m: m["best"]["valid_delta_term"], ".2e"),
    ("train min", lambda m: m["train_seconds"] / 60, ".1f"),
)


def _mean_sd(values: list, spec: str) -> str:
    """'mean ± sd' (sample sd over the seeds), the mean alone for one value."""
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return "-"
    mean = statistics.fmean(values)
    if len(values) == 1:
        return format(mean, spec)
    return f"{format(mean, spec)} ± {format(statistics.stdev(values), spec)}"


def collect_results(out_dir) -> dict:
    """
    {variant: [metrics of each finished run]} for every variant with a
    finished run in `out_dir`, whatever `--variants` and `--seeds` the
    current invocation asked for: a rerun of a subset rewrites summary.md
    with every row, not only that subset's (BUGS.md, B-035). It does not
    compare settings: a run made with others (a smoke run's `--epochs`)
    left in `out_dir` is summarized with the rest, so give those their
    own `--out_dir`.
    """
    results = {}
    for variant in ALL_VARIANTS:
        paths = sorted((Path(out_dir) / variant).glob("seed_*/metrics.json"))
        if paths:
            results[variant] = [json.loads(p.read_text()) for p in paths]
    return results


def summarize(results: dict) -> str:
    """
    {variant: [metrics of each seed's run]} as a markdown table, one row
    per variant: mean ± standard deviation over its seeds.
    """
    header = ["variant", "seeds", *(h for h, _, _ in SUMMARY_COLUMNS)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for variant, runs in results.items():
        cells = [variant, str(len(runs))]
        cells += [_mean_sd([float(f(m)) for m in runs], spec) for _, f, spec in SUMMARY_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# SetThreadExecutionState flags (winbase.h): keep the setting until changed,
# keep the system running, keep the display on
ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002


def _set_execution_state(flags: int) -> None:
    """The Win32 call, apart so tests can replace it."""
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(flags)


@contextlib.contextmanager
def keep_awake():
    """
    Keeps Windows from suspending the run while inside, and releases it
    after (also on an error or Ctrl-C). On a Modern Standby PC (S0 low
    power idle) turning the screen off is standby, which "sleep: never"
    does not prevent: a first overnight run got ~30 min of compute in 9 h
    (BUGS.md). So the display is held on too: dim it rather than turn it
    off. Does nothing on other systems.
    """
    if sys.platform != "win32":
        yield
        return
    _set_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    try:
        yield
    finally:
        _set_execution_state(ES_CONTINUOUS)  # back to the normal power settings


def run_jobs(jobs: list, args, grid: dict) -> list:
    """
    Computes the shared reference on the grid built over the `grid` domain,
    then runs `jobs` in `args.jobs` worker processes. Returns the names of
    the runs that failed (each one's traceback is in its error.txt); the
    others are not stopped by them.
    """
    from optionpricing.ml.surrogate.tree.tree_evaluation import grid_reference

    failed = []
    # before the workers: one reference for every run
    grid_reference(
        grid,
        args.ref_steps,
        args.ref_method,
        args.n_z,
        args.n_sigma,
        args.n_T,
        args.rates,
        cache_path=Path(args.out_dir) / "reference_grid.npz",
    )

    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker, initargs=(args.threads,)) as pool:
        futures = {pool.submit(train_and_score, job): job for job in jobs}
        bar = tqdm(as_completed(futures), total=len(jobs), desc="Ablation runs", unit="run")
        for future in bar:
            job = futures[future]
            name = f"{job['variant']} seed {job['config']['seed']}"
            try:
                done = future.result()
                bar.set_postfix(last=name, minutes=f"{done['train_seconds'] / 60:.1f}")
            except Exception:
                # one failed run must not stop the night: logged next to it, listed at the end
                failed.append(name)
                (Path(job["dir"]) / "error.txt").write_text(traceback.format_exc())
                tqdm.write(f"FAILED: {name} (see {Path(job['dir']) / 'error.txt'})")
    return failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_parser() -> argparse.ArgumentParser:
    from optionpricing.ml.surrogate.tree.tree_evaluation import (
        GRID_N_SIGMA,
        GRID_N_T,
        GRID_N_Z,
        GRID_RATES,
        REF_METHOD,
        REF_STEPS,
    )

    parser = argparse.ArgumentParser(description="Seeded ablations of the tree surrogate (PLAN.md, step 10).")
    parser.add_argument("--dataset_path", default="artifacts/data/tree_put_dataset.npz", type=str)
    parser.add_argument(
        "--out_dir",
        default="artifacts/runs/ablations",
        type=str,
        help="Where the runs, the reference cache and summary.md go.",
    )
    parser.add_argument(
        "--variants",
        default=list(VARIANTS),
        nargs="+",
        choices=list(ALL_VARIANTS),
        help=(
            "Default: every config variant, including the slow width-256 ones (~60 min a run). "
            "Dataset variants (T_from_0.25, exercised_0.25) run only when named."
        ),
    )
    parser.add_argument("--seeds", default=list(SEEDS), nargs="+", type=int)
    parser.add_argument("--jobs", default=4, type=int, help="Runs at a time, one process each.")
    parser.add_argument(
        "--threads",
        default=2,
        type=int,
        help="Torch/BLAS threads per run (4 runs x 2 threads: ~2.8x one 8-thread run).",
    )
    parser.add_argument("--epochs", default=None, type=int, help="Override TrainConfig's epochs (smoke runs).")
    parser.add_argument("--ref_steps", default=REF_STEPS, type=int)
    parser.add_argument("--ref_method", default=REF_METHOD, choices=METHODS)
    parser.add_argument("--n_z", default=GRID_N_Z, type=int)
    parser.add_argument("--n_sigma", default=GRID_N_SIGMA, type=int)
    parser.add_argument("--n_T", default=GRID_N_T, type=int)
    parser.add_argument(
        "--grid_T_min",
        default=None,
        type=float,
        help="Score every run on a grid whose T starts here (0.25 for the T-range ablation); use its own --out_dir.",
    )
    parser.add_argument("--rates", default=list(GRID_RATES), nargs="+", type=float)
    parser.add_argument(
        "--tols",
        default=list(TOLS),
        nargs="+",
        type=float,
        help="Violation tolerances; the summary shows the default two.",
    )
    return parser


def evaluation_settings(args) -> dict:
    """
    The scoring settings every run of this invocation shares, stored in its
    metrics.json: a finished run with other ones is refused (is_done).
    """
    evaluation = {
        "ref_steps": args.ref_steps,
        "ref_method": args.ref_method,
        "n_z": args.n_z,
        "n_sigma": args.n_sigma,
        "n_T": args.n_T,
        "rates": args.rates,
        "tols": args.tols,
    }
    # only when set, so the runs scored before the flag existed still count as done
    if args.grid_T_min is not None:
        evaluation["grid_T_min"] = args.grid_T_min
    return evaluation


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    if not Path(args.dataset_path).is_file():
        raise FileNotFoundError(
            f"no dataset at {args.dataset_path} (python -m optionpricing.ml.surrogate.tree.generate_tree_dataset)"
        )
    # the threads of the worker processes' BLAS (the tree pricer's numpy); set before they start
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(args.threads)

    evaluation = evaluation_settings(args)
    grid = domain_from(args.grid_T_min, dataset_domain(args.dataset_path))
    # every config and dataset validated and every finished run checked before any work
    jobs, skipped = [], 0
    for variant in args.variants:
        dataset_path = dataset_path_for(variant, args.dataset_path)
        check_dataset(variant, dataset_path, grid, args.ref_method, args.ref_steps)
        for seed in args.seeds:
            config = run_config(variant, seed, args.epochs)
            directory = run_dir(args.out_dir, variant, seed)
            if is_done(directory, config, evaluation):
                skipped += 1
                continue
            jobs.append(
                {
                    "variant": variant,
                    "dir": str(directory),
                    "config": config,
                    "evaluation": evaluation,
                    "dataset_path": str(dataset_path),
                    "grid_domain": grid,
                    "reference_cache": str(Path(args.out_dir) / "reference_grid.npz"),
                }
            )
    # the slow runs first, so they don't end up alone at the end of the night
    jobs.sort(key=lambda j: -run_cost(j["config"]))
    print(f"{len(jobs)} runs to do, {skipped} already done; {args.jobs} at a time, {args.threads} threads each")

    failed = []
    if jobs:
        # held for the whole run: on a Modern Standby PC the screen turning off suspends it
        with keep_awake():
            failed = run_jobs(jobs, args, grid)

    table = summarize(collect_results(args.out_dir))
    (Path(args.out_dir) / "summary.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    if failed:
        print(f"\n{len(failed)} run(s) failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
