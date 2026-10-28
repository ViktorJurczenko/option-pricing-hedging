from pathlib import Path

from optionpricing.ml.surrogate.tree.tree_evaluation import (
    GRID_N_SIGMA,
    GRID_N_T,
    GRID_N_Z,
    GRID_RATES,
    REF_METHOD,
    REF_STEPS,
    VIOLATION_TOL,
    check_reference,
    evaluate_grid,
    evaluate_test_split,
    print_report,
    teacher_settings,
)
from optionpricing.ml.surrogate.tree.tree_pricer import PutSurrogate
from optionpricing.pricing.american.batch_binomial_tree import METHODS


def add_arguments(parser):
    parser.add_argument(
        "--dataset_path",
        default="artifacts/data/tree_put_dataset.npz",
        type=str,
        help="The tree put dataset the surrogate was trained on (its test split is scored).",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="artifacts/checkpoints/put_surrogate.pth",
        type=str,
        help="The trained PutSurrogate checkpoint.",
    )
    parser.add_argument("--split_ratio", default=0.8, type=float, help="Train/test split of the training run.")
    parser.add_argument("--seed", default=42, type=int, help="Seed of the training run's split.")
    parser.add_argument(
        "--ref_steps",
        default=REF_STEPS,
        type=int,
        help="Steps of the reference tree the grid is scored against (~48 ms per option at 4000; why 4000: tree_evaluation.REF_STEPS).",
    )
    parser.add_argument("--ref_method", default=REF_METHOD, choices=METHODS, help="Method of the reference tree.")
    parser.add_argument(
        "--n_z",
        default=GRID_N_Z,
        type=int,
        help="Grid points in z (evenly spaced over the domain; 27 = a step of 0.25).",
    )
    parser.add_argument("--n_sigma", default=GRID_N_SIGMA, type=int, help="Grid points in sigma (log-spaced).")
    parser.add_argument("--n_T", default=GRID_N_T, type=int, help="Grid points in T (log-spaced).")
    parser.add_argument("--rates", default=list(GRID_RATES), type=float, nargs="+", help="Grid rates.")
    parser.add_argument(
        "--violation_tol",
        default=[VIOLATION_TOL],
        type=float,
        nargs="+",
        help="A scaled no-arbitrage violation counts above this; several values give one table each (e.g. 1e-4 1e-5).",
    )
    parser.add_argument(
        "--reference_cache",
        default=None,
        type=str,
        help="A .npz file caching the grid's reference tree and vega (the same for every model, "
        "~2 min to compute): loaded if it exists, else computed and written. Refused if built "
        "for other grid/reference settings.",
    )


def run(args):
    # Checked up front: otherwise a missing file surfaces as a bare np.load/torch.load error
    for flag, path in [("--dataset_path", args.dataset_path), ("--checkpoint_path", args.checkpoint_path)]:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{flag}: no file at {path} (generate the dataset with "
                "`python -m optionpricing.ml.surrogate.tree.generate_tree_dataset`, then train the surrogate with "
                "`python main.py train-ml`)."
            )
    # before any work: a reference with the teacher's settings would score how well the net copies it
    teacher = teacher_settings(args.dataset_path)
    check_reference(teacher, args.ref_method, args.ref_steps)
    surrogate = PutSurrogate.load(args.checkpoint_path)
    print(f"Teacher: {teacher[0]}, {teacher[1]} steps; reference: {args.ref_method}, {args.ref_steps} steps")

    test = evaluate_test_split(surrogate, args.dataset_path, args.split_ratio, args.seed, args.violation_tol)
    grid = evaluate_grid(
        surrogate,
        args.ref_steps,
        args.ref_method,
        args.n_z,
        args.n_sigma,
        args.n_T,
        args.rates,
        args.violation_tol,
        args.reference_cache,
    )
    print_report(grid, test)
