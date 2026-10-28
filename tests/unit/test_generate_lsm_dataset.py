import numpy as np
import pytest

from optionpricing.ml.surrogate.legacy.lsm_model import smooth_intrinsic_value
from optionpricing.core.gbm import GBM
from optionpricing.ml.surrogate.chunks import chunk_file
from optionpricing.ml.surrogate.legacy.generate_lsm_dataset import (gen_data, gen_rdm_params, generate_chunks, load_chunks,
                                                      process_sample)
from optionpricing.pricing.european.black_scholes import BlackScholes

NORM = dict(log_sigma_mean=np.log(0.2), log_sigma_std=0.5, r_mean=0.03, r_std=0.02,
            log_T_mean=0.0, log_T_std=0.5)


def _sample(option_types=("call", "put"), S0=100.0, K=100.0, sigma=0.2, r=0.05, T=1.0, **kw):
    return process_sample(S0, K, sigma, NORM["log_sigma_mean"], NORM["log_sigma_std"], r,
                          NORM["r_mean"], NORM["r_std"], T, NORM["log_T_mean"], NORM["log_T_std"],
                          500, 20, option_types, **kw)


@pytest.mark.parametrize("S0, K", [(100.0, 100.0), (120.0, 100.0), (80.0, 100.0)])
def test_call_label_reconstructs_black_scholes_price(S0, K):
    # Calls use the control variate, so label*K + smooth intrinsic is exactly BS
    # (an American call without dividends is never exercised early).
    sim = GBM(0.2, S0, 0.05, seed=1).simulate_paths(500, 1.0, 20)
    label = gen_data(sim, S0, K, 0.05, 1.0, 0.2, "call", 3, "simple")
    price = label * K + smooth_intrinsic_value(S0, K, "call")
    assert price == pytest.approx(BlackScholes(0.2, S0, K, 0.05, "call").calculate_option_price(1.0), abs=1e-4)

def test_put_label_is_not_the_black_scholes_european_price():
    # Puts use plain LSM (early exercise has value), so the label must differ
    # from the European price; a deep ITM put makes the gap large vs MC noise.
    S0, K = 70.0, 100.0
    sim = GBM(0.2, S0, 0.05, seed=1).simulate_paths(5000, 1.0, 20)
    label = gen_data(sim, S0, K, 0.05, 1.0, 0.2, "put", 3, "simple")
    price = label * K + smooth_intrinsic_value(S0, K, "put")
    eu = BlackScholes(0.2, S0, K, 0.05, "put").calculate_option_price(1.0)
    assert price > eu + 0.3  # early-exercise premium here is ~0.7+

def test_process_sample_returns_call_then_put_rows_with_is_call_feature():
    rows = _sample()
    assert len(rows) == 2
    (f_call, _), (f_put, _) = rows
    assert len(f_call) == len(f_put) == 5
    assert f_call[4] == 1.0 and f_put[4] == 0.0
    assert f_call[:4] == f_put[:4]  # same market inputs, one simulation
    assert f_call[0] == pytest.approx(1.0)
    assert f_call[2] == pytest.approx((0.05 - 0.03) / 0.02)
    assert f_call[3] == pytest.approx(0.0)  # sigma == exp(log_sigma_mean)

def test_process_sample_single_type_gives_one_row():
    rows = _sample(option_types=("put",))
    assert len(rows) == 1 and rows[0][0][4] == 0.0

def test_process_sample_returns_empty_list_on_failure(capsys):
    # Regression: failures used to return (None, None); main() flattens lists, so [] is required.
    assert _sample(sigma=-1.0) == []
    assert "Skipping sample" in capsys.readouterr().out


# --- gen_rdm_params ---

RANGES = dict(S0_range=(60, 140), moneyness_range=(0.4, 1.6), sigma_range=(0.05, 0.6),
              r_range=(0.0, 0.08), T_range=(0.25, 5))


def test_gen_rdm_params_values_stay_inside_their_ranges():
    np.random.seed(0)
    S0, K, sigma, r, T = gen_rdm_params(500, samples_per_param=3, **RANGES)
    assert S0.min() >= 60 and S0.max() <= 140
    assert sigma.min() >= 0.05 and sigma.max() <= 0.6
    assert r.min() >= 0.0 and r.max() <= 0.08
    assert T.min() >= 0.25 and T.max() <= 5
    assert (K > 0).all()

def test_gen_rdm_params_moneyness_is_inside_range_and_log_uniform():
    np.random.seed(1)
    S0, K, *_ = gen_rdm_params(2000, samples_per_param=2, **RANGES)
    m = S0 / K
    assert m.min() >= 0.4 - 1e-12 and m.max() <= 1.6 + 1e-12
    # Log-uniform on (0.4, 1.6): the median of log m is the midpoint log(0.8), not log(1.0).
    # With 4000 draws the median's std error is ~0.014 in log space; 0.06 is >4 sigma away from log(1).
    assert np.median(np.log(m)) == pytest.approx(np.log(0.8), abs=0.06)
    # Continuous coverage: every tenth of the log range is populated (rules out a few discrete buckets).
    counts, _ = np.histogram(np.log(m), bins=10, range=(np.log(0.4), np.log(1.6)))
    assert (counts > 200).all()

def test_gen_rdm_params_repeats_each_combination_samples_per_param_times_with_new_strikes():
    np.random.seed(2)
    S0, K, sigma, r, T = gen_rdm_params(6, samples_per_param=4, **RANGES)
    for arr in (S0, K, sigma, r, T):
        assert arr.shape == (24,)
    for arr in (S0, sigma, r, T):
        groups = arr.reshape(6, 4)
        assert (groups == groups[:, :1]).all()       # shared within a combination
        assert len(np.unique(groups[:, 0])) == 6      # distinct across combinations
    assert all(len(np.unique(k)) == 4 for k in K.reshape(6, 4))  # a different strike per repeat

def test_gen_rdm_params_is_reproducible_under_numpy_seed():
    np.random.seed(5); a = gen_rdm_params(10)
    np.random.seed(5); b = gen_rdm_params(10)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


# --- chunked, resumable generation ---

def _args_list(n):
    return [(100.0 + i, 100.0, 0.2, NORM["log_sigma_mean"], NORM["log_sigma_std"], 0.05, NORM["r_mean"],
             NORM["r_std"], 1.0, NORM["log_T_mean"], NORM["log_T_std"], 200, 10, ("put",), 2, "simple", 1000 + i)
            for i in range(n)]

def test_generate_chunks_resumes_by_redoing_only_the_missing_chunk(tmp_path):
    args_list = _args_list(5)
    generate_chunks(args_list, tmp_path, chunk_size=2, n_jobs=1)
    assert [chunk_file(tmp_path, i).exists() for i in range(3)] == [True] * 3
    full = load_chunks(tmp_path, 3)
    assert full[0].shape == (5, 5) and full[1].shape == (5,)  # the last chunk holds the single leftover sample

    kept = chunk_file(tmp_path, 0).stat().st_mtime_ns
    chunk_file(tmp_path, 1).unlink()
    generate_chunks(args_list, tmp_path, chunk_size=2, n_jobs=1)

    assert chunk_file(tmp_path, 0).stat().st_mtime_ns == kept  # finished chunks are not recomputed
    resumed = load_chunks(tmp_path, 3)
    # the redone chunk is identical to the uninterrupted run, labels included (each sample is seeded)
    np.testing.assert_array_equal(resumed[0], full[0])
    np.testing.assert_array_equal(resumed[1], full[1])

def test_process_sample_is_reproducible_for_a_seed_and_differs_across_seeds():
    a, b, other = _sample(seed=3), _sample(seed=3), _sample(seed=4)
    assert a == b
    assert [label for _, label in a] != [label for _, label in other]
