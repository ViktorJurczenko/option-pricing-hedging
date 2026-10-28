import numpy as np

from optionpricing.pricing.american.basis_functions import (
    laguerre_deriv_k,
    laguerre_poly_gen,
    simple_poly,
    simple_poly_deriv_k,
)

X = np.array([0.5, 1.0, 1.5, 2.0])
MAX_DEGREE = 4


def test_simple_poly_matches_hand_computation():
    poly = simple_poly(np.array([2.0]), max_degree=3)
    # (1, x, x^2, x^3) at x=2 -> (1, 2, 4, 8)
    np.testing.assert_allclose(poly, [[1.0, 2.0, 4.0, 8.0]])


def test_simple_poly_first_derivative_matches_finite_difference():
    h = 1e-5
    analytic = simple_poly_deriv_k(X, MAX_DEGREE, 1)
    numeric = (simple_poly(X + h, MAX_DEGREE) - simple_poly(X - h, MAX_DEGREE)) / (2 * h)
    np.testing.assert_allclose(analytic, numeric, atol=1e-3)


def test_simple_poly_second_derivative_matches_finite_difference():
    h = 1e-3
    analytic = simple_poly_deriv_k(X, MAX_DEGREE, 2)
    numeric = (simple_poly(X + h, MAX_DEGREE) - 2 * simple_poly(X, MAX_DEGREE) + simple_poly(X - h, MAX_DEGREE)) / (
        h**2
    )
    np.testing.assert_allclose(analytic, numeric, atol=1e-1)


def test_laguerre_poly_initial_terms():
    poly = laguerre_poly_gen(np.array([2.0]), max_degree=1)
    # L0 = 1, L1 = 1 - x
    np.testing.assert_allclose(poly, [[1.0, -1.0]])


def test_laguerre_first_derivative_matches_finite_difference():
    h = 1e-5
    analytic = laguerre_deriv_k(X, MAX_DEGREE, 1)
    numeric = (laguerre_poly_gen(X + h, MAX_DEGREE) - laguerre_poly_gen(X - h, MAX_DEGREE)) / (2 * h)
    np.testing.assert_allclose(analytic, numeric, atol=1e-3)


def test_laguerre_second_derivative_matches_finite_difference():
    h = 1e-3
    analytic = laguerre_deriv_k(X, MAX_DEGREE, 2)
    numeric = (
        laguerre_poly_gen(X + h, MAX_DEGREE)
        - 2 * laguerre_poly_gen(X, MAX_DEGREE)
        + laguerre_poly_gen(X - h, MAX_DEGREE)
    ) / (h**2)
    np.testing.assert_allclose(analytic, numeric, atol=1e-1)
