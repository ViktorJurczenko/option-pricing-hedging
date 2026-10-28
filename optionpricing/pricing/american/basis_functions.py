import numpy as np


def simple_poly(x: np.ndarray, max_degree: int = 3) -> np.ndarray:
    """
    Creates a 2D-array containing a polynomial of the input x at each dimension like
    (1, x, x^2, ...)

    Args:
        x: The evaluation of the polynomial
        max_degree: The maximum degree of the polynomial

    Returns:
        polynomials 2D-array (x.shape[0], max_degree + 1) (+1 for the bias term)
    """
    poly = np.zeros((max_degree + 1, x.shape[0]))

    for i in range(0, max_degree + 1):
        poly[i] = x**i

    return poly.T


def simple_poly_deriv_k(x: np.ndarray, max_degree: int = 3, k: int = 0) -> np.ndarray:
    poly = np.zeros((max_degree + 1, x.shape[0]))

    for i in range(0, max_degree + 1):
        if i < k:
            poly[i] = 0
        else:
            coeff = np.prod(range(i - k + 1, i + 1)) if k > 0 else 1
            poly[i] = coeff * x ** (i - k)

    return poly.T


def laguerre_poly_gen(x: np.ndarray, max_degree: int = 3, alpha: int = 0) -> np.ndarray:
    """
    Creates a 2D-array containing the Laguerre polynomial of the input x at each dimension

    Args:
        x: The evaluation of the Laguerre polynomial
        max_degree: The maximum degree of the Laguerre polynomial

    Returns:
        polynomials 2D-array (x.shape[0], max_degree + 1) (+1 for the bias term)
    """

    poly = np.zeros((max_degree + 1, x.shape[0]))

    # Initial conditions
    poly[0] = 1
    poly[1] = 1 + alpha - x

    for i in range(2, max_degree + 1):
        k = i - 1
        # Closed form solution for Laguerre polynomial
        poly[i] = ((2 * k + alpha + 1 - x) * poly[k] - (k + alpha) * poly[k - 1]) / (k + 1)

    return poly.T


def laguerre_deriv_k(x: np.ndarray, max_degree: int = 3, k: int = 0, alpha: int = 0) -> np.ndarray:
    deriv = np.zeros((x.shape[0], max_degree + 1))

    poly_alpha = laguerre_poly_gen(x, max_degree, alpha + k)
    # The k-th derivative: L_n^(alpha)(k)(x) = (-1)^k * L_{n-k}^{alpha+k}(x), zero for n < k
    deriv[:, k:] = ((-1) ** k) * poly_alpha[:, : -k if k > 0 else None]
    return deriv
