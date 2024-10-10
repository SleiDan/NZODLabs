from typing import Tuple, Union
import numpy as np
from scipy.linalg import qr

def get_matrix(n: int, m: int) -> np.ndarray:
    """Create random matrix n * m."""
    return np.random.rand(n, m)

def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrix addition."""
    return x + y

def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Matrix multiplication by scalar."""
    return a * x

def dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrices dot product."""
    return np.dot(x, y)

def identity_matrix(dim: int) -> np.ndarray:
    """Create identity matrix with dimension dim."""
    return np.eye(dim)

def matrix_inverse(x: np.ndarray) -> np.ndarray:
    """Compute inverse matrix."""
    return np.linalg.inv(x)

def matrix_transpose(x: np.ndarray) -> np.ndarray:
    """Compute transpose matrix."""
    return x.T

def hadamard_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute hadamard product."""
    return np.multiply(x, y)

def basis(x: np.ndarray) -> Tuple[int]:
    """Compute matrix basis."""
    Q, R, P = qr(x, pivoting=True)
    rank = np.linalg.matrix_rank(x)
    basis_indexes = tuple(P[:rank])
    return basis_indexes

def norm(x: np.ndarray, order: Union[int, float, str]) -> float:
    """Matrix norm: Frobenius, Spectral or Max."""
    return np.linalg.norm(x, ord=order)

import numpy as np

def test_get_matrix():
    result = get_matrix(3, 3)
    print("Test: get_matrix(3, 3)")
    print("Expected: 3x3 random matrix")
    print(f"Provided:\n{result}\n")

def test_add():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    expected = np.array([[6, 8], [10, 12]])
    result = add(x, y)
    print("Test: add([[1, 2], [3, 4]], [[5, 6], [7, 8]])")
    print(f"Expected:\n{expected}")
    print(f"Provided:\n{result}\n")

def test_scalar_multiplication():
    x = np.array([[1, 2], [3, 4]])
    a = 2
    expected = np.array([[2, 4], [6, 8]])
    result = scalar_multiplication(x, a)
    print("Test: scalar_multiplication([[1, 2], [3, 4]], 2)")
    print(f"Expected:\n{expected}")
    print(f"Provided:\n{result}\n")

def test_dot_product():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    expected = np.array([[19, 22], [43, 50]])
    result = dot_product(x, y)
    print("Test: dot_product([[1, 2], [3, 4]], [[5, 6], [7, 8]])")
    print(f"Expected:\n{expected}")
    print(f"Provided:\n{result}\n")

def test_identity_matrix():
    dim = 3
    expected = np.eye(3)
    result = identity_matrix(dim)
    print("Test: identity_matrix(3)")
    print(f"Expected:\n{expected}")
    print(f"Provided:\n{result}\n")

def test_matrix_inverse():
    x = np.array([[1, 2], [3, 4]])
    expected = np.linalg.inv(x)
    result = matrix_inverse(x)
    print("Test: matrix_inverse([[1, 2], [3, 4]])")
    print(f"Expected:\n{expected}")
    print(f"Provided:\n{result}\n")

def test_matrix_transpose():
    x = np.array([[1, 2], [3, 4]])
    expected = np.array([[1, 3], [2, 4]])
    result = matrix_transpose(x)
    print("Test: matrix_transpose([[1, 2], [3, 4]])")
    print(f"Expected:\n{expected}")
    print(f"Provided:\n{result}\n")

def test_hadamard_product():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    expected = np.array([[5, 12], [21, 32]])
    result = hadamard_product(x, y)
    print("Test: hadamard_product([[1, 2], [3, 4]], [[5, 6], [7, 8]])")
    print(f"Expected:\n{expected}")
    print(f"Provided:\n{result}\n")

def test_basis():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = basis(x)
    print("Test: basis([[1, 2, 3], [4, 5, 6], [7, 8, 9]])")
    print(f"Provided: {result}\n")

def test_norm():
    x = np.array([[1, 2], [3, 4]])
    order = 'fro'
    expected = np.linalg.norm(x, ord='fro')
    result = norm(x, order)
    print("Test: norm([[1, 2], [3, 4]], 'fro')")
    print(f"Expected: {expected}")
    print(f"Provided: {result}\n")


# Run all tests
if __name__ == "__main__":
    test_get_matrix()
    test_add()
    test_scalar_multiplication()
    test_dot_product()
    test_identity_matrix()
    test_matrix_inverse()
    test_matrix_transpose()
    test_hadamard_product()
    test_basis()
    test_norm()
