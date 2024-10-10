from typing import Sequence
from typing import Union
import numpy as np
from scipy import sparse


def get_vector(dim: int) -> np.ndarray:
    """Create random column vector with dimension dim."""
    return np.random.rand(dim, 1)


def get_sparse_vector(dim: int) -> sparse.coo_matrix:
    """Create random sparse column vector with dimension dim."""
    data = np.random.rand(dim)
    rows = np.arange(dim)
    cols = np.zeros(dim)
    return sparse.coo_matrix((data, (rows, cols)), shape=(dim, 1))


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector addition."""
    return np.add(x, y)


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Vector multiplication by scalar."""
    return np.multiply(x, a)


def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    """Linear combination of vectors."""
    return sum(c * v for c, v in zip(coeffs, vectors))


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Vectors dot product."""
    return np.dot(x.T, y).item()


def norm(x: np.ndarray, order: Union[int, float] = 2) -> float:
    """Vector norm: Manhattan (1), Euclidean (2), or Max (inf)."""
    return np.linalg.norm(x, ord=order)


def distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between vectors."""
    return np.linalg.norm(x - y)


def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine of the angle between vectors in degrees."""
    cos_theta = dot_product(x, y) / (norm(x) * norm(y))
    return np.degrees(np.arccos(cos_theta))


def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    """Check if vectors are orthogonal."""
    return np.isclose(dot_product(x, y), 0)


def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system of linear equations."""
    return np.linalg.solve(a, b)

def test_get_vector():
    result = get_vector(3)
    print("Test: get_vector(3)")
    print("Expected: 3x1 random column vector")
    print(f"Provided:\n{result}\n")


def test_get_sparse_vector():
    result = get_sparse_vector(3)
    print("Test: get_sparse_vector(3)")
    print("Expected: 3x1 sparse column vector")
    print(f"Provided:\n{result}\n")


def test_add():
    x = np.array([[1], [2], [3]])
    y = np.array([[4], [5], [6]])
    result = add(x, y)
    print("Test: add([[1], [2], [3]], [[4], [5], [6]])")
    print("Expected: [[5], [7], [9]]")
    print(f"Provided:\n{result}\n")


def test_scalar_multiplication():
    x = np.array([[1], [2], [3]])
    result = scalar_multiplication(x, 2)
    print("Test: scalar_multiplication([[1], [2], [3]], 2)")
    print("Expected: [[2], [4], [6]]")
    print(f"Provided:\n{result}\n")


def test_linear_combination():
    vectors = [np.array([[1], [2]]), np.array([[3], [4]])]
    coeffs = [2, 3]
    result = linear_combination(vectors, coeffs)
    print("Test: linear_combination([[[1], [2]], [[3], [4]]], [2, 3])")
    print("Expected: [[11], [16]]")
    print(f"Provided:\n{result}\n")


def test_dot_product():
    x = np.array([[1], [2], [3]])
    y = np.array([[4], [5], [6]])
    result = dot_product(x, y)
    print("Test: dot_product([[1], [2], [3]], [[4], [5], [6]])")
    print("Expected: 32")
    print(f"Provided: {result}\n")


def test_norm():
    x = np.array([[1], [2], [3]])
    result = norm(x, 2)
    print("Test: norm([[1], [2], [3]], 2)")
    print("Expected: 3.74166 (Euclidean norm)")
    print(f"Provided: {result}\n")


def test_distance():
    x = np.array([[1], [2], [3]])
    y = np.array([[4], [5], [6]])
    result = distance(x, y)
    print("Test: distance([[1], [2], [3]], [[4], [5], [6]])")
    print("Expected: 5.19615 (L2 distance)")
    print(f"Provided: {result}\n")


def test_cos_between_vectors():
    x = np.array([[1], [0]])
    y = np.array([[0], [1]])
    result = cos_between_vectors(x, y)
    print("Test: cos_between_vectors([[1], [0]], [[0], [1]])")
    print("Expected: 90 (cosine between orthogonal vectors)")
    print(f"Provided: {result}\n")


def test_is_orthogonal():
    x = np.array([[1], [0]])
    y = np.array([[0], [1]])
    result = is_orthogonal(x, y)
    print("Test: is_orthogonal([[1], [0]], [[0], [1]])")
    print("Expected: True (vectors are orthogonal)")
    print(f"Provided: {result}\n")


def test_solves_linear_systems():
    a = np.array([[3, 1], [1, 2]])
    b = np.array([[9], [8]])
    result = solves_linear_systems(a, b)
    print("Test: solves_linear_systems([[3, 1], [1, 2]], [[9], [8]])")
    print("Expected: [[2], [3]] (solution of the linear system)")
    print(f"Provided:\n{result}\n")


# Call all test functions
test_get_vector()
test_get_sparse_vector()
test_add()
test_scalar_multiplication()
test_linear_combination()
test_dot_product()
test_norm()
test_distance()
test_cos_between_vectors()
test_is_orthogonal()
test_solves_linear_systems()
