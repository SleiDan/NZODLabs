import numpy as np
from typing import Tuple

def lu_decomposition(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform LU decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            The permutation matrix P, lower triangular matrix L, and upper triangular matrix U.
    """
    from scipy.linalg import lu
    P, L, U = lu(x)
    return P, L, U

def qr_decomposition(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform QR decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The orthogonal matrix Q and upper triangular matrix R.
    """
    Q, R = np.linalg.qr(x)
    return Q, R

def determinant(x: np.ndarray) -> float:
    """
    Calculate the determinant of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        float: The determinant of the matrix.
    """
    return np.linalg.det(x)

def eigen(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and right eigenvectors of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The eigenvalues and the right eigenvectors of the matrix.
    """
    values, vectors = np.linalg.eig(x)
    return values, vectors

def svd(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Singular Value Decomposition (SVD) of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The matrices U, S, and V.
    """
    U, S, V = np.linalg.svd(x)
    return U, S, V


def test_lu_decomposition():
    x = np.array([[4, 3], [6, 3]])
    expected_P = np.array([[0., 1.], [1., 0.]])
    expected_L = np.array([[1., 0.], [0.66666667, 1.]])
    expected_U = np.array([[6., 3.], [0., 1.]])
    P, L, U = lu_decomposition(x)
    print("LU Decomposition Test")
    print("Input Matrix:\n", x)
    print("Expected P:\n", expected_P)
    print("Actual P:\n", P)
    print("Expected L:\n", expected_L)
    print("Actual L:\n", L)
    print("Expected U:\n", expected_U)
    print("Actual U:\n", U)

def test_qr_decomposition():
    x = np.array([[1, 2], [3, 4]])
    expected_Q = np.array([[-0.31622777, -0.9486833], [-0.9486833, 0.31622777]])
    expected_R = np.array([[-3.16227766, -4.42718872], [0., -0.63245553]])
    Q, R = qr_decomposition(x)
    print("QR Decomposition Test")
    print("Input Matrix:\n", x)
    print("Expected Q:\n", expected_Q)
    print("Actual Q:\n", Q)
    print("Expected R:\n", expected_R)
    print("Actual R:\n", R)

def test_determinant():
    x = np.array([[1, 2], [3, 4]])
    expected_result = -2.0
    result = determinant(x)
    print("Determinant Test")
    print("Input Matrix:\n", x)
    print("Expected Determinant:", expected_result)
    print("Actual Determinant:", result)

def test_eigen():
    x = np.array([[1, 2], [3, 4]])
    expected_values = np.array([-0.37228132, 5.37228132])
    expected_vectors = np.array([[-0.82456484, -0.41597356], [0.56576746, -0.90937671]])
    values, vectors = eigen(x)
    print("Eigenvalues and Eigenvectors Test")
    print("Input Matrix:\n", x)
    print("Expected Eigenvalues:\n", expected_values)
    print("Actual Eigenvalues:\n", values)
    print("Expected Eigenvectors:\n", expected_vectors)
    print("Actual Eigenvectors:\n", vectors)

def test_svd():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    expected_U = np.array([[-0.2298477, 0.88346102, 0.40824829],
                           [-0.52474482, 0.24078249, -0.81649658],
                           [-0.81964194, -0.40189604, 0.40824829]])
    expected_S = np.array([9.52551809, 0.51430058])
    expected_V = np.array([[-0.61962948, -0.78489445],
                           [-0.78489445, 0.61962948]])
    U, S, V = svd(x)
    print("Singular Value Decomposition Test")
    print("Input Matrix:\n", x)
    print("Expected U:\n", expected_U)
    print("Actual U:\n", U)
    print("Expected Singular Values S:\n", expected_S)
    print("Actual Singular Values S:\n", S)
    print("Expected V:\n", expected_V)
    print("Actual V:\n", V)

test_lu_decomposition()
test_qr_decomposition()
test_determinant()
test_eigen()
test_svd()