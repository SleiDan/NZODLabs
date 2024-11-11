import numpy as np
import pytest
from matrix_decompostion import lu_decomposition, qr_decomposition, determinant, eigen, svd

def test_lu_decomposition():
    x = np.array([[4, 3], [6, 3]])
    expected_P = np.array([[0., 1.], [1., 0.]])
    expected_L = np.array([[1., 0.], [0.66666667, 1.]])
    expected_U = np.array([[6., 3.], [0., 1.]])
    
    P, L, U = lu_decomposition(x)
    
    assert np.allclose(P, expected_P), f"Expected P:\n{expected_P}\nActual P:\n{P}"
    assert np.allclose(L, expected_L), f"Expected L:\n{expected_L}\nActual L:\n{L}"
    assert np.allclose(U, expected_U), f"Expected U:\n{expected_U}\nActual U:\n{U}"

def test_qr_decomposition():
    x = np.array([[1, 2], [3, 4]])
    expected_Q = np.array([[-0.31622777, -0.9486833], [-0.9486833, 0.31622777]])
    expected_R = np.array([[-3.16227766, -4.42718872], [0., -0.63245553]])
    
    Q, R = qr_decomposition(x)
    
    assert np.allclose(Q, expected_Q), f"Expected Q:\n{expected_Q}\nActual Q:\n{Q}"
    assert np.allclose(R, expected_R), f"Expected R:\n{expected_R}\nActual R:\n{R}"

def test_determinant():
    x = np.array([[1, 2], [3, 4]])
    expected_result = -2.0
    
    result = determinant(x)
    
    assert np.isclose(result, expected_result), f"Expected determinant: {expected_result}, Actual determinant: {result}"

def test_eigen():
    x = np.array([[1, 2], [3, 4]])
    expected_values = np.array([-0.37228132, 5.37228132])
    expected_vectors = np.array([[-0.82456484, -0.41597356], [0.56576746, -0.90937671]])
    
    values, vectors = eigen(x)
    
    assert np.allclose(values, expected_values), f"Expected eigenvalues:\n{expected_values}\nActual eigenvalues:\n{values}"
    assert np.allclose(np.abs(vectors), np.abs(expected_vectors)), f"Expected eigenvectors:\n{expected_vectors}\nActual eigenvectors:\n{vectors}"

def test_svd():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    expected_U = np.array([[-0.2298477, 0.88346102, 0.40824829],
                           [-0.52474482, 0.24078249, -0.81649658],
                           [-0.81964194, -0.40189604, 0.40824829]])
    expected_S = np.array([9.52551809, 0.51430058])
    expected_V = np.array([[-0.61962948, -0.78489445],
                           [-0.78489445, 0.61962948]])
    
    U, S, V = svd(x)
    
    assert np.allclose(U, expected_U), f"Expected U:\n{expected_U}\nActual U:\n{U}"
    assert np.allclose(S, expected_S), f"Expected Singular Values S:\n{expected_S}\nActual Singular Values S:\n{S}"
    assert np.allclose(V, expected_V), f"Expected V:\n{expected_V}\nActual V:\n{V}"
