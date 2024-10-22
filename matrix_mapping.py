import numpy as np
from typing import Tuple  # Import Tuple from typing

def negative_matrix(x: np.ndarray) -> np.ndarray:
    """ Returns the negation of each element in the input vector or matrix. """
    return -x

def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """ Returns the input vector or matrix with the order of elements reversed. """
    return np.flip(x)

def affine_transform(
    x: np.ndarray, alpha_deg: float, scale: Tuple[float, float], shear: Tuple[float, float],
    translate: Tuple[float, float],
) -> np.ndarray:
    """Compute affine transformation """

    alpha_rad = np.deg2rad(alpha_deg)

    rotation_matrix = np.array([[np.cos(alpha_rad), -np.sin(alpha_rad)],
                                [np.sin(alpha_rad), np.cos(alpha_rad)]])

    scale_matrix = np.array([[scale[0], 0],
                             [0, scale[1]]])

    shear_matrix = np.array([[1, shear[0]],
                             [shear[1], 1]])

    transformed = x @ scale_matrix @ shear_matrix @ rotation_matrix

    transformed[:, 0] += translate[0]
    transformed[:, 1] += translate[1]

    return transformed

def test_negative_matrix():
    x = np.array([[1, 2], [3, 4]])
    expected = np.array([[-1, -2], [-3, -4]])
    result = negative_matrix(x)
    print("Test: negative_matrix([[1, 2], [3, 4]])")
    print(f"Expected:\n{expected}")
    print(f"Provided:\n{result}\n")

def test_reverse_matrix():
    x = np.array([[1, 2], [3, 4]])
    expected = np.array([[4, 3], [2, 1]])
    result = reverse_matrix(x)
    print("Test: reverse_matrix([[1, 2], [3, 4]])")
    print(f"Expected:\n{expected}")
    print(f"Provided:\n{result}\n")

def test_affine_transform():
    x = np.array([[1, 2], [3, 4]])
    alpha_deg = 45
    scale = (1, 1)
    shear = (0, 0)
    translate = (1, 1)
    expected = np.array([[3.12132, 1.70711], [5.94975, 1.70711]])
    result = affine_transform(x, alpha_deg, scale, shear, translate)
    print("Test: affine_transform([[1, 2], [3, 4]], 45, (1, 1), (0, 0), (1, 1))")
    print(f"Expected:\n{expected}")
    print(f"Provided:\n{result}\n")


test_negative_matrix()
test_reverse_matrix()
test_affine_transform()
