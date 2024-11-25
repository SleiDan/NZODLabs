import pytest
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from regularization import (
    get_regression_data,
    get_classification_data,
    linear_regression,
    ridge_regression,
    lasso_regression,
    logistic_regression,
    logistic_l2_regression,
    logistic_l1_regression,
)


def test_linear_regression():
    X_train, X_test, y_train, y_test = get_regression_data()
    model = linear_regression(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    assert mse < 4000, f"Linear regression MSE too high: {mse}"


def test_ridge_regression():
    X_train, X_test, y_train, y_test = get_regression_data()
    model = ridge_regression(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    assert mse < 4000, f"Ridge regression MSE too high: {mse}"


def test_lasso_regression():
    X_train, X_test, y_train, y_test = get_regression_data()
    model = lasso_regression(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    assert mse < 4000, f"Lasso regression MSE too high: {mse}"


def test_logistic_regression():
    X_train, X_test, y_train, y_test = get_classification_data()
    model = logistic_regression(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.9, f"Logistic regression accuracy too low: {accuracy}"


def test_logistic_l2_regression():
    X_train, X_test, y_train, y_test = get_classification_data()
    model = logistic_l2_regression(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.9, f"Logistic regression (L2) accuracy too low: {accuracy}"


def test_logistic_l1_regression():
    X_train, X_test, y_train, y_test = get_classification_data()
    model = logistic_l1_regression(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.9, f"Logistic regression (L1) accuracy too low: {accuracy}"
