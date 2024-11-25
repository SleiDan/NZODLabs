from typing import List
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes


def preprocess(X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    return [X_train, X_test, y_train, y_test]


def get_regression_data() -> List[np.ndarray]:
    data = load_diabetes()
    X, y = data.data, data.target
    return preprocess(X, y)


def get_classification_data() -> List[np.ndarray]:
    data = load_breast_cancer()
    X, y = data.data, data.target
    return preprocess(X, y)


def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    model = LinearRegression()
    model.fit(X, y)
    return model


def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    parameters = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    model = Ridge()
    grid = GridSearchCV(model, parameters, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_


def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    parameters = {'alpha': [0.1, 0.5, 1.0, 5.0]}
    model = Lasso()
    grid = GridSearchCV(model, parameters, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_


def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    model = LogisticRegression(penalty=None, max_iter=10000)
    model.fit(X, y)
    return model


def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    parameters = {'C': [0.1, 1.0, 10.0, 100.0]}
    model = LogisticRegression(penalty='l2', max_iter=10000)
    grid = GridSearchCV(model, parameters, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_


def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    parameters = {'C': [0.1, 1.0, 10.0, 100.0]}
    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
    grid = GridSearchCV(model, parameters, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_
