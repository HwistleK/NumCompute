"""Unit tests for the pipeline module (NumCompute framework)."""
import numpy as np
import pytest
from numcompute.pipeline import Pipeline, Transformer, Estimator
from numcompute.preprocessing import StandardScaler

class DummyMultiplier(Transformer):
    def __init__(self, factor=2.0):
        self.factor = factor
    def transform(self, X):
        return np.asarray(X) * self.factor

class DummyEstimator(Estimator):
    def fit(self, X, y=None):
        self.coef_ = 1.0
        return self
    def predict(self, X):
        return np.asarray(X) * 2

def test_pipeline_transform():
    X = [[1.0, 2.0], [3.0, 4.0]]
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('mult', DummyMultiplier(10.0))
    ])
    Xt = pipe.fit_transform(X)
    assert Xt.shape == (2, 2)

def test_pipeline_predict():
    X = [1, 2, 3]
    pipe = Pipeline([
        ('model', DummyEstimator())
    ])
    preds = pipe.fit(X).predict(X)
    assert preds.tolist() == [2, 4, 6]