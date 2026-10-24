"""Simple classification and regression metrics."""

import numpy as np

__all__ = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "mse",
]


def _check_inputs(y_true, y_pred):
    """Convert inputs to arrays and check that they have the same size."""
    true = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    if true.size != pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    return true, pred


def accuracy(y_true, y_pred):
    """Return the proportion of correct predictions."""
    true, pred = _check_inputs(y_true, y_pred)
    if true.size == 0:
        raise ValueError("cannot calculate accuracy for empty data")
    return np.mean(true == pred)


def confusion_matrix(y_true, y_pred, labels=None):
    """Return a confusion matrix with rows for true labels."""
    true, pred = _check_inputs(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate((true, pred)))
    else:
        labels = np.asarray(labels)

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = np.sum((true == true_label) & (pred == pred_label))
    return matrix


def precision(y_true, y_pred, positive_label=1):
    """Return binary precision for the selected positive label."""
    true, pred = _check_inputs(y_true, y_pred)
    true_positive = np.sum((true == positive_label) & (pred == positive_label))
    false_positive = np.sum((true != positive_label) & (pred == positive_label))
    total = true_positive + false_positive
    return 0.0 if total == 0 else true_positive / total


def recall(y_true, y_pred, positive_label=1):
    """Return binary recall for the selected positive label."""
    true, pred = _check_inputs(y_true, y_pred)
    true_positive = np.sum((true == positive_label) & (pred == positive_label))
    false_negative = np.sum((true == positive_label) & (pred != positive_label))
    total = true_positive + false_negative
    return 0.0 if total == 0 else true_positive / total


def f1(y_true, y_pred, positive_label=1):
    """Return the binary F1 score for the selected positive label."""
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def mse(y_true, y_pred):
    """Return the mean squared error."""
    true, pred = _check_inputs(y_true, y_pred)
    if true.size == 0:
        raise ValueError("cannot calculate MSE for empty data")
    return np.mean((true - pred) ** 2)
