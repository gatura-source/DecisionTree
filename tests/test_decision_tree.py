"""Tests for the DecisionTreeClassifier."""

import numpy as np
import pytest

from ds import DecisionTree, DecisionTreeClassifier, generate_samples, train_test_split


def make_dataset(n=200, seed=42):
    """Small learnable dataset: label = 1 if feature0 > feature1."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 4)
    y = (X[:, 0] > X[:, 1]).astype(int)
    return X, y


def test_fit_predict_accuracy():
    X, y = make_dataset()
    dt = DecisionTreeClassifier(max_depth=5, random_state=0)
    dt.fit(X, y)
    assert dt.score(X, y) > 0.9


def test_backwards_compatible_alias():
    assert DecisionTree is DecisionTreeClassifier
    dt = DecisionTree(max_depth=3)
    assert isinstance(dt, DecisionTreeClassifier)


def test_predict_before_fit_raises():
    dt = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        dt.predict(np.zeros((1, 2)))


def test_mismatched_lengths_raise():
    dt = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        dt.fit(np.zeros((5, 2)), np.zeros(4))


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_depth=0)
    with pytest.raises(ValueError):
        DecisionTreeClassifier(min_samples_split=1)
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="foo")


def test_train_test_split_sizes():
    X, y = make_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
    assert len(Xtr) == 160
    assert len(Xte) == 40
    assert len(ytr) == len(Xtr)
    assert len(yte) == len(Xte)


def test_train_test_split_bad_test_size():
    with pytest.raises(ValueError):
        train_test_split(np.zeros((5, 2)), np.zeros(5), test_size=1.5)


def test_predict_output_shape():
    X, y = make_dataset(n=100)
    dt = DecisionTreeClassifier(max_depth=4, random_state=0)
    dt.fit(X, y)
    preds = dt.predict(X[:10])
    assert preds.shape == (10,)
    assert set(preds).issubset({0, 1})


def test_generate_samples_reproducible():
    a = generate_samples(n=50, seed=7)
    b = generate_samples(n=50, seed=7)
    assert a == b
    assert all("label" in s and "feature1" in s for s in a)


def test_single_class_leaf():
    X = np.zeros((10, 2))
    y = np.zeros(10, dtype=int)
    dt = DecisionTreeClassifier()
    dt.fit(X, y)
    assert np.all(dt.predict(X) == 0)
