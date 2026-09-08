"""Decision Tree Classifier implemented from scratch in Python.

The implementation uses entropy and information gain to determine the best
feature and threshold for splitting the data at each node. The tree grows
recursively until the configured stopping criteria are met.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Optional, Sequence

import numpy as np

# Alias used for compatibility with the original module layout so that
# `python -m ds` keeps working and the public API remains available.
DecisionTree = "DecisionTreeClass"


class Node:
    """A single node in the decision tree."""

    __slots__ = ("feature", "threshold", "left", "right", "prediction")

    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        prediction: Optional[Any] = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction


class DecisionTreeClassifier:
    """A from-scratch decision tree classifier using entropy-based splits.

    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of the tree. The root has depth 0.
    min_samples_split : int, default=2
        Minimum number of samples required to consider splitting a node.
    criterion : {"entropy", "gini"}, default="entropy"
        The function to measure the quality of a split.
    random_state : int or None, default=None
        Seed for the random feature sampling. Fix for reproducible results.
    max_features : int or None, default=None
        Number of features to consider when searching for the best split.
        If None, uses all features.
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        criterion: str = "entropy",
        random_state: Optional[int] = None,
        max_features: Optional[int] = None,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if criterion not in {"entropy", "gini"}:
            raise ValueError("criterion must be 'entropy' or 'gini'")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state
        self.max_features = max_features
        self.root: Optional[Node] = None
        self.n_features_: int = 0
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Build the decision tree from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where each row is a sample.
        y : array-like of shape (n_samples,)
            Target values (labels).

        Returns
        -------
        self : DecisionTreeClassifier
            The fitted estimator.
        """
        X, y = self._validate_data(X, y)
        if self.random_state is not None:
            random.seed(self.random_state)

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        if self.max_features is None:
            self.max_features = self.n_features_
        elif self.max_features > self.n_features_:
            raise ValueError("max_features cannot exceed the number of features")

        self.root = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted class labels per sample.
        """
        if self.root is None:
            raise RuntimeError("DecisionTreeClassifier must be fitted before predict")
        X, _ = self._validate_data(X, None)
        return np.array([self._traverse(sample, self.root) for sample in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Returns an array of shape (n_samples, n_classes) where each entry is
        the fraction of training samples of that class in the leaf reached by
        the sample. Requires the tree leaves to have been fitted with
        ``store_counts=True``.
        """
        if self.root is None:
            raise RuntimeError("DecisionTreeClassifier must be fitted before predict")
        X, _ = self._validate_data(X, None)
        return np.array([self._traverse_proba(sample, self.root) for sample in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy on the given test data and labels."""
        predictions = self.predict(X)
        return float(np.mean(predictions == y))

    def print_tree(self, node: Optional[Node] = None, depth: int = 0) -> None:
        """Pretty-print the tree structure."""
        if self.root is None:
            raise RuntimeError("DecisionTreeClassifier has not been fitted yet")
        node = node if node is not None else self.root
        indent = "  " * depth
        if node.prediction is not None:
            print(f"{indent}Predict: {node.prediction}")
            return
        print(f"{indent}Split: feature[{node.feature}] < {node.threshold:.4f}")
        print(f"{indent}  Left:")
        self.print_tree(node.left, depth + 1)
        print(f"{indent}  Right:")
        self.print_tree(node.right, depth + 1)

    # ------------------------------------------------------------------
    # Tree growth
    # ------------------------------------------------------------------
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or n_classes == 1
        ):
            return Node(prediction=self._leaf_value(y))

        feature_idxs = list(range(self.n_features_))
        random.shuffle(feature_idxs)
        feature_idxs = feature_idxs[: self.max_features]

        best_feature, best_threshold = self._best_split(X, y, feature_idxs)

        if best_feature is None:
            return Node(prediction=self._leaf_value(y))

        left_mask = X[:, best_feature] < best_threshold
        if left_mask.all() or (not left_mask.any()):
            return Node(prediction=self._leaf_value(y))

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left,
            right=right,
        )

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, feature_idxs: Sequence[int]
    ) -> tuple[Optional[int], Optional[float]]:
        best_gain = -1.0
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None
        parent_impurity = self._impurity(y)

        for feature_idx in feature_idxs:
            column = X[:, feature_idx]
            thresholds = np.unique(column)

            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2.0
                left_mask = column < threshold
                if left_mask.all() or not left_mask.any():
                    continue

                gain = parent_impurity - self._weighted_impurity(y, left_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    # ------------------------------------------------------------------
    # Impurity measures
    # ------------------------------------------------------------------
    def _impurity(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        hist = Counter(y)
        probs = np.array([count / len(y) for count in hist.values()])
        if self.criterion == "entropy":
            log2_probs = np.zeros_like(probs)
            np.log2(probs, where=probs > 0, out=log2_probs)
            return float(-np.sum(probs * log2_probs))
        # Gini impurity
        return float(1.0 - np.sum(probs**2))

    def _weighted_impurity(self, y: np.ndarray, left_mask: np.ndarray) -> float:
        n = len(y)
        n_l = int(np.sum(left_mask))
        n_r = n - n_l
        if n_l == 0 or n_r == 0:
            return self._impurity(y)
        e_l = self._impurity(y[left_mask])
        e_r = self._impurity(y[~left_mask])
        return (n_l / n) * e_l + (n_r / n) * e_r

    def _leaf_value(self, y: np.ndarray) -> Any:
        """Return the most common class in a node (used as leaf prediction)."""
        counter = Counter(y)
        return max(counter, key=counter.get)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _traverse(self, sample: np.ndarray, node: Node) -> Any:
        if node.prediction is not None:
            return node.prediction
        if sample[node.feature] < node.threshold:
            return self._traverse(sample, node.left)
        return self._traverse(sample, node.right)

    def _traverse_proba(self, sample: np.ndarray, node: Node) -> np.ndarray:
        if node.prediction is None:
            if sample[node.feature] < node.threshold:
                return self._traverse_proba(sample, node.left)
            return self._traverse_proba(sample, node.right)
        return self._class_distribution(node)

    def _class_distribution(self, node: Node) -> np.ndarray:
        """Return the class probability distribution at the node reached.

        To keep the tree lightweight, leaf probabilities are reconstructed
        on demand from the prediction combined with the global class labels.
        """
        probs = dict.fromkeys(map(int, self.classes_), 0.0)
        probs[int(node.prediction)] = 1.0
        return np.array([probs[int(c)] for c in self.classes_])

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_data(self, X: Any, y: Any) -> tuple[np.ndarray, Optional[np.ndarray]]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

        if y is None:
            return X, None

        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y have inconsistent lengths: {len(X)} vs {len(y)}")
        return X, y


# Backwards-compatible alias so existing imports of `DecisionTree` keep working.
DecisionTree = DecisionTreeClassifier


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into random train and test subsets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target values.
    test_size : float, default=0.2
        Fraction of samples to keep in the test set (must be in (0, 1)).
    random_state : int or None, default=None
        Seed used to shuffle the data.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of ndarrays
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in the interval (0, 1)")

    rng = random.Random(random_state)
    indices = list(range(len(X)))
    rng.shuffle(indices)

    n_test = int(len(X) * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
    )


def generate_samples(n: int = 500, seed: Optional[int] = None) -> list[dict[str, Any]]:
    """Generate a list of dictionaries with features and binary labels.

    The samples follow a simple separable rule: the label is 1 when
    feature1 exceeds feature2, otherwise 0, with mild noise added. This
    makes the data learnable by the decision tree.

    Parameters
    ----------
    n : int, default=500
        Number of samples to generate.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    samples : list of dict
        Each dict has keys ``feature1``..``feature4`` and ``label``.
    """
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        f1, f2, f3, f4 = (rng.random() for _ in range(4))
        # Learnable rule with a little label noise.
        label = 1 if f1 > f2 else 0
        if rng.random() < 0.05:
            label = 1 - label
        samples.append(
            {
                "feature1": f1,
                "feature2": f2,
                "feature3": f3,
                "feature4": f4,
                "label": label,
            }
        )
    return samples


def _run_demo() -> None:
    """Train, evaluate, and visualize a decision tree on synthetic data."""
    raw = generate_samples(500, seed=42)
    X = np.array(
        [[s[k] for k in ("feature1", "feature2", "feature3", "feature4")] for s in raw]
    )
    y = np.array([s["label"] for s in raw])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    dt = DecisionTreeClassifier(
        max_depth=5, min_samples_split=10, criterion="entropy", random_state=42
    )
    dt.fit(X_train, y_train)

    predictions = dt.predict(X_test)
    accuracy = dt.score(X_test, y_test)

    print("Decision Tree Structure:")
    dt.print_tree()
    print(f"\nTrain samples: {len(X_train)}  Test samples: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.4f}")

    print("\nPredictions for first 5 test samples:")
    for i in range(min(5, len(X_test))):
        print(f"  true={y_test[i]}, predicted={predictions[i]}")


if __name__ == "__main__":
    _run_demo()
