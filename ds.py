import numpy as np
from collections import Counter
from typing import Dict, Any, List
import random

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.prediction = prediction

    def fit(self, X, y):
        self.features = list(X[0].keys())
        self.label = "label"
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_features = len(self.features)
        n_classes = len(set(y))

        # Stopping criteria
        if (depth >= self.max_depth 
            or n_samples < self.min_samples_split 
            or n_classes == 1):
            leaf_value = max(set(y), key=y.count)
            return self.Node(prediction=leaf_value)

        feature_idxs = list(range(n_features))
        random.shuffle(feature_idxs)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, feature_idxs)

        # Split the data
        left_idxs, right_idxs = self._split(X, best_feature, best_threshold)

        # Grow the children
        left = self._grow_tree([X[i] for i in left_idxs], [y[i] for i in left_idxs], depth+1)
        right = self._grow_tree([X[i] for i in right_idxs], [y[i] for i in right_idxs], depth+1)

        return self.Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left,
            right=right
        )

    def _best_split(self, X, y, feature_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in feature_idxs:
            feature_name = self.features[feature_idx]
            X_column = [sample[feature_name] for sample in X]
            thresholds = sorted(set(X_column))

            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i+1]) / 2
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    print(f"Current Best Gain: {gain}")
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return self.features[split_idx], split_threshold

    def _information_gain(self, y, X_column, threshold):
        # Parent entropy
        parent_entropy = self._entropy(y)

        # Generate split
        left_idxs, right_idxs = self._split_by_column(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate the weighted avg. of the entropy for the two groups
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy([y[i] for i in left_idxs]), self._entropy([y[i] for i in right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Calculate the information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X, feature, threshold):
        left_idxs = [i for i, sample in enumerate(X) if sample[feature] < threshold]
        right_idxs = [i for i in range(len(X)) if i not in left_idxs]
        return left_idxs, right_idxs

    def _split_by_column(self, X_column, threshold):
        left_idxs = [i for i, value in enumerate(X_column) if value < threshold]
        right_idxs = [i for i in range(len(X_column)) if i not in left_idxs]
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = Counter(y)
        ps = [count / len(y) for count in hist.values()]
        return -sum(p * np.log2(p) for p in ps if p > 0)

    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        if node.prediction is not None:
            return node.prediction

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        if node.prediction is not None:
            print("  " * depth + f"Predict: {node.prediction}")
            return

        print("  " * depth + f"Split: {node.feature} < {node.threshold}")
        print("  " * depth + "Left:")
        self.print_tree(node.left, depth + 1)
        print("  " * depth + "Right:")
        self.print_tree(node.right, depth + 1)

def generate_samples(n: int = 500) -> List[Dict[str, Any]]:
    """Generate a list of dictionaries with features and labels."""
    def generate_sample() -> Dict[str, Any]:
        return {
            "feature1": random.random(),
            "feature2": random.random(),
            "feature3": random.random(),
            "feature4": random.random(),
            "label": random.choice([0, 1])
        }
    return [generate_sample() for _ in range(n)]

# Usage example
if __name__ == "__main__":
    # Generate samples
    samples = generate_samples(500)

    # Separate features and labels
    X = [{k: v for k, v in sample.items() if k != 'label'} for sample in samples]
    y = [sample['label'] for sample in samples]

    # Create and train the decision tree
    dt = DecisionTree(max_depth=5, min_samples_split=10)
    dt.fit(X, y)

    # Print the tree structure
    print("Decision Tree Structure:")
    dt.print_tree()

    # Make predictions
    test_samples = generate_samples(5)
    X_test = [{k: v for k, v in sample.items() if k != 'label'} for sample in test_samples]
    predictions = dt.predict(X_test)

    print("\nPredictions for 5 test samples:")
    for i, (sample, prediction) in enumerate(zip(X_test, predictions)):
        print(f"Sample {i+1}: {sample}")
        print(f"Predicted: {prediction}\n")
