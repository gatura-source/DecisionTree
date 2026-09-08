# DecisionTree

A decision tree classifier implemented from scratch in Python. Splitting is
based on **entropy** (or **Gini impurity**) and **information gain**, giving a
hands-on understanding of the inner workings of decision trees.

## Features

- From-scratch `DecisionTreeClassifier` with:
  - Entropy and Gini impurity criteria
  - Information-gain-based best-split search
  - Recursive tree growth with stopping criteria (`max_depth`, `min_samples_split`)
  - Random feature subsampling (`max_features`) with seeded `random_state`
  - `fit`, `predict`, `predict_proba`, and `score` methods
  - Pretty-printed tree visualization (`print_tree`)
- `train_test_split` helper
- Synthetic data generation (`generate_samples`)
- A demo script and a pytest test suite
- Input validation with descriptive error messages

## Requirements

- Python 3.8+
- NumPy

## Installation

```bash
make install
# or
pip install -r requirements.txt
```

## Usage

### Run the demo

```bash
make run
```

Or directly:

```bash
python ds.py
```

This trains a tree on synthetic data, prints the tree structure, and reports
the test accuracy.

### In your own code

```python
import numpy as np
from ds import DecisionTreeClassifier, train_test_split, generate_samples

# Generate synthetic data (label = 1 when feature1 > feature2, plus noise)
raw = generate_samples(500, seed=42)
X = np.array([[s["feature1"], s["feature2"], s["feature3"], s["feature4"]] for s in raw])
y = np.array([s["label"] for s in raw])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, criterion="entropy", random_state=42)
dt.fit(X_train, y_train)

predictions = dt.predict(X_test)
print(dt.score(X_test, y_test))
dt.print_tree()
```

### Development

```bash
make test     # run tests
make lint     # lint with flake8
make format   # format with black
make clean    # remove cache artifacts
```

## Testing

The test suite lives in `tests/` and uses pytest:

```bash
make test
```

## License

MIT
