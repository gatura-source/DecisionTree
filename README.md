# DecisionTree
A simple Decision Tree using Python that uses entropy
# Decision Tree Classifier

This project implements a Decision Tree Classifier from scratch in Python. It's based on the concepts from the Google Developers machine learning course and provides a hands-on approach to understanding the inner workings of decision trees.

## Features

- Custom Decision Tree implementation
- Entropy-based splitting criteria
- Information Gain calculation
- Tree visualization
- Sample data generation for testing

## Requirements

- Python 3.6+
- NumPy

## Installation

1. Clone this repository:
## Usage

The main script `ds.py` contains the `DecisionTree` class and a usage example. You can run it directly:
This will:
1. Generate sample data
2. Train a decision tree
3. Print the tree structure
4. Make predictions on test samples

To use the Decision Tree in your own projects:

```python
from ds import DecisionTree, generate_samples

# Generate or load your data
X, y = generate_samples(500)  # or load your own data

# Create and train the decision tree
dt = DecisionTree(max_depth=5, min_samples_split=10)
dt.fit(X, y)

# Make predictions
predictions = dt.predict(X_test)

# Visualize the tree
dt.print_tree()
```
