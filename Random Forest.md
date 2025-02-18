# Random Forest Implementation Using PyTorch

## Introduction
Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the majority vote for classification or the average prediction for regression tasks. It enhances predictive performance and reduces overfitting compared to individual decision trees.

## Random Forest Concept
A Random Forest consists of multiple decision trees trained on different subsets of the dataset. The final prediction is obtained by aggregating the results from all trees. It introduces randomness through:

1. **Bootstrap Sampling**: Each tree is trained on a random subset of the data.
2. **Feature Subsampling**: At each split, a random subset of features is considered.

### Mathematical Formulation
For classification, let each tree provide an output $h_i(x)$ for an input $x$. The final prediction is given by majority voting:

$$
H(x) = \text{argmax}_{y} \sum_{i=1}^{n} I(h_i(x) = y)
$$

where $I$ is the indicator function, and $n$ is the number of trees.

For regression, the final output is the average of all tree outputs:

$$
H(x) = \frac{1}{n} \sum_{i=1}^{n} h_i(x)
$$

## Implementation Using PyTorch
PyTorch does not provide a built-in Random Forest implementation, so we manually construct an ensemble of decision trees.

### Steps:
1. **Prepare the dataset**
2. **Define a single decision tree model**
3. **Train multiple decision trees on different subsets**
4. **Aggregate predictions**

### Code Implementation

```python
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest parameters
num_trees = 10
sample_size = int(0.8 * len(X_train))
feature_size = int(np.sqrt(X_train.shape[1]))

# Train multiple decision trees
forest = []
for _ in range(num_trees):
    sample_indices = np.random.choice(len(X_train), sample_size, replace=True)
    feature_indices = np.random.choice(X_train.shape[1], feature_size, replace=False)
    X_sample, y_sample = X_train[sample_indices][:, feature_indices], y_train[sample_indices]
    
    tree = DecisionTreeClassifier()
    tree.fit(X_sample, y_sample)
    forest.append((tree, feature_indices))

# Prediction function
def predict_forest(X):
    predictions = np.array([
        tree.predict(X[:, features]) for tree, features in forest
    ])
    return np.round(np.mean(predictions, axis=0)).astype(int)

# Evaluate model
y_pred = predict_forest(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy:.4f}')
```

## Explanation of the Code

### Dataset Preparation
- A synthetic classification dataset is generated using `make_classification`.
- It is split into training and test sets.

### Creating Multiple Decision Trees
- A loop constructs multiple decision trees.
- Bootstrap sampling selects random training instances.
- Feature subsampling selects a subset of features for each tree.
- Each tree is trained on its respective subset.

### Making Predictions
- Each tree predicts on test data using only the features it was trained on.
- Predictions are aggregated by averaging for classification.
- The final classification decision is obtained by rounding the mean.

## Advantages of Random Forest

1. **Reduces Overfitting**: Multiple trees generalize better than a single tree.
2. **Handles High-Dimensional Data**: Works well with a large number of features.
3. **Robust to Noise**: Bootstrapping helps mitigate the impact of noisy data.
4. **Feature Importance Estimation**: Can rank feature importance.

## Limitations

1. **Computationally Expensive**: Training multiple trees is resource-intensive.
2. **Less Interpretable**: Harder to understand than a single decision tree.

## Conclusion
Random Forest is a powerful ensemble learning method that improves the predictive performance of decision trees. Implementing it in PyTorch requires manual construction of decision trees and aggregating their predictions. This approach can be extended to deep learning frameworks by integrating neural network-based tree structures.

