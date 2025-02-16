A **Decision Tree** is a non-parametric supervised learning algorithm used for classification and regression tasks. It splits data into branches at decision nodes based on feature values, creating a tree-like structure. In this explanation, we will implement a Decision Tree from scratch using PyTorch, covering its mathematical foundation.

---

## **1. Decision Tree: Mathematical Foundation**

A Decision Tree partitions data based on feature values using recursive binary splitting. The main goal is to minimize impurity at each split.

### **1.1 Impurity Measures**
Two common impurity measures are:

1. **Gini Impurity** (for classification):
   $G = 1 - \sum_{i=1}^{C} p_i^2$
   where $p_i$ is the proportion of samples belonging to class $i$.

2. **Entropy** (for classification):
   $H = -\sum_{i=1}^{C} p_i \log_2 p_i$
   where $p_i$ is the probability of a class.

3. **Mean Squared Error (MSE)** (for regression):
   $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$
   where $y_i$ are the target values and $\bar{y}$ is the mean.

The best split is chosen by minimizing the weighted impurity of child nodes.

### **1.2 Information Gain**
The **Information Gain (IG)** quantifies the reduction in impurity after splitting:

$IG = I_{parent} - \left( \frac{N_L}{N} I_{left} + \frac{N_R}{N} I_{right} \right)$

where:
- $I_{parent}$ is the impurity before the split.
- $I_{left}$ and $I_{right}$ are the impurities of the left and right child nodes.
- $N_L$, $N_R$ are the number of samples in left and right nodes.
- $N$ is the total number of samples.

---

## **2. Implementing Decision Tree using PyTorch**
We will implement a **binary classification** Decision Tree using **PyTorch tensors**.

### **2.1 Building the Decision Tree from Scratch**
```python
import torch

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    def _gini(self, y):
        classes, counts = torch.unique(y, return_counts=True)
        probs = counts.float() / y.size(0)
        return 1 - torch.sum(probs ** 2)

    def _entropy(self, y):
        classes, counts = torch.unique(y, return_counts=True)
        probs = counts.float() / y.size(0)
        return -torch.sum(probs * torch.log2(probs + 1e-9))  # Adding epsilon to avoid log(0)

    def _best_split(self, X, y):
        best_gain = 0
        best_feature, best_threshold = None, None
        parent_impurity = self._gini(y) if self.criterion == "gini" else self._entropy(y)
        
        for feature in range(X.shape[1]):
            thresholds = torch.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_impurity = self._gini(y[left_mask]) if self.criterion == "gini" else self._entropy(y[left_mask])
                right_impurity = self._gini(y[right_mask]) if self.criterion == "gini" else self._entropy(y[right_mask])

                n, nL, nR = y.size(0), left_mask.sum().item(), right_mask.sum().item()
                weighted_impurity = (nL / n) * left_impurity + (nR / n) * right_impurity
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature, best_threshold = feature, threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or y.size(0) < self.min_samples_split or len(torch.unique(y)) == 1:
            return torch.mode(y).values.item()

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return torch.mode(y).values.item()

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            "feature": best_feature,
            "threshold": best_threshold.item(),
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, sample, node):
        if isinstance(node, dict):
            if sample[node["feature"]] <= node["threshold"]:
                return self._predict_sample(sample, node["left"])
            else:
                return self._predict_sample(sample, node["right"])
        else:
            return node

    def predict(self, X):
        return torch.tensor([self._predict_sample(sample, self.tree) for sample in X])
```

---

## **3. Example Usage**
We will train our decision tree on a simple dataset.

```python
# Creating synthetic data
X = torch.tensor([[2.5], [1.5], [3.5], [5.0], [1.0], [4.5], [6.0]])
y = torch.tensor([0, 0, 1, 1, 0, 1, 1])

# Initialize and train the decision tree
tree = DecisionTree(max_depth=3, criterion="gini")
tree.fit(X, y)

# Test data
X_test = torch.tensor([[1.2], [4.0], [5.5]])

# Predict class labels
predictions = tree.predict(X_test)
print(predictions)
```

---

## **4. Explanation of Code**
1. **_gini() and _entropy()**: Compute impurity.
2. **_best_split()**: Iterates through all features and thresholds to find the best split.
3. **_build_tree()**: Recursively builds the decision tree.
4. **fit()**: Calls `_build_tree()` to construct the tree.
5. **predict()**: Traverses the tree for each sample and returns predicted labels.

---

## **5. Summary**
- We built a **Decision Tree Classifier** from scratch using PyTorch tensors.
- The implementation uses **Gini Impurity** and **Entropy** for classification.
- The **best split** is determined using **Information Gain**.
- The decision tree is **recursively built** up to `max_depth`.

This approach provides fundamental insights into **tree-based learning** while leveraging PyTorch for tensor operations.
