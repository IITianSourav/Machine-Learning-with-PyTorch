# Theory of Linear Regression

Linear Regression is a supervised learning algorithm used to model the relationship between an independent variable (features) and a dependent variable (target/output), which must be a continuous variable. The goal is to fit a line of the form:

$y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n$ where ${x_1, x_2, ..., x_n}$ represent the input features and $y$ is the continuous variable to predict.

## Types of Linear Regression

There are two types of Linear Regression in general:

1. **Simple Linear Regression** (single input feature):
   
   $y = b_0 + b_1x_1$

2. **Multiple Linear Regression** (multiple input features, more than one):
   
   $y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n$

   Simple and multiple linear regression both predict continuous values based on input features, provided the input features are linearly distributed. If the data is not linearly distributed, **Polynomial Linear Regression** can be applied.

3. **Polynomial Linear Regression** (discussed in a later section).

## Mathematical Formulation

For a given dataset with n observations:

$y = wX + b$

where:
- $X$ = Input features
- $y$ = Target output
- $w$ = Weights (slope of the line)
- $b$ = Bias (y-intercept)

### Loss Function (Mean Squared Error - MSE)

The MSE loss is used to measure how far the predictions are from the actual values:

$L = \frac{1}{n} \sum (y_i - \hat{y_i})^2$

where $y_i$ is the actual value and $\hat{y_i}$ is the predicted value.

### Gradient Descent for Optimization

We update w and b using gradient descent:

$w = w - \alpha * (\partial L/\partial w)$

$b = b - \alpha * (\partial L/\partial b)$

where $\alpha$ is the learning rate.

---

# PyTorch Implementation

## Import Required Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
```

## Generate Sample Data

Generating a synthetic dataset where y = 2x + 3 with some noise.

```python
np.random.seed(42)
X = np.random.rand(100, 1).astype(np.float32) * 10  # Features (100 samples)
y = 2 * X + 3 + np.random.randn(100, 1).astype(np.float32)  # True relation + noise

# Convert NumPy arrays to PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)
```

## Define the Linear Regression Model

```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)  # One input, one output
    
    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = LinearRegressionModel()
```

## Define Loss Function and Optimizer

```python
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

## Train the Model

```python
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass: Compute predicted y
    y_pred = model(X_tensor)

    # Compute the loss
    loss = criterion(y_pred, y_tensor)

    # Backward pass: Compute gradients
    optimizer.zero_grad()  # Reset gradients to zero
    loss.backward()  # Compute new gradients

    # Update model parameters
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

## Plot the Results

```python
[w, b] = model.linear.parameters()
w = w.item()
b = b.item()
print(f"Learned Parameters: w = {w:.2f}, b = {b:.2f}")

plt.scatter(X, y, label="Original Data")
plt.plot(X, w * X + b, color='red', label="Fitted Line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression using PyTorch")
plt.show()
```

## Making Predictions

```python
X_new = torch.tensor([[7.0]])
y_pred = model(X_new).item()
print(f"Prediction for X=7: {y_pred:.2f}")
```

---

# Summary of Implementation

- Generated synthetic data for a simple linear relationship.
- Defined a Linear Regression model using `nn.Linear`.
- Used MSE Loss and SGD optimizer to minimize loss.
- Trained the model using forward and backward passes.
- Visualized the learned regression line.
- Made predictions using the trained model.

---

# Key Takeaways

- PyTorch provides an easy way to define and train linear models using `nn.Linear`.
- Autograd handles differentiation automatically for optimization.
- Loss functions and optimizers play a key role in model training.
- Training involves multiple epochs of computing loss, gradients, and updating weights.
- PyTorch makes it simple to visualize results and make predictions.

---

This implementation can be extended to multiple linear regression or optimized using GPU acceleration.

