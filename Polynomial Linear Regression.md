# Polynomial Regression Using PyTorch

## Introduction
Polynomial Regression is an extension of **Linear Regression** that models non-linear relationships between input features and the target variable. Instead of fitting a straight line, it fits a polynomial curve to the data.

## Mathematical Formulation
A polynomial regression model of degree $\( d \)$ is represented as:

$y = w_0 + w_1 x + w_2 x^2 + ... + w_d x^d$

where:
- $\( y \)$ is the target variable.
- $\( x \)$ is the input feature.
- $\( w_0, w_1, ..., w_d \)$ are the coefficients (weights) to be learned.

To apply **Linear Regression** to this problem, we transform the input feature \( x \) into **polynomial features**:

$X_{poly} = [x, x^2, x^3, ..., x^d]$

## Real-Life Example: House Price Prediction
A common real-world application of Polynomial Regression is predicting house prices based on size.
- A **linear model** might not capture the real-world trend since house prices often increase at an **accelerating rate** as the size increases.
- A **polynomial model** (quadratic or cubic) provides a better fit by capturing non-linear price growth.

For instance, a house price model can be represented as:


$\text{Price} = w_0 + w_1 \times \text{Size} + w_2 \times \text{Size}^2 + w_3 \times \text{Size}^3$

This approach helps capture **diminishing returns** and **accelerating costs** in real estate markets.

---

## Implementation Using PyTorch

### Step 1: Import Required Libraries
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
```

### Step 2: Generate Synthetic Data
We generate a dataset following a **quadratic equation** with some noise.
```python
# Set random seed for reproducibility
np.random.seed(42)

# Generate input data (100 samples)
X = np.random.rand(100, 1).astype(np.float32) * 10  # Random values between 0 and 10

# Generate quadratic relationship with noise
y = 2 * X**2 + 3 * X + 5 + np.random.randn(100, 1).astype(np.float32) * 3

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)
```

### Step 3: Create Polynomial Features
```python
def polynomial_features(X, degree):
    """Transforms input data X into polynomial features of a given degree."""
    poly_X = torch.cat([X**i for i in range(1, degree + 1)], dim=1)
    return poly_X

# Define polynomial degree
degree = 2
X_poly = polynomial_features(X_tensor, degree)
```

### Step 4: Define the Polynomial Regression Model
```python
class PolynomialRegression(nn.Module):
    def __init__(self, degree):
        super(PolynomialRegression, self).__init__()
        self.linear = nn.Linear(degree, 1)  # 'degree' input features, 1 output

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = PolynomialRegression(degree)
```

### Step 5: Define Loss Function and Optimizers
```python
# Mean Squared Error (MSE) Loss
criterion = nn.MSELoss()

# Different optimizers to compare
optimizers = {
    "SGD": optim.SGD(model.parameters(), lr=0.001),
    "Adam": optim.Adam(model.parameters(), lr=0.01),
    "RMSprop": optim.RMSprop(model.parameters(), lr=0.01)
}
```

### Step 6: Train the Model Using Different Optimizers
```python
def train_model(optimizer_name, model, optimizer, num_epochs=1000):
    """Trains the model using a given optimizer."""
    losses = []
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(X_poly)
        loss = criterion(y_pred, y_tensor)
        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 200 epochs
        if (epoch + 1) % 200 == 0:
            print(f'[{optimizer_name}] Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return losses

# Train with each optimizer
losses_dict = {}
for opt_name, opt in optimizers.items():
    print(f"Training with {opt_name}...")
    model = PolynomialRegression(degree)  # Reset model
    optimizer = optimizers[opt_name]
    losses_dict[opt_name] = train_model(opt_name, model, optimizer)
```

### Step 7: Visualize the Results
```python
# Generate smooth X values for visualization
X_test = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
X_test_tensor = torch.from_numpy(X_test)
X_test_poly = polynomial_features(X_test_tensor, degree)

# Predict using trained model
y_test_pred = model(X_test_poly).detach().numpy()

# Plot original data and the learned curve
plt.scatter(X, y, label="Original Data")
plt.plot(X_test, y_test_pred, color='red', label="Fitted Polynomial Curve")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Polynomial Regression using PyTorch")
plt.show()
```

### Step 8: Make Predictions
```python
# Predict for a new value (e.g., X=7)
X_new = torch.tensor([[7.0]])
X_new_poly = polynomial_features(X_new, degree)

y_new_pred = model(X_new_poly).item()
print(f"Prediction for X=7: {y_new_pred:.2f}")
```

---

## Conclusion
1. **Polynomial Regression** helps capture **non-linear** relationships between variables.
2. **Higher-degree polynomials** can model more complex data but may lead to **overfitting**.
3. **Adam and RMSprop** optimizers often converge **faster** and give **lower loss** compared to **SGD**.
4. **Feature scaling** can improve training stability, especially for **higher-degree polynomials**.

This implementation provides a robust **framework for polynomial regression** using PyTorch, which can be extended for real-world applications such as **house price prediction, stock price forecasting, and weather modeling**.

For further improvements, consider:
- Using **cross-validation** to find the best polynomial degree.
- Experimenting with **regularization techniques** like **Lasso or Ridge Regression** to prevent overfitting.
- Extending this model to **multivariate polynomial regression**.

