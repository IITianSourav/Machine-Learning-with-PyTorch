# Multiple Linear Regression Using Linear Algebra

## 1. Introduction
Multiple Linear Regression (MLR) is a supervised learning algorithm used to model the relationship between one dependent variable and multiple independent variables. It is an extension of Simple Linear Regression when there are multiple predictors.

Mathematically, the MLR model is represented as:

$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n + \epsilon$

where:
- $\( Y \)$ is the dependent variable,
- $\( X_1, X_2, ..., X_n \)$ are independent variables,
- $\( \beta_0 \)$ is the intercept,
- $\( \beta_1, \beta_2, ..., \beta_n \)$ are coefficients,
- $\( \epsilon \)$ is the error term.

Using matrix notation, the equation can be rewritten as:

$\mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}$

where:
- $\( \mathbf{Y} \)$ is an $\( m \times 1 \)$ column vector of observed values,
- $\( \mathbf{X} \)$ is an $\( m \times (n+1) \)$ matrix including a column of ones for the intercept,
- $\( \boldsymbol{\beta} \)$ is an $\( (n+1) \times 1 \)$ column vector of coefficients,
- $\( \boldsymbol{\epsilon} \)$ is an $\( m \times 1 \)$ error term vector.

## 2. Estimating Coefficients Using Normal Equation
The optimal parameters $\( \boldsymbol{\beta} \)$ can be estimated using the **least squares** method, which minimizes the residual sum of squares:

$\min || \mathbf{Y} - \mathbf{X} \boldsymbol{\beta} ||^2$

The solution is obtained using the **Normal Equation**:

$\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}$

where:
- $\( \mathbf{X}^T \)$ is the transpose of $\( \mathbf{X} \)$,
- $\( (\mathbf{X}^T \mathbf{X})^{-1} \)$ is the inverse of $\( \mathbf{X}^T \mathbf{X} \)$,
- $\( \mathbf{X}^T \mathbf{Y} \) is the projection of \( \mathbf{Y} \) onto \( \mathbf{X} \)$.

## 3. Estimating Coefficients Using Gradient Descent
If $\( \mathbf{X}^T \mathbf{X} \)$ is non-invertible or computationally expensive to invert, **Gradient Descent** is used to find $\( \boldsymbol{\beta} \)$.

### Gradient Descent Algorithm
1. Initialize $\( \boldsymbol{\beta} \)$ with random values.
2. Update $\( \boldsymbol{\beta} \)$ iteratively using:
   $\boldsymbol{\beta} := \boldsymbol{\beta} - \alpha \frac{1}{m} \mathbf{X}^T (\mathbf{X} \boldsymbol{\beta} - \mathbf{Y})$
   where $\( \alpha \)$ is the learning rate.
3. Repeat until convergence (when the cost function stops decreasing significantly).

## 4. Model Evaluation
To measure the performance of the model, we use:

1. **Mean Squared Error (MSE)**:
   $MSE = \frac{1}{m} \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2$
2. **Coefficient of Determination (R-squared)**:
   $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$
   where:
   - $SS_{res} = \sum (Y_i - \hat{Y}_i)^2 \)$ is the residual sum of squares,
   - $\( SS_{tot} = \sum (Y_i - \bar{Y})^2 \)$ is the total sum of squares.

## 5. Conclusion
Multiple Linear Regression is a fundamental statistical method for modeling relationships between variables. The Normal Equation provides an analytical solution, while Gradient Descent is useful for large datasets. Understanding the underlying Linear Algebra concepts helps in efficient implementation and optimization of the model.

# Multiple Linear Regression using PyTorch

This code demonstrates how to implement Multiple Linear Regression using PyTorch.

```python
import torch

# Sample dataset (3 features)
X_train = torch.tensor([[1.0, 2.0, 3.0], 
                        [4.0, 5.0, 6.0], 
                        [7.0, 8.0, 9.0], 
                        [10.0, 11.0, 12.0]], dtype=torch.float32)

Y_train = torch.tensor([[6.0], [15.0], [24.0], [33.0]], dtype=torch.float32)  # Target values

# Number of samples (m) and features (n)
m, n = X_train.shape

# Initialize weights and bias
W = torch.randn((n, 1), requires_grad=True)  # Weight matrix (n x 1)
b = torch.randn(1, requires_grad=True)       # Bias term

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD([W, b], lr=learning_rate)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Predicted values
    Y_pred = X_train @ W + b
    
    # Compute Mean Squared Error (MSE)
    loss = torch.mean((Y_pred - Y_train) ** 2)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')

# Final trained parameters
print("Trained weights:", W.detach().numpy())
print("Trained bias:", b.item())
