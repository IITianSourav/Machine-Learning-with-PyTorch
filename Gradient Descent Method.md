## **Gradient Descent Methods in PyTorch**  

Gradient Descent is an optimization algorithm used to minimize a function by iteratively updating model parameters. In machine learning, it is used to minimize the loss function in training models. PyTorch provides built-in optimizers for different types of Gradient Descent. Below is a detailed explanation of **Stochastic Gradient Descent (SGD), Batch Gradient Descent, ADAM, and Adagrad**, along with PyTorch implementations.  

---

## **1. Stochastic Gradient Descent (SGD)**
### **Concept**
Stochastic Gradient Descent (SGD) updates model parameters using **one training example at a time** rather than using the entire dataset. This leads to **faster updates** but introduces more variance in parameter updates.  

### **Mathematical Formulation**
Given a loss function \( J(\theta) \), the update rule for SGD is:

$\theta := \theta - \alpha \nabla J(\theta; x_i, y_i)$

where:
- $\( \theta \)$ represents model parameters (weights and bias).
- $\( \alpha \)$ is the learning rate.
- $\( \nabla J(\theta; x_i, y_i) \)$ is the gradient of the loss function computed for a single training example $\( (x_i, y_i) \)$.

### **PyTorch Implementation**
```python
import torch

# Sample dataset
X_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
Y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)

# Initialize parameters
W = torch.randn((1, 1), requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Hyperparameters
learning_rate = 0.01
epochs = 100

# SGD optimizer
optimizer = torch.optim.SGD([W, b], lr=learning_rate)

# Training loop using SGD
for epoch in range(epochs):
    for i in range(len(X_train)):  # Updating parameters for each training example
        optimizer.zero_grad()
        Y_pred = X_train[i] @ W + b  # Prediction for one sample
        loss = (Y_pred - Y_train[i]) ** 2  # MSE loss
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print(f'Trained Weight: {W.item()}, Bias: {b.item()}')
```
### **Advantages**
- Faster parameter updates.
- More computationally efficient for large datasets.

### **Disadvantages**
- More variance in updates, leading to potential oscillations.
- May not always converge to the optimal solution.

---

## **2. Batch Gradient Descent**
### **Concept**
Batch Gradient Descent computes the **gradient over the entire dataset** before updating parameters. This results in **stable updates** but can be computationally expensive for large datasets.

### **Mathematical Formulation**

$\theta := \theta - \alpha \frac{1}{m} \sum_{i=1}^{m} \nabla J(\theta; x_i, y_i)$
where:
- $\( m \)$ is the total number of training samples.
- The gradient is computed over the entire dataset before updating parameters.

### **PyTorch Implementation**
```python
# Reinitialize parameters
W = torch.randn((1, 1), requires_grad=True)
b = torch.randn(1, requires_grad=True)

# SGD optimizer (Batch Gradient Descent)
optimizer = torch.optim.SGD([W, b], lr=learning_rate)

# Training loop using Batch Gradient Descent
for epoch in range(epochs):
    optimizer.zero_grad()
    Y_pred = X_train @ W + b  # Compute predictions for all samples
    loss = torch.mean((Y_pred - Y_train) ** 2)  # Compute MSE over all samples
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print(f'Trained Weight: {W.item()}, Bias: {b.item()}')
```
### **Advantages**
- Stable updates due to averaging gradients.
- Converges smoothly without oscillations.

### **Disadvantages**
- Computationally expensive for large datasets.
- Requires more memory.

---

## **3. Adaptive Moment Estimation (ADAM)**
### **Concept**
ADAM combines **Momentum** and **RMSprop** to adaptively adjust the learning rate for each parameter. It maintains two moving averages:
1. **First moment estimate (mean of gradients):** Helps in accelerating convergence.
2. **Second moment estimate (mean squared gradients):** Helps in stabilizing updates.

### **Mathematical Formulation**
$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$

$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$

$\theta := \theta - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}$

where:
- $\( \beta_1 \)$ and $\( \beta_2 \)$ are decay rates.
- $\( g_t \)$ is the gradient at time $\( t \)$.

### **PyTorch Implementation**
```python
# Reinitialize parameters
W = torch.randn((1, 1), requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Adam optimizer
optimizer = torch.optim.Adam([W, b], lr=learning_rate)

# Training loop using ADAM
for epoch in range(epochs):
    optimizer.zero_grad()
    Y_pred = X_train @ W + b
    loss = torch.mean((Y_pred - Y_train) ** 2)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print(f'Trained Weight: {W.item()}, Bias: {b.item()}')
```
### **Advantages**
- Efficient and adaptive learning rates.
- Works well with sparse data.
- Faster convergence.

### **Disadvantages**
- May not generalize well.
- Requires tuning of $\( \beta_1, \beta_2 \)$ parameters.

---

## **4. Adagrad (Adaptive Gradient Algorithm)**
### **Concept**
Adagrad adapts the learning rate for each parameter based on **past gradients**. It gives **smaller updates** to frequent features and **larger updates** to rare features.

### **Mathematical Formulation**

$\theta := \theta - \frac{\alpha}{\sqrt{G_t + \epsilon}} g_t$

where:
- $\( G_t \)$ accumulates squared gradients.
- $\( \alpha \)$ is the learning rate.

### **PyTorch Implementation**
```python
# Reinitialize parameters
W = torch.randn((1, 1), requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Adagrad optimizer
optimizer = torch.optim.Adagrad([W, b], lr=learning_rate)

# Training loop using Adagrad
for epoch in range(epochs):
    optimizer.zero_grad()
    Y_pred = X_train @ W + b
    loss = torch.mean((Y_pred - Y_train) ** 2)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print(f'Trained Weight: {W.item()}, Bias: {b.item()}')
```
### **Advantages**
- No need to manually tune learning rates.
- Good for sparse data.

### **Disadvantages**
- Learning rate decays aggressively.
- May stop learning prematurely.

---

### **Conclusion**
- **SGD**: Faster updates, but high variance.
- **Batch Gradient Descent**: Stable but slow.
- **Adam**: Efficient, adaptive learning rates.
- **Adagrad**: Good for sparse data but aggressive learning rate decay.

Would you like a comparison table summarizing these methods?
