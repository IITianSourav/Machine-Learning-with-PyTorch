# Logistic Regression with PyTorch

## Introduction
This document explains Logistic Regression using PyTorch for both Binary and Multinomial Classification. It includes theoretical explanations and practical examples.

## Theory
### What is Logistic Regression?
Logistic Regression is a statistical method used for binary and multiclass classification tasks. Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts probabilities using a logistic (sigmoid) or softmax function.

### How It Works
- It applies a linear transformation to input features:  
  $z = w^T x + b$
- A nonlinear activation function (Sigmoid for binary and Softmax for multinomial classification) maps the result to probabilities.
- The model is trained using a loss function that minimizes the difference between predicted and actual labels.

### Mathematical Formulation
For binary classification, the probability of class 1 is given by the sigmoid function:
$P(y=1|x) = \frac{1}{1 + e^{-z}}$
where $z = w^T x + b$.

For multinomial classification with $K$ classes, we use the softmax function:
$P(y=k|x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$

### Loss Function
- **Binary Classification**: Binary Cross Entropy Loss (`BCELoss`) is used, defined as:
  $L = -\frac{1}{N} \sum_{i=1}^{N} \big[y_i \log(\hat{y_i}) + (1 - y_i) \log(1 - \hat{y_i}) \big]$
- **Multinomial Classification**: Cross Entropy Loss (`CrossEntropyLoss`) is used, which generalizes binary cross-entropy for multiple classes.

## Implementation
We implement Logistic Regression using PyTorch and train it using synthetic data generated from scikit-learn.

### Steps
1. **Define the Logistic Regression Model**
2. **Train the Model using an Optimizer**
3. **Evaluate Performance using Accuracy**

```python
# Import Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Define Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)  # No activation here, handled in loss function

# Function to Train the Model
def train_model(model, criterion, optimizer, train_loader, epochs=100):
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Function to Evaluate the Model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    print(f'Accuracy: {100 * correct / total:.2f}%')

# Binary Classification Example
def binary_classification():
    print("\n## Binary Classification Example")
    X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
    
    model = LogisticRegression(input_dim=5, output_dim=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    train_model(model, criterion, optimizer, train_loader)
    evaluate_model(model, test_loader)

# Multinomial Classification Example
def multinomial_classification():
    print("\n## Multinomial Classification Example")
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=3, n_informative=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
    
    model = LogisticRegression(input_dim=5, output_dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    train_model(model, criterion, optimizer, train_loader)
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    binary_classification()
    multinomial_classification()
```

## Conclusion
This document provided an overview of Logistic Regression using PyTorch, with implementations for both binary and multinomial classification. The examples demonstrate how to train and evaluate models effectively using PyTorch.
