# Deep Learning

This section introduces the neural network foundations that power modern AI. Deep learning extends traditional machine learning by using layered functions that transform data into richer internal representations.

## Objective

The deep learning section focuses on:

- perceptrons
- activation functions
- neural networks
- backpropagation
- PyTorch basics
- optimization and training loops

## Core concepts

### 1. Concept
Deep learning is the study of layered, parameterized functions that learn by minimizing a loss function.

### 2. Intuition
A neural network transforms input data through successive layers. Each layer extracts or recombines features so that the final output is better aligned with the task objective.

### 3. Math
The key relation is:

z = W x + b

a = f(z)

where:
- x is input
- W and b are learnable parameters
- f is an activation function

Training updates parameters with:

w <- w - η * ∂L/∂w

### 4. Architecture of a neural network

Input -> Linear layer -> Activation -> Hidden layer -> Activation -> Output

### 5. Code pattern

```python
import torch
import torch.nn as nn

x = torch.randn(64, 10)
y = torch.randn(64, 1)

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 6. Interview-ready explanation

"Deep learning models learn hierarchical representations by composing linear transformations and nonlinear activations. The learning process is driven by a loss function and backpropagation, which computes gradients to update weights in the direction that reduces error."

### 7. Common pitfalls

- forgetting to zero gradients before backward pass
- using the wrong activation function for the layer
- poor learning rate choice
- ignoring data shape compatibility

### 8. Takeaway

The core deep learning idea is simple: build a parameterized function, compute error, and adjust parameters using gradients until the model performs well.

---

## Important topics

- activation functions: ReLU, sigmoid, tanh
- perceptron from scratch
- backpropagation
- PyTorch basics and autograd
- datasets and dataloaders
- ANN training workflow

## Interview questions

- Why do we need activation functions?
- Why is ReLU widely used?
- What is backpropagation?
- What does autograd do in PyTorch?
- Why do dataloaders matter?

## Recommended study order

1. Learn tensors and autograd
2. Learn activation functions and perceptron
3. Study backpropagation
4. Train a simple ANN
5. Practice with project notebooks
