# Deep Learning

This folder introduces the foundations of neural networks and optimization. It moves from simple trainable decision units to PyTorch-based training pipelines and fully functional neural models.

## Objective

The goal of this section is to understand how neural networks learn, how optimization works, and how to implement models using modern deep learning frameworks.

## Core topics

- activation functions
- perceptron learning
- backpropagation
- PyTorch tensors and autograd
- datasets and dataloaders
- neural network training loops
- ANN implementation and optimization

## Notebook index

- [3_activation_functions.ipynb](./3_activation_functions.ipynb)
- [ANN_Project_using_cpu_gpu.ipynb](./ANN_Project_using_cpu_gpu.ipynb)
- [BackPropagation_Practice.ipynb](./BackPropagation_Practice.ipynb)
- [DL_Assignment_01_Question.ipynb](./DL_Assignment_01_Question.ipynb)
- [DL_Assignment_02_Question.ipynb](./DL_Assignment_02_Question.ipynb)
- [DL_Mid_Term_Exam_Question.ipynb](./DL_Mid_Term_Exam_Question.ipynb)
- [Dataset_and_dataloader.ipynb](./Dataset_and_dataloader.ipynb)
- [NN_Module.ipynb](./NN_Module.ipynb)
- [Neural_Network_using_nn_module_pytorch.ipynb](./Neural_Network_using_nn_module_pytorch.ipynb)
- [Perceptron_from_scratch.ipynb](./Perceptron_from_scratch.ipynb)
- [Pytorch_basics.ipynb](./Pytorch_basics.ipynb)
- [pytorch_autograd.ipynb](./pytorch_autograd.ipynb)

---

## Concept

Deep learning represents a function as a composition of parameterized transformations. A layer computes a linear transformation, applies a nonlinearity, and the overall network learns by minimizing a loss function.

## Intuition

Neural networks are hierarchical feature extractors. Each layer reformats the input into a more useful representation until the final output is suitable for prediction.

## Math

The building block of deep learning is:

z = Wx + b

a = f(z)

where:
- x is the input vector
- W and b are learnable parameters
- f is a nonlinear activation function

Training uses gradient descent:

w <- w - η * ∂L/∂w

The chain rule allows the model to determine how each weight contributed to the final loss.

## Coding pattern

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

This is the standard deep learning training loop: forward pass -> compute loss -> backward pass -> update weights.

## Architectural design of a neural network

A simple neural network can be visualized as:

Input -> Linear layer -> Activation -> Hidden layer -> Activation -> Output layer

The design matters because:
- width determines representation capacity
- depth influences abstraction
- activation choice affects optimization behavior
- loss choice matches the task objective

## Common interview questions

- Why do we need nonlinearity in neural networks?
- What is the role of backpropagation?
- Why is ReLU so common in modern networks?
- How does autograd simplify training in PyTorch?
- What is the difference between a training loop and a model definition?
- Why are dataloaders used in deep learning?

## Interview cheat sheet

### Perceptron
- simplest trainable decision unit
- combines inputs with weights
- uses a threshold or activation to decide output

### Activation functions
- sigmoid: bounded output, saturation risk
- tanh: zero-centered, often stronger than sigmoid
- ReLU: simple and fast, widely used in deep models

### Backpropagation
- computes gradients with the chain rule
- enables learning for multiple layers
- essentially explains how the network changes its weights

### Optimizers
- SGD: basic and interpretable
- Adam: widely used adaptive optimizer
- learning rate controls update size

### Dataloaders
- batch training makes optimization stable
- shuffling reduces ordering bias
- efficiency improves with batching and parallel loading

## Recommended order

1. Learn PyTorch basics and autograd
2. Understand activation functions and perceptron
3. Practice backpropagation and training loops
4. Learn dataloaders and ANN structure
5. Complete project notebooks and explain the training flow
