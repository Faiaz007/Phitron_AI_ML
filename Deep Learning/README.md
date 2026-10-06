# Deep Learning

This folder covers the neural-network foundations of modern AI. It is designed to build intuition from the simplest trainable unit (the perceptron) to full PyTorch-based models and optimization loops.

## Core focus

- activation functions
- perceptron learning
- artificial neural networks
- backpropagation
- PyTorch basics and autograd
- dataset and dataloader workflow
- model training on real optimization loops

## Table of contents

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

## Learning path

### 1. Concept
Deep learning is the study of layered, parameterized functions trained by minimizing a loss. The model learns using gradient descent and backpropagation.

### 2. Intuition
A neural network transforms raw input into more useful internal representations. Early layers detect simple structure; deeper layers combine them into complex abstractions.

### 3. Math
Key ideas include:

- linear transformation: z = Wx + b
- activation: a = f(z)
- loss minimization
- chain rule for gradients
- gradient descent update: w <- w - eta * grad

### 4. Coding pattern
Typical PyTorch training loop:

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

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 5. Common interview questions
- Why are activation functions necessary?
- What is the difference between sigmoid and ReLU?
- Why does backpropagation work mathematically?
- What does autograd do in PyTorch?
- Why is gradient descent important in neural networks?
- What is the role of a dataloader in training?

## Interview cheat sheet

### Perceptron
- simplest trainable unit
- linear decision boundary
- updates weights with mistakes

### Activation functions
- sigmoid: smooth output, saturates
- tanh: centered, often better than sigmoid
- ReLU: efficient and widely used

### Backpropagation
- uses the chain rule
- computes gradients for every weight
- enables learning on multi-layer networks

### PyTorch basics
- tensor is the core data structure
- autograd computes gradients automatically
- optimizer updates parameters based on gradients

### Datasets and dataloaders
- prepare data efficiently
- handle batching, shuffling, and scaling
- improve memory use and training stability

## Recommended order

1. Start with PyTorch basics and autograd
2. Study activation functions and perceptron
3. Learn backpropagation and training loop
4. Practice dataset/dataloader workflow
5. Move into ANN and project notebooks
