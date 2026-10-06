# PyTorch Basics

## Why this notebook matters
This notebook introduces the basic building blocks of PyTorch: tensors, operations, and the computational graph. These are the essential concepts behind all neural network work.

## Core idea
PyTorch tensors are like NumPy arrays but with automatic differentiation support. They allow efficient GPU computation and gradient tracking.

## Mathematical intuition
Tensor operations are simply linear algebra operations. When a tensor requires gradients, PyTorch tracks the operations used to create it.

This enables:

- automatic differentiation
- efficient backpropagation
- implementation of optimization loops

## Intuition
PyTorch makes the mathematical model executable. It gives a way to represent numbers, transformations, and trainable parameters in a framework that is both flexible and efficient.

## Practical workflow in the notebook
- create tensors
- do arithmetic and reshaping
- inspect gradients
- perform simple optimization

## Interview-ready explanation
"PyTorch basics cover the computational foundation of deep learning: tensors for data, operations for math, and autograd for differentiation. Without these, modern neural networks would be much harder to build and train."

## Common pitfalls
- confusing tensor shapes
- forgetting requires_grad
- not understanding autograd behavior on reused tensors

## Takeaway
If you master this notebook, the rest of deep learning becomes much easier to understand because the framework is no longer mysterious.
