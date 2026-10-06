# NN Module

## Why this notebook matters
The PyTorch nn module abstracts away repetitive code for layers, activations, and optimization. It makes neural networks easier to build and reason about.

## Core idea
Instead of manually writing matrix operations for every layer, users can define a model as a sequence of modules such as Linear, ReLU, and Sequential.

## Mathematical intuition
A layer in nn.Module is a differentiable function parameterized by weights and biases. The model is just a composition of functions:

f(x) = f_L(f_{L-1}(...f_1(x)))

The training loop still uses the same principles: forward pass, loss, backpropagation, update.

## Intuition
This notebook captures the idea that deep learning frameworks give us reusable building blocks. The complexity is hidden behind clean abstractions, but the mathematics is unchanged.

## Practical workflow in the notebook
- define layers using nn.Linear
- stack layers using nn.Sequential
- compute output
- train with optimizer
- inspect learned parameters

## Interview-ready explanation
"nn.Module introduces a modular architecture for building neural networks. It encapsulates parameters, forward logic, and optimization hooks while preserving the same mathematical foundations of differentiable computation."

## Common pitfalls
- forgetting to zero gradients before backward pass
- using wrong optimizer settings
- not understanding what parameters are being learned

## Takeaway
This notebook is about moving from manual math to practical deep learning engineering. It is a major transition point in learning PyTorch.
