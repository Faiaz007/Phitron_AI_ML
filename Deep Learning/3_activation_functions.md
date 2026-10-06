# 3 Activation Functions

## Why this notebook matters
Activation functions decide whether a neuron should fire. Without them, a neural network would behave like a large linear model and would fail to model nonlinear patterns such as image boundaries, text relationships, or decision surfaces.

## The core idea
Each neuron computes a weighted sum of its inputs and then applies a nonlinear activation function.

z = W x + b

a = f(z)

The nonlinearity is the key: without it, stacking layers would not create richer representations.

## Mathematical intuition
The activation function shapes the output of a neuron and affects learning dynamics.

### Sigmoid
f(z) = 1 / (1 + e^{-z})
- outputs in (0,1)
- good for probabilistic interpretation
- suffers from saturation and vanishing gradients

### Tanh
f(z) = (e^z - e^{-z}) / (e^z + e^{-z})
- outputs in (-1,1)
- zero-centered, often better than sigmoid

### ReLU
f(z) = max(0, z)
- simple and efficient
- works well in deep networks
- can cause dead neurons if weights become too negative

## Intuition
A neuron is like a gate. Activation functions decide how much signal gets passed forward. Some functions behave like soft switches, while others are more aggressive.

## Practical workflow in the notebook
- compare function plots
- observe their behavior on negative and positive inputs
- analyze gradient flow
- discuss which one is suitable for which architecture

## Interview-ready explanation
"Activation functions introduce nonlinearity so deep networks can model complex patterns. They determine signal strength, influence gradients during backpropagation, and impact convergence speed and stability."

## Common pitfalls
- using sigmoid in deep layers when ReLU is more suitable
- forgetting that derivatives matter during optimization
- assuming all activations behave similarly under backpropagation

## Takeaway
Choosing the right activation function is a foundational design decision in deep learning. It affects representational power, optimization stability, and final model performance.
