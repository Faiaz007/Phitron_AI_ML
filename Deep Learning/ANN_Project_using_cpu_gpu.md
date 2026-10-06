# ANN Project using CPU GPU

## Why this notebook matters
This project brings together an end-to-end neural network workflow and shows how the same code can run across CPU and GPU environments. In practice, hardware choice affects throughput and training time.

## Core idea
An artificial neural network learns a mapping from input features to outputs. The network performs forward propagation, computes loss, and then updates weights through backpropagation.

## Mathematical intuition
For an ANN with layers l = 1..L, the forward pass is:

z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
a^{(l)} = f(z^{(l)})

Backpropagation uses the chain rule to compute gradients:

\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(l)}} a^{(l-1)T}

This tells us how much each weight contributed to the loss and how to correct it.

## Intuition
A deep network is essentially a sequence of transformations. Early layers learn low-level patterns, while deeper layers combine them into more abstract representations.

## Practical workflow in the notebook
- build data pipeline
- define network architecture
- train on CPU/GPU
- compare speed and stability
- evaluate metrics

## Interview-ready explanation
"A neural network learns hierarchical representations by stacking affine transformations and nonlinear activations. GPU acceleration helps because matrix operations are highly parallelizable, making training much faster for large models."

## Common pitfalls
- not checking device availability
- using too large batches for memory limitations
- misinterpreting training speed differences

## Takeaway
This notebook connects theory to real training behavior. It highlights that understanding architecture, optimization, and hardware matters as much as code correctness.
