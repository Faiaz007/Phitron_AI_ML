# DL Assignment 01 Question

## Why this notebook matters
Assignments are often designed to combine the most important concepts in a compact way: forward pass, loss, differentiation, and network intuition. This notebook is a checkpoint for understanding core deep learning fundamentals.

## Core idea
The assignment likely tests whether the student can reason from first principles: how data flows through a network, how loss changes with parameters, and how training updates the model.

## Mathematical intuition
A neural network is a sequence of nested transforms:

a^{(1)} = f(W^{(1)}x + b^{(1)})
a^{(2)} = f(W^{(2)}a^{(1)} + b^{(2)})
...

The objective is to minimize a loss such as MSE or cross-entropy. The optimization uses gradient descent.

## Intuition
Assignments force you to connect concepts rather than just memorize formulas. This is where the theory becomes operational.

## Practical workflow in the notebook
- read the problem carefully
- identify the expected architecture
- derive the update rule
- implement the solution
- validate output numerically

## Interview-ready explanation
"A strong modeler should be able to explain why a network learns, what the loss is measuring, and how gradients flow through parameterized layers."

## Common pitfalls
- reading the problem too superficially
- ignoring dimension matching
- not checking whether the model is learning or just memorizing

## Takeaway
Assignments are miniature experiments in understanding. They strengthen the bridge between theory and implementation.
