# Perceptron from Scratch

## Why this notebook matters
The perceptron is the simplest trainable neural unit. It is the foundation of classification and a great way to understand weights, decision boundaries, and learning.

## Core idea
A perceptron computes a weighted sum and applies a threshold or step activation. It learns by updating its weights when it makes a mistake.

## Mathematical intuition
Given inputs x and weights w, the output is:

y = sign(w^T x + b)

The update rule is:

w <- w + \eta (y_true - y_pred) x

This is one of the earliest examples of online learning.

## Intuition
A perceptron draws a hyperplane that separates two classes. If the data are linearly separable, the algorithm can learn a decision boundary.

## Practical workflow in the notebook
- initialize weights
- predict class for each sample
- compare with true labels
- update weights when errors occur
- visualize decision boundary

## Interview-ready explanation
"The perceptron is a linear classifier that updates its weights based on mistakes. It demonstrates the fundamental learning loop: predict -> compare -> adjust parameters."

## Common pitfalls
- assuming perceptrons can solve every classification problem
- ignoring linearly separable assumptions
- not understanding threshold behavior

## Takeaway
This notebook is essential because it teaches the first learning algorithm in a way that is mathematically simple but conceptually deep.
