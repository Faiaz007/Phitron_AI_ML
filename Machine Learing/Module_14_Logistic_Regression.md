# Module 14: Logistic Regression

## Why this notebook matters
Logistic regression is one of the most important classification algorithms in machine learning. It provides a probabilistic view of classification while still being easy to interpret.

## Core idea
Rather than predicting a continuous value directly, logistic regression transforms a linear score into a probability between 0 and 1 using the sigmoid function.

## Mathematical intuition
The model computes:

z = w^T x + b

p = 1 / (1 + e^{-z})

The loss is usually binary cross-entropy, and the parameters are optimized via gradient descent.

## Intuition
Logistic regression is a linear classifier, but it models probabilities. This makes it valuable for decision-making and interpretable classification.

## Practical workflow in the notebook
- prepare inputs and labels
- fit logistic regression
- interpret coefficients and probabilities
- evaluate accuracy, precision, recall, and confidence

## Interview-ready explanation
"Logistic regression is a linear classifier with a sigmoid output. It estimates class probabilities while retaining interpretable coefficients, making it foundational in both classical ML and interviews."

## Common pitfalls
- confusing classification probability with certainty
- ignoring class imbalance
- assuming linear boundaries are sufficient for every problem

## Takeaway
This notebook is a must-know concept in machine learning interviews and applied modeling.
