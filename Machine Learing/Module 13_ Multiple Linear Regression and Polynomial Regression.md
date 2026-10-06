# Module 13: Multiple Linear Regression and Polynomial Regression

## Why this notebook matters
This notebook extends simple regression to richer models that can capture several predictive variables and nonlinear relationships.

## Core idea
Simple linear regression uses one predictor. Multiple linear regression uses many predictors:

y = b0 + b1 x1 + b2 x2 + ... + bn xn + e

Polynomial regression models nonlinear relationships by including higher-order terms such as x^2 or x^3.

## Mathematical intuition
The optimization goal is to minimize squared error between predictions and actual values. This leads to a least-squares solution. In polynomial regression, the model becomes nonlinear in x but linear in the coefficients, which is why it is still tractable.

## Intuition
More features allow richer patterns, but risk overfitting. Polynomial terms can fit curved data well, but too much complexity leads to poor generalization.

## Practical workflow in the notebook
- prepare feature matrix
- fit linear and polynomial models
- compare training error and prediction quality
- inspect coefficients and model shape

## Interview-ready explanation
"Multiple linear regression models how several inputs influence a target simultaneously, while polynomial regression captures curvature by adding nonlinear terms. Both are fitted with least squares and require careful regularization and validation."

## Common pitfalls
- confusing features with target variables
- ignoring scaling when using certain optimization algorithms
- overfitting with high-degree polynomials

## Takeaway
This notebook is a major step from simpler models to realistic data relationships.
