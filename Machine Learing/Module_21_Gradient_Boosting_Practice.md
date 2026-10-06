# Module 21 Gradient Boosting Practice

## Why this notebook matters
This notebook gives a hands-on view of the residual-correction idea behind gradient boosting.

## Core idea
Each learner fits the residuals of the previous model, not just the raw target.

## Mathematical intuition
The series of models approximates the negative gradient of the loss. This is why the method is called gradient boosting.

## Intuition
A model learns from the mistakes left behind by the previous model. Over time, the residuals shrink and the ensemble becomes stronger.

## Practical workflow in the notebook
- compute residual errors
- fit a weak learner to residuals
- update the ensemble
- observe decreasing error

## Interview-ready explanation
"Gradient boosting trains models to predict the error of the current ensemble. This residual learning approach is mathematically aligned with gradient descent in function space."

## Common pitfalls
- misunderstanding residuals
- tuning too aggressively without validation
- overfitting if the number of trees is too high

## Takeaway
Residual learning is a key idea in modern ensemble modeling.
