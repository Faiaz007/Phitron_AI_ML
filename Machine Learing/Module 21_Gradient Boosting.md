# Module 21: Gradient Boosting

## Why this notebook matters
Gradient boosting is a powerful ensemble method that builds models sequentially, each improving on the mistakes of the previous ones.

## Core idea
Instead of training one strong model, boosting trains a sequence of weak learners. Each new learner focuses on the residual error left by the current ensemble.

## Mathematical intuition
The model updates as:

F_{m+1}(x) = F_m(x) + \eta h_m(x)

where h_m(x) is a learner trained to approximate the residuals of F_m.

This is closely related to gradient descent in function space.

## Intuition
Boosting is like a team of learners where each member specializes in correcting the prior mistakes. It often performs very well on structured tabular data.

## Practical workflow in the notebook
- train a weak base learner
- compute residuals
- fit a new learner to residuals
- combine predictions
- tune learning rate and number of estimators

## Interview-ready explanation
"Gradient boosting builds an ensemble by fitting models to the errors of prior models. It uses residual minimization and often delivers strong predictive performance with careful tuning."

## Common pitfalls
- too many estimators can overfit
- poor learning rate choices can cause unstable training
- treating boosting as black-box without understanding residuals

## Takeaway
This notebook is a cornerstone for modern tabular modeling and likely appears in interviews.
