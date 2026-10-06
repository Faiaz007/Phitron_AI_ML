# Module 18 Practice: Random Forest

## Why this notebook matters
Random forest is a classic ensemble method based on bagging and decision trees. It is robust, interpretable, and widely used in tabular data modeling.

## Core idea
A random forest builds many decision trees on bootstrapped data and combines their predictions. Randomness in feature selection reduces correlation among trees.

## Mathematical intuition
Decision trees split data using impurity measures such as Gini or entropy. Each tree captures a different view of the data, and averaging reduces variance.

## Intuition
Random forests are like asking many slightly different experts and then taking a majority vote. This usually improves stability and reduces overfitting compared with a single deep tree.

## Practical workflow in the notebook
- train multiple trees on sampled subsets
- tune tree depth and number of estimators
- evaluate out-of-bag or validation performance
- inspect feature importance

## Interview-ready explanation
"Random forests combine many decision trees trained on different sampled subsets of data. The ensemble reduces variance, handles nonlinear patterns well, and provides feature importance estimates."

## Common pitfalls
- assuming more trees always help
- ignoring class imbalance
- treating feature importance as causal truth

## Takeaway
Random forest is one of the best classic interview topics because it balances interpretability, accuracy, and robustness.
