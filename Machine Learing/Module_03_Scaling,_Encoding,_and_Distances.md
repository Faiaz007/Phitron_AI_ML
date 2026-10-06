# Module 03: Scaling, Encoding, and Distances

## Why this notebook matters
Before applying many machine learning algorithms, data must be transformed so models can interpret it correctly. Scaling, encoding, and distance metrics dramatically affect performance.

## Core idea
Machine learning algorithms are sensitive to feature scales and representation. Distance-based methods, for example, can be distorted by features with large ranges.

## Mathematical intuition
For Euclidean distance:

d(x, y) = sqrt(sum_i (x_i - y_i)^2)

If one feature varies from 0 to 1000 and another from 0 to 1, the first feature dominates the distance even if it is less important.

This is why normalization and standardization matter.

## Intuition
Encoding turns categories into numbers without losing meaning, while scaling ensures features contribute fairly.

## Practical workflow in the notebook
- standardize continuous features
- one-hot encode categorical variables
- calculate distance between samples
- compare results before and after transformation

## Interview-ready explanation
"Feature scaling and encoding are preprocessing steps that improve the fairness and stability of learning algorithms. Distance-based models in particular are highly sensitive to scale and representation."

## Common pitfalls
- applying one-hot encoding incorrectly
- forgetting to fit scaler on training data only
- ignoring categorical missingness or rare categories

## Takeaway
Good preprocessing is often the difference between an algorithm that works and one that fails.
