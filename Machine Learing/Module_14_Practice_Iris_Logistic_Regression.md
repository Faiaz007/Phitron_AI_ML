# Module 14 Practice: Iris Logistic Regression

## Why this notebook matters
The Iris dataset is a classic benchmark for classification. It is perfect for learning how logistic regression behaves in a simple but meaningful task.

## Core idea
The features describe flower measurements, and the target is the species. Logistic regression maps these measurements to class probabilities.

## Mathematical intuition
Each class may be modeled with a separate decision boundary. For multiclass settings, softmax generalizes logistic regression.

## Intuition
The model finds a linear boundary that separates classes as well as possible. Since iris is low-dimensional and well-structured, it is ideal for visualization.

## Practical workflow in the notebook
- load the Iris dataset
- split into train and test
- fit the classifier
- evaluate metrics
- inspect decision boundaries and misclassifications

## Interview-ready explanation
"The Iris dataset is a canonical example used to illustrate logistic regression and classification boundaries. It is one of the clearest ways to explain how probabilistic classifiers work in practice."

## Common pitfalls
- ignoring class overlap
- overfitting by tuning too aggressively on tiny datasets
- misreading visualizations without considering feature scales

## Takeaway
This notebook is a classic teaching example: simple, interpretable, and conceptually rich.
