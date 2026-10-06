# Machine Learning

This section covers the classical machine learning ideas that form the basis of modern AI. Here the emphasis is on learning patterns from data, optimizing an objective, and evaluating generalization.

## Objective

The machine learning section focuses on:

- statistics and data understanding
- regression and classification
- ensemble methods
- clustering
- dimensionality reduction
- model validation and iteration

## Core concepts

### 1. Concept
Machine learning is the process of learning a mapping from input features to output targets by minimizing an objective function.

### 2. Intuition
A model is effectively a function approximator. It uses data to estimate parameters such that it reduces prediction error and generalizes to unseen examples.

### 3. Math
Key formulas include:

Linear regression:

y = β0 + β1x1 + β2x2 + ... + βnxn + ε

Logistic regression:

p = 1 / (1 + e^{-z}), where z = w^T x + b

Error function for regression:

L = Σ (y_i - ŷ_i)^2

### 4. Architecture of classical ML

Raw data -> EDA -> preprocessing -> feature engineering -> model -> validation -> tuning -> deployment

### 5. Code pattern

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### 6. Interview-ready explanation

"Machine learning learns from data by finding parameters that minimize a loss function. The choice of feature representation, algorithm, and evaluation metric determines whether the model can generalize. Classical ML methods still matter because they teach the core ideas behind optimization, bias-variance tradeoff, and decision boundaries."

### 7. Common pitfalls

- using accuracy alone for imbalanced classification
- ignoring scaling for distance-based methods
- not validating with a proper split
- interpreting feature importance as causality

### 8. Takeaway

The core of ML is not only prediction, but understanding assumptions, measuring error, and building models that generalize.

---

## Major topics in this section

### Regression and classification
- linear regression
- polynomial regression
- multiple regression
- logistic regression

### Tree models and ensembles
- decision trees
- random forest
- AdaBoost
- gradient boosting
- XGBoost

### Clustering
- KMeans
- DBSCAN
- hierarchical clustering

### Dimensionality reduction
- PCA

## Interview questions

- What is bias-variance tradeoff?
- Why is scaling important?
- What is the difference between linear and logistic regression?
- What is bagging versus boosting?
- When do you prefer PCA over raw features?
- Why can KMeans fail on non-spherical clusters?

## Recommended study order

1. Start with statistics and preprocessing
2. Learn regression and classification
3. Move to trees and boosting
4. Study clustering and PCA
5. Revisit notebooks and explain each model in simple terms
