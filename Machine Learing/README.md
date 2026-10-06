# Machine Learing

This folder is the core of the repository. It covers the essential mathematical and algorithmic foundations of machine learning: data understanding, supervised learning, tree-based models, boosting methods, clustering, and dimensionality reduction.

## Objective

The purpose of this section is to build a strong understanding of how machine learning works from first principles, including:

- how data is transformed into features
- how models fit patterns in data
- how error is measured and minimized
- how algorithms differ in assumptions and behavior
- how to explain the idea in interview-ready language

## Core topics

### Fundamentals
- descriptive statistics and distributions
- feature scaling and encoding
- preprocessing and feature engineering
- exploratory data analysis (EDA)

### Regression and classification
- linear and polynomial regression
- logistic regression
- classification metrics
- model evaluation

### Ensemble methods
- random forest
- AdaBoost
- gradient boosting
- XGBoost

### Clustering
- KMeans
- hierarchical clustering
- DBSCAN

### Dimensionality reduction
- PCA

## Notebook index

### Fundamentals
- [AI_ML_Week_01_Module_01_Descriptive_Statistics_and_Distributions.ipynb](./AI_ML_Week_01_Module_01_Descriptive_Statistics_and_Distributions.ipynb)
- [Module_03_Scaling,_Encoding,_and_Distances.ipynb](./Module_03_Scaling,_Encoding,_and_Distances.ipynb)
- [Module_07_Data_Preprocessing_and_Feature_Engineering.ipynb](./Module_07_Data_Preprocessing_and_Feature_Engineering.ipynb)
- [Module_09_Data_Preprocessing_and_Feature_Engineering_Part 2.ipynb](./Module_09_Data_Preprocessing_and_Feature_Engineering_Part%202.ipynb)
- [Module_6_5_Practice_on_Module_06_EDA_.ipynb](./Module_6_5_Practice_on_Module_06_EDA_.ipynb)

### Regression and classification
- [Module 13_ Multiple Linear Regression and Polynomial Regression.ipynb](./Module%2013_%20Multiple%20Linear%20Regression%20and%20Polynomial%20Regression.ipynb)
- [Module_13_Practice.ipynb](./Module_13_Practice.ipynb)
- [Module_14_Logistic_Regression.ipynb](./Module_14_Logistic_Regression.ipynb)
- [Module_14_Practice_Iris_Logistic_Regression.ipynb](./Module_14_Practice_Iris_Logistic_Regression.ipynb)
- [Module_15_Practice.ipynb](./Module_15_Practice.ipynb)

### Ensemble and trees
- [Module_18_Practice_Random_Forest.ipynb](./Module_18_Practice_Random_Forest.ipynb)
- [Module_20_AdaBoost.ipynb](./Module_20_AdaBoost.ipynb)
- [Module_20_AdaBoost_Practice.ipynb](./Module_20_AdaBoost_Practice.ipynb)
- [Module 21_Gradient Boosting.ipynb](./Module%2021_Gradient%20Boosting.ipynb)
- [Module_21_Gradient_Boosting_Practice.ipynb](./Module_21_Gradient_Boosting_Practice.ipynb)
- [Module 22_XGBoost.ipynb](./Module%2022_XGBoost.ipynb)
- [Module_22_XGBoost_Practice.ipynb](./Module_22_XGBoost_Practice.ipynb)

### Clustering and dimensionality reduction
- [Module_23_KMeans_Clustering.ipynb](./Module_23_KMeans_Clustering.ipynb)
- [Module_23_KMeans_Practice.ipynb](./Module_23_KMeans_Practice.ipynb)
- [Module_24_PCA.ipynb](./Module_24_PCA.ipynb)
- [Module_24_PCA_Practice.ipynb](./Module_24_PCA_Practice.ipynb)
- [Module_25_DBSCAN_Hierarchical_Clustering.ipynb](./Module_25_DBSCAN_Hierarchical_Clustering.ipynb)
- [Module_25_Practice_DBSCAN_&_Hierarchical.ipynb](./Module_25_Practice_DBSCAN_&_Hierarchical.ipynb)

### Applied ML problems
- [Module_08_ML_Assignment_02.ipynb](./Module_08_ML_Assignment_02.ipynb)
- [Module_10_Part_01.ipynb](./Module_10_Part_01.ipynb)
- [Module_11_Composite_Colab_Notebook.ipynb](./Module_11_Composite_Colab_Notebook.ipynb)
- [Titanic_Data_Preparation.ipynb](./Titanic_Data_Preparation.ipynb)
- [laptop_price_prediction.ipynb](./laptop_price_prediction.ipynb)
- [student_performance_prediction.ipynb](./student_performance_prediction.ipynb)

---

## Concept

Machine learning models learn patterns from data by optimizing an objective function. The algorithm choice depends on the data geometry, the target type, and the desired interpretability.

## Intuition

A model is not magical; it is a function approximator. It maps input features to outputs by finding a parameterization that reduces error on observed data and hopefully generalizes to unseen data.

## Math

The main mathematical ideas in this folder are:

- mean, variance, covariance
- squared error and least squares
- sigmoid and binary cross-entropy
- entropy and Gini impurity
- Euclidean distance and centroid updates
- variance maximization in PCA
- residual fitting in boosting

For example, linear regression estimates:

y = β0 + β1x1 + β2x2 + ... + βnxn + ε

and minimizes the squared residual error:

L = Σ (yi - ŷi)^2

Meanwhile, logistic regression estimates probabilities with:

p = 1 / (1 + e^{-z}), where z = w^T x + b

## Coding pattern

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

This is the standard ML workflow: split, preprocess, model, evaluate, iterate.

## Architectural design of classical ML

A classical ML pipeline usually looks like this:

Raw data -> EDA -> preprocessing -> feature engineering -> train model -> validate -> tune -> deploy

This architecture is important because success is often driven by representation and validation, not only by model type.

## Common interview questions

- What is the bias-variance tradeoff?
- Why is feature scaling important?
- What is the difference between linear and logistic regression?
- Why does a decision tree split on impurity?
- What is the difference between bagging and boosting?
- Why do we use PCA?
- When is KMeans not appropriate?

## Interview cheat sheet

### Regression
- linear: continuous prediction, least squares
- polynomial: captures curvature
- regularization controls complexity

### Classification
- logistic regression: probabilistic classification
- decision boundaries depend on feature space geometry
- metrics: accuracy, precision, recall, F1, ROC-AUC

### Ensembles
- random forest: bagging + random feature subsets
- AdaBoost: weight difficult samples more strongly
- gradient boosting: fit residuals with weak learners
- XGBoost: boosted trees with regularization and efficiency

### Clustering
- KMeans: centroid-based, fixed K
- DBSCAN: density-based, finds arbitrary shapes and noise
- hierarchical clustering: nested cluster structures

### Reducing dimensions
- PCA: compress features using principal directions of variance

## Recommended order

1. Understand statistics and preprocessing
2. Learn regression and classification
3. Cover trees and ensembles
4. Study clustering and PCA
5. Practice with project notebooks and explain results aloud
