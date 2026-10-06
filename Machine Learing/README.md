# Machine Learing

This folder covers the core machine learning journey: fundamentals, supervised learning, ensembles, clustering, and dimensionality reduction. It is the central section of the repository and the most interview-heavy set of topics.

## Core focus

- descriptive statistics and distributions
- preprocessing and feature engineering
- linear and polynomial regression
- logistic regression and classification basics
- tree-based learning and random forest
- boosting methods: AdaBoost, Gradient Boosting, XGBoost
- clustering: KMeans, DBSCAN, hierarchical clustering
- PCA for dimensionality reduction
- practical datasets and project notebooks

## Table of contents

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

### Tree models and ensembles
- [Module_18_Practice_Random_Forest.ipynb](./Module_18_Practice_Random_Forest.ipynb)
- [Module_20_AdaBoost.ipynb](./Module_20_AdaBoost.ipynb)
- [Module_20_AdaBoost_Practice.ipynb](./Module_20_AdaBoost_Practice.ipynb)
- [Module 21_Gradient Boosting.ipynb](./Module%2021_Gradient%20Boosting.ipynb)
- [Module_21_Gradient_Boosting_Practice.ipynb](./Module_21_Gradient_Boosting_Practice.ipynb)
- [Module 22_XGBoost.ipynb](./Module%2022_XGBoost.ipynb)
- [Module_22_XGBoost_Practice.ipynb](./Module_22_XGBoost_Practice.ipynb)

### Clustering and reduction
- [Module_23_KMeans_Clustering.ipynb](./Module_23_KMeans_Clustering.ipynb)
- [Module_23_KMeans_Practice.ipynb](./Module_23_KMeans_Practice.ipynb)
- [Module_24_PCA.ipynb](./Module_24_PCA.ipynb)
- [Module_24_PCA_Practice.ipynb](./Module_24_PCA_Practice.ipynb)
- [Module_25_DBSCAN_Hierarchical_Clustering.ipynb](./Module_25_DBSCAN_Hierarchical_Clustering.ipynb)
- [Module_25_Practice_DBSCAN_&_Hierarchical.ipynb](./Module_25_Practice_DBSCAN_&_Hierarchical.ipynb)

### Projects and applied tasks
- [Module_08_ML_Assignment_02.ipynb](./Module_08_ML_Assignment_02.ipynb)
- [Module_10_Part_01.ipynb](./Module_10_Part_01.ipynb)
- [Module_11_Composite_Colab_Notebook.ipynb](./Module_11_Composite_Colab_Notebook.ipynb)
- [Practice_22_5.ipynb](./Practice_22_5.ipynb)
- [Titanic_Data_Preparation.ipynb](./Titanic_Data_Preparation.ipynb)
- [laptop_price_prediction.ipynb](./laptop_price_prediction.ipynb)
- [laptop_price_prediction (1).ipynb](./laptop_price_prediction%20(1).ipynb)
- [student_performance_prediction.ipynb](./student_performance_prediction.ipynb)

## Learning path

### 1. Concept
This folder teaches the standard machine learning lifecycle:

- understand the data
- preprocess and engineer features
- choose a suitable model
- evaluate properly
- iterate on assumptions

### 2. Intuition
ML is not just about fitting numbers. It is about finding patterns that generalize. A good model is one that captures signal and ignores noise.

### 3. Math
The central math concepts are:

- expected value and variance
- linear combinations and coefficients
- least squares for regression
- sigmoid and log-loss for classification
- entropy and Gini for decision trees
- distance and covariance for clustering and PCA
- residuals and boosting

### 4. Coding pattern
Typical flow for ML notebooks:

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

### 5. Common interview questions
- What is the difference between bias and variance?
- Why is feature scaling important for distance-based algorithms?
- Which metric is better for imbalanced classification: accuracy or F1?
- How does logistic regression differ from linear regression?
- What is the difference between bagging and boosting?
- Why does PCA matter for high-dimensional data?
- When would you prefer DBSCAN over KMeans?

## Interview cheat sheet

### Regression
- Linear regression minimizes squared error.
- Multiple regression models several inputs at once.
- Polynomial regression handles non-linear curvature.
- Use RMSE/MAE for evaluation.

### Classification
- Logistic regression outputs probabilities using sigmoid.
- Decision boundaries are linear in feature space.
- Metrics: accuracy, precision, recall, F1, ROC-AUC.

### Ensembles
- Random forest reduces variance by averaging many trees.
- AdaBoost increases attention to misclassified samples.
- Gradient boosting fits residuals iteratively.
- XGBoost adds regularization and efficient optimization.

### Clustering
- KMeans assumes a fixed number of clusters and spherical groups.
- DBSCAN finds dense groups and isolates noise.
- Hierarchical clustering reveals nested structure.

### Dimensionality reduction
- PCA compresses features while preserving variance.
- It is useful for visualization and noise reduction.

## Recommended order

1. Start with statistics and preprocessing
2. Learn linear/logistic regression
3. Move to tree models and ensembles
4. Then study clustering, PCA, and project notebooks
5. Finish by revisiting the notebooks and explaining each algorithm in your own words
