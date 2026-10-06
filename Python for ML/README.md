# Python for ML

This folder builds the Python and data preparation foundation required before diving into machine learning and deep learning.

## Core focus

- Python essentials for ML workflows
- NumPy and array-based thinking
- pandas for real data analysis
- feature engineering and model-ready preprocessing
- an intuitive understanding of how raw data becomes trainable input

## Table of contents

- [From_Pandas_to_Model_Ready_Titanic_Data_Preparation_(Step_by_Step).ipynb](./From_Pandas_to_Model_Ready_Titanic_Data_Preparation_(Step_by_Step).ipynb)
- [Phitron_Module_10.ipynb](./Phitron_Module_10.ipynb)
- [Phitron_Module_11.ipynb](./Phitron_Module_11.ipynb)
- [Phitron_Module_15.ipynb](./Phitron_Module_15.ipynb)
- [Phitron_Practice_15.5.ipynb](./Phitron_Practice_15.5.ipynb)

## Learning path

### 1. Concept
Python is the glue between data, math, and model training. ML pipelines are built with iteration, function design, tabular manipulation, and structured preprocessing.

### 2. Intuition
Raw data is not directly useful to a model. You need to clean, reshape, encode, and summarize it before optimization can begin.

### 3. Math
The math here is mostly linear algebra and data summarization:

- vectors and matrices
- broadcasting and shape rules
- means, variances, and distributions
- feature transformation and scaling

### 4. Coding pattern
Typical pipeline:

```python
import pandas as pd
import numpy as np

# load data
df = pd.read_csv("data.csv")

# inspect and clean
df = df.dropna()

# generate feature matrix
X = df.drop(columns=["target"])
y = df["target"]

# preprocess
X = pd.get_dummies(X)
X = (X - X.mean()) / X.std()
```

### 5. Common interview questions
- Why is data preprocessing essential before model training?
- What is the difference between a list, a NumPy array, and a pandas DataFrame?
- Why do we encode categorical features?
- How does feature scaling affect distance-based models?
- What is the difference between rows as observations and columns as features?

## Interview cheat sheet

- Python is not the model; it is the execution language for the model.
- pandas is for tabular understanding and transformation.
- NumPy is for efficient numerical computation.
- Data cleaning is often where the real signal is discovered.
- A strong model starts with strong representation.

## Recommended order

1. Start with Module 10 and Module 11
2. Understand pandas workflows in Module 15
3. Practice the Titanic preparation notebook
4. Revisit each notebook and reimplement the feature logic in your own style
