# Module 24: PCA

## Why this notebook matters
Principal Component Analysis (PCA) is one of the most important dimensionality reduction methods. It helps compress data while keeping as much variation as possible.

## Core idea
PCA finds orthogonal directions of maximum variance in the data. These directions become principal components.

## Mathematical intuition
The covariance matrix captures relationships among features. Its eigenvectors give principal directions, and eigenvalues tell how much variance each direction explains.

## Intuition
PCA reduces redundancy by projecting data to a smaller space where variation is preserved. This is especially helpful in high-dimensional datasets.

## Practical workflow in the notebook
- standardize features
- compute covariance matrix
- find eigenvalues and eigenvectors
- sort components by explained variance
- project data to fewer dimensions

## Interview-ready explanation
"PCA is a linear dimensionality reduction technique that rotates data into a new basis where variance is concentrated in fewer dimensions. It helps visualize data and may improve model performance by removing noise and redundancy."

## Common pitfalls
- not scaling the data before PCA
- interpreting principal components as direct features
- using too few dimensions without checking variance loss

## Takeaway
PCA is a foundational concept in both data science and modern ML pipelines.
