# Module 24 PCA Practice

## Why this notebook matters
Practice makes PCA intuitive. The goal is to see how variance is redistributed and how information can be compressed without destroying structure.

## Core idea
Not all dimensions are equally informative. PCA identifies the directions that matter most.

## Mathematical intuition
The first principal component captures the largest variance. Each subsequent component captures the remaining variance subject to orthogonality.

## Intuition
If data lives on a curved or high-dimensional manifold, PCA approximates it in a lower-dimensional, more informative basis.

## Practical workflow in the notebook
- standardize data
- compute covariance matrix
- extract principal components
- inspect explained variance ratio
- decide number of components

## Interview-ready explanation
"The purpose of PCA is to retain the maximum signal in fewer dimensions. This makes data easier to visualize, less noisy, and often easier for downstream algorithms."

## Common pitfalls
- assuming PCA always improves performance
- ignoring interpretation of reduced coordinates
- choosing dimensions based only on intuition instead of explained variance

## Takeaway
PCA is one of the clearest examples of representation learning in classical ML.
