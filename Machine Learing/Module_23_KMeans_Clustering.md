# Module 23: KMeans Clustering

## Why this notebook matters
Clustering is an unsupervised learning method used to group similar data points without labels.

## Core idea
KMeans tries to partition data into K clusters by minimizing the within-cluster squared distance to the cluster centers.

## Mathematical intuition
The objective is:

J = sum_{i=1}^{n} sum_{k=1}^{K} r_{ik} ||x_i - mu_k||^2

where mu_k is the centroid of cluster k and r_{ik} indicates whether point i belongs to cluster k.

## Intuition
The algorithm repeatedly assigns points to the nearest centroid and recomputes centroids until stable.

## Practical workflow in the notebook
- initialize K centroids
- assign points to nearest centers
- recompute centers
- repeat until convergence
- visualize clusters

## Interview-ready explanation
"KMeans groups data into K clusters by finding centroids that minimize within-cluster distances. It is simple, scalable, and widely used, but it requires a chosen K and assumes roughly spherical clusters."

## Common pitfalls
- choosing K poorly
- sensitivity to initialization
- poor performance on non-spherical or uneven clusters

## Takeaway
KMeans is a canonical clustering algorithm and one of the most interview-asked unsupervised learning topics.
