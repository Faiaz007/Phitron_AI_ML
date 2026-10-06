# Module 25 Practice: DBSCAN & Hierarchical

## Why this notebook matters
This notebook is the practical extension of clustering theory, highlighting how different algorithms behave on the same data.

## Core idea
The same dataset can produce different cluster structures depending on the chosen algorithm and assumptions.

## Mathematical intuition
Distance metrics, linkage criterion, neighborhood radius, and density thresholds define the cluster structure. These choices are algorithmic assumptions, not just hyperparameters.

## Intuition
Cluster analysis is as much about the question you are asking as about the data. Different criteria reveal different meaningful structures.

## Practical workflow in the notebook
- compare KMeans, DBSCAN, and hierarchical clustering
- inspect clusters and density patterns
- choose the method that matches the data geometry

## Interview-ready explanation
"Clustering quality depends on the selected method and assumptions. There is no universally best algorithm; the right choice depends on cluster shape, density, and the problem context."

## Common pitfalls
- evaluating clustering without domain context
- ignoring noise points
- forcing a single algorithm onto all datasets

## Takeaway
This notebook teaches the most important lesson in clustering: the method must respect the geometry of the data.
