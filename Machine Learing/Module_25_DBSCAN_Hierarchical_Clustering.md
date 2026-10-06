# Module 25: DBSCAN & Hierarchical Clustering

## Why this notebook matters
This notebook introduces density-based and hierarchical clustering, which are powerful alternatives to KMeans when cluster shapes and densities vary.

## Core idea
DBSCAN groups points that are close to one another and labels sparse regions as noise. Hierarchical clustering builds a nested cluster tree.

## Mathematical intuition
DBSCAN depends on a neighborhood radius eps and minimum points threshold. Points are connected if they are dense enough. Hierarchical clustering uses distances between groups to merge or split clusters.

## Intuition
Not all clusters are equal-sized spheres. Real data may contain irregular shapes, noise, and varying densities, which is where methods like DBSCAN shine.

## Practical workflow in the notebook
- compute distance relationships
- choose eps and min_samples for DBSCAN
- build a dendrogram for hierarchical clustering
- compare cluster structures

## Interview-ready explanation
"DBSCAN identifies dense regions and noise, while hierarchical clustering builds nested cluster relationships. These methods are useful when clusters are irregular or when the number of clusters is unknown."

## Common pitfalls
- assuming all clustering problems are spherical
- choosing eps poorly
- interpreting raw dendrograms without understanding linkage choices

## Takeaway
This notebook broadens the learner's view of unsupervised learning beyond centroid-based clustering.
