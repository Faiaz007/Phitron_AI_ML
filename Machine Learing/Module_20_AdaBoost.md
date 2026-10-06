# Module 20: AdaBoost

## Why this notebook matters
AdaBoost is a foundational boosting algorithm that sequentially focuses on hard examples. It is historically important and conceptually elegant.

## Core idea
Each weak learner is trained on the data but weighted according to mistakes. Misclassified points receive higher importance in the next iteration.

## Mathematical intuition
The ensemble prediction is a weighted vote of weak learners.

H(x) = sign(sum_t alpha_t h_t(x))

The learner weights alpha_t depend on the training error of each model.

## Intuition
AdaBoost is like giving more attention to the examples the model gets wrong. It gradually turns weak learners into a strong classifier.

## Practical workflow in the notebook
- train a weak classifier
- compute sample weights
- reweight errors
- add the next learner
- evaluate the final ensemble

## Interview-ready explanation
"AdaBoost is a boosting algorithm that emphasizes difficult examples by increasing their weights in each iteration. It converts a collection of weak learners into a stronger classifier."

## Common pitfalls
- poor base learners can limit performance
- overfitting when too many learners are added
- confusion between boosting and bagging

## Takeaway
AdaBoost is an excellent conceptual bridge between ensemble thinking and modern boosting methods.
