# AI/ML Interview Cheat Sheet

This cheat sheet is designed for quick revision and interview-ready understanding. The goal is to explain each important idea in plain language, with the core equation and the key intuition behind it.

---

## 1. Linear Regression

### Concept
Predicts a continuous value using a linear combination of features.

### Formula

y = β0 + β1x1 + β2x2 + ... + βnxn + ε

### Intuition
The model finds the line or hyperplane that minimizes prediction error.

### Interview answer
"Linear regression models the relationship between input features and a continuous target by estimating coefficients that minimize squared error. It is simple, interpretable, and widely used when the output is numeric."

---

## 2. Logistic Regression

### Concept
Classifies data by estimating class probabilities.

### Formula

p = 1 / (1 + e^{-z}), where z = w^T x + b

### Intuition
It turns a linear score into a probability between 0 and 1.

### Interview answer
"Logistic regression is a linear classifier with a sigmoid output. It estimates class probabilities and is often chosen for binary classification because it is interpretable and efficient."

---

## 3. Decision Trees

### Concept
Splits data using feature thresholds to reduce impurity.

### Intuition
Each split tries to create purer groups with respect to the target variable.

### Interview answer
"A decision tree recursively splits the input space based on feature conditions. It is easy to interpret, but can overfit if not regularized."

---

## 4. Random Forest

### Concept
An ensemble of many decision trees.

### Intuition
Each tree sees a different random subset of data/features; averaging reduces variance.

### Interview answer
"Random forest builds many independent trees and aggregates their outputs. This reduces variance and often improves generalization compared with a single decision tree."

---

## 5. Boosting

### Concept
Train weak learners sequentially, each correcting mistakes of the previous ones.

### Intuition
Boosting focuses on difficult examples.

### Interview answer
"Boosting builds models sequentially, where each new learner focuses on the errors of the previous one. It often results in a strong model, especially on tabular data."

---

## 6. XGBoost

### Concept
Optimized gradient boosting using regularization and efficient tree building.

### Intuition
It improves the boosting objective while controlling complexity.

### Interview answer
"XGBoost is a highly optimized gradient boosting framework that includes regularization, second-order optimization, and efficient tree split search. It is widely used in structured-data competitions and production tasks."

---

## 7. KMeans

### Concept
Unsupervised clustering by minimizing within-cluster distance to centroids.

### Formula

J = Σ ||x_i - μ_k||^2

### Intuition
Points are assigned to the nearest cluster center, and centers are updated repeatedly.

### Interview answer
"KMeans partitions data into K clusters by minimizing the squared distance between points and their cluster centers. It is fast and simple, but it assumes clusters are roughly spherical and requires K to be chosen in advance."

---

## 8. DBSCAN

### Concept
Density-based clustering that can find non-spherical clusters and noise.

### Intuition
Points that are densely connected belong to the same cluster; sparse regions are treated as noise.

### Interview answer
"DBSCAN groups points based on local density rather than centroids. It is useful when clusters have irregular shapes and when you want to identify noise points."

---

## 9. PCA

### Concept
Reduce dimensions while preserving as much variance as possible.

### Intuition
Find the axes with maximum variance and project the data onto them.

### Interview answer
"PCA rotates data into a lower-dimensional coordinate system where variance is concentrated. It is useful for compression, visualization, and noise reduction."

---

## 10. Perceptron

### Concept
The simplest trainable unit for binary classification.

### Formula

y = sign(w^T x + b)

### Intuition
It learns a linear decision boundary from errors.

### Interview answer
"The perceptron is a single-layer linear classifier. It updates weights when a sample is misclassified, which makes it the foundation of neural-network learning."

---

## 11. Activation Functions

### Concept
Introduce nonlinearity so neural networks can model complex patterns.

### Common functions
- Sigmoid: σ(z) = 1 / (1 + e^{-z})
- Tanh: tanh(z)
- ReLU: max(0, z)

### Intuition
Without activation functions, stacked layers would behave like one large linear transformation.

### Interview answer
"Activation functions determine whether a neuron should fire and introduce the nonlinearity required to model complex data distributions. ReLU is widely used because it is simple and works well in deep networks."

---

## 12. Backpropagation

### Concept
Propagates the error backward through the network using the chain rule.

### Formula

w <- w - η * ∂L/∂w

### Intuition
It tells each parameter how much it contributed to the loss and in which direction to adjust.

### Interview answer
"Backpropagation computes gradients for every parameter by applying the chain rule through the network. This allows the model to update its weights and reduce the loss iteratively."

---

## 13. RNN

### Concept
Processes sequences step by step using hidden states.

### Formula

h_t = f(W_x x_t + W_h h_{t-1} + b)

### Intuition
The model remembers earlier context when reading later tokens.

### Interview answer
"RNNs are designed for sequence data. They update a hidden state at each time step, enabling the model to incorporate previous context into later predictions, though they struggle with long-range dependencies."

---

## 14. LSTM

### Concept
A gated RNN designed to handle long-term dependencies.

### Intuition
It keeps a cell state and uses gates to decide what to keep, forget, and expose.

### Interview answer
"LSTM adds explicit memory control through gates. This allows it to preserve useful information over longer sequence lengths and avoid the vanishing-gradient problem more effectively than simple RNNs."

---

## 15. GRU

### Concept
A simplified version of LSTM with fewer parameters.

### Intuition
It uses update and reset gates to manage memory efficiently.

### Interview answer
"GRU is a lighter alternative to LSTM. It has fewer gates and parameters, but still captures sequence dependencies and is often faster and easier to train."

---

## 16. Attention

### Concept
Selectively focuses on the most relevant parts of the sequence.

### Formula

scores = QK^T / sqrt(d_k)
weights = softmax(scores)
context = weights * V

### Intuition
Instead of compressing the entire sequence into one hidden state, the model can attend to the informative elements selectively.

### Interview answer
"Attention allows a model to assign importance to different tokens based on context. This is especially useful in language tasks where some words matter more than others at a given moment."

---

## 17. Transformer

### Concept
A neural architecture built around self-attention with no recurrence.

### Intuition
Every token can directly look at all other tokens in parallel, which makes it very powerful for NLP.

### Interview answer
"A transformer uses self-attention to capture relationships between all tokens in a sequence in parallel. It avoids recurrence, improves scalability, and is now the foundation of most modern NLP models."

---

## 18. NLP Tasks

### Classification
- sentiment analysis
- topic classification
- spam detection

### Translation
- sequence-to-sequence mapping from source to target language

### Question answering
- passage + question -> predicted answer

### Summarization
- compressed understanding of long text

### Interview answer
"Modern NLP tasks are all about learning representations of language and then mapping them to a relevant output, whether that output is a label, a translated sentence, or an answer span."

---

## Final interview strategy

When asked to explain a concept, always cover four things:

1. What is it?
2. Why does it exist?
3. What is the core equation?
4. What is the trade-off?

Examples:
- Logistic regression: classification probability + interpretability
- LSTM: memory control + longer dependency handling
- Transformer: attention + parallelization + modern NLP dominance

That structure is what makes your answer strong in interviews.

---

## Quick memory line

"Machine learning finds patterns; deep learning learns representations; sequence models learn context; transformers learn relationships at scale."
