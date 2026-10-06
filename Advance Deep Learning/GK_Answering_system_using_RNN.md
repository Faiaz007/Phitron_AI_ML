# GK Answering System using RNN

## Why this notebook matters
This notebook explores how a recurrent neural network can process sequential information and answer general-knowledge style questions. In real-world AI systems, the meaningful signal is often not a single vector but a time-ordered sequence of tokens, words, or events.

## Core idea
RNNs keep a hidden state that evolves as new input arrives. This makes them useful for sequences where earlier context changes the meaning of later tokens.

For a sequence x1, x2, ..., xt, the hidden state updates as:

h_t = f(W_x x_t + W_h h_{t-1} + b)

The model then predicts the next token or answer based on the final hidden state or the full sequence state.

## Mathematical intuition
The key idea is that the network shares parameters across time steps. Instead of learning a separate model for every position, it reuses the same transformation repeatedly. This reduces parameter count while modeling temporal dependence.

The challenge is the vanishing gradient problem: as the sequence gets longer, gradients can shrink excessively. This is one reason why later architectures like LSTMs and GRUs were developed.

## How to think about it
An RNN is like reading a sentence and updating your understanding after each word. Each word changes the current memory. For question-answering, the model reads the question and context, updates hidden states, and decides which answer is most likely.

## Practical workflow in the notebook
- prepare question/context pairs
- tokenize text
- encode tokens into vectors
- feed sequences into an RNN
- train using cross-entropy loss
- evaluate answer correctness

## Interview-ready explanation
"An RNN is a neural network for ordered data. It passes information from one time step to the next using hidden states. This allows the model to incorporate context from earlier inputs when predicting later outputs."

## Common pitfalls
- not handling variable-length sequences properly
- ignoring tokenization and padding
- training with too little data
- treating the final hidden state as the only source of information when attention or context pooling may help

## Takeaway
This notebook is a practical introduction to sequence modeling and the foundation behind many NLP systems. The main lesson is that sequence understanding requires memory across time, and RNNs are a direct way to encode that idea.
