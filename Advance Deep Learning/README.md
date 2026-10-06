# Advanced Deep Learning

This folder is the bridge between classical neural networks and modern sequence models. It focuses on architectures that can handle ordered data, memory, and contextual reasoning, which are essential in natural language processing, speech, time-series analysis, and many other AI systems.

## Why this section matters

Feedforward networks assume each input is independent. That is not enough for language, text, and temporal signals where order matters. Sequence models were designed to capture dependencies across time steps and contexts.

The progression is:

- RNN: simple sequential memory
- LSTM: gated memory for long-range dependencies
- GRU: simplified gated memory
- Attention: dynamic focus on important tokens
- Transformer: attention-first architecture for parallelizable sequence modeling

---

## Learning goals

By the end of this section, you should be able to explain:

- what sequential data is and why it is different from tabular input
- how recurrent models preserve memory across time steps
- why RNNs struggle with long-range dependencies
- how gates improve memory in LSTM and GRU
- how attention lets a model focus on relevant context
- why transformers dominate modern NLP and sequence tasks
- how these ideas apply to classification, translation, and question answering

---

## Table of contents

- [RNN: Recurrent Neural Networks](#rnn-recurrent-neural-networks)
- [LSTM: Long Short-Term Memory](#lstm-long-short-term-memory)
- [GRU: Gated Recurrent Unit](#gru-gated-recurrent-unit)
- [Attention Mechanism](#attention-mechanism)
- [Transformer Architecture](#transformer-architecture)
- [NLP Tasks](#nlp-tasks)
- [Coding Patterns](#coding-patterns)
- [Interview Questions](#interview-questions)
- [Recommended Study Roadmap](#recommended-study-roadmap)

---

## RNN: Recurrent Neural Networks

### Concept

A recurrent neural network processes a sequence one step at a time and maintains a hidden state that carries information from previous time steps.

This makes the model capable of remembering context from earlier parts of a sequence when processing later parts.

### Intuition

An RNN is like reading a sentence word by word. At each word, the model updates its internal understanding based on the current word and the memory from previous words.

If the model reads:

"The cat sat on the mat"

the word "sat" is understood differently depending on the earlier words "The cat". This is precisely the motivation behind sequence modeling.

### Math

The core recurrence is:

h_t = f(W_x x_t + W_h h_{t-1} + b)

where:
- x_t is the current input
- h_{t-1} is the previous hidden state
- W_x and W_h are learnable weights
- b is the bias
- f is a nonlinear activation such as tanh or ReLU

The output at time t can be:

y_t = W_y h_t + b_y

### Why RNNs are useful

- naturally handle ordered inputs
- good for short sequences
- useful for text, speech, and time-series signals

### Why RNNs fail on long sequences

RNNs suffer from vanishing gradients and exploding gradients. As the sequence gets longer, gradients become too small or too large during backpropagation, making it hard to learn long-range dependencies.

This is the key reason LSTM and GRU were introduced.

### Architecture sketch

Input sequence: x1 -> x2 -> x3 -> ... -> xt

Hidden state flow:

h1 -> h2 -> h3 -> ... -> ht

The hidden state is updated recursively.

### Interview-ready explanation

"An RNN is a neural network designed for ordered data. It maintains a hidden state that carries context from earlier time steps so that later inputs can be interpreted with prior information. The limitation is that it struggles with long-range dependencies because gradients can vanish during backpropagation."

---

## LSTM: Long Short-Term Memory

### Concept

LSTM is a special recurrent architecture designed to solve the long-term dependency problem in standard RNNs.

### Key idea

Instead of relying only on a simple hidden state, LSTM introduces a cell state and gates that regulate what information is kept, forgotten, and used.

### The three main gates

1. Forget gate
   - decides what information to drop from the cell state

2. Input gate
   - decides what new information to store

3. Output gate
   - decides what to output from the current cell state

### Math intuition

For each time step t:

f_t = σ(W_f [h_{t-1}, x_t] + b_f)
i_t = σ(W_i [h_{t-1}, x_t] + b_i)

tilde{C}_t = tanh(W_C [h_{t-1}, x_t] + b_C)
C_t = f_t * C_{t-1} + i_t * tilde{C}_t

o_t = σ(W_o [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)

Where:
- C_t is the cell state
- h_t is the hidden state
- f_t, i_t, o_t are gate activations
- σ is the sigmoid function

### Why LSTM matters

- better memory control than standard RNN
- handles long-range dependencies more effectively
- widely used in NLP and sequence modeling before transformers

### Intuition

Think of the cell state as long-term memory and the hidden state as working memory. The gates decide which information is useful to keep and which to forget.

### Interview-ready explanation

"LSTM adds gated memory to standard RNNs. It uses a cell state and three gates to decide what to forget, what to remember, and what to expose to the next step. This makes it much better at long-range dependency learning than a vanilla RNN."

---

## GRU: Gated Recurrent Unit

### Concept

GRU is a simpler version of LSTM that also uses gates, but it has fewer parameters and a simpler update structure.

### Key idea

GRU combines the forget and input gates into a single update mechanism. It is computationally lighter while still capturing important sequence dependencies.

### Core equations

z_t = σ(W_z [h_{t-1}, x_t])
r_t = σ(W_r [h_{t-1}, x_t])

h_t = (1 - z_t) * h_{t-1} + z_t * tilde{h}_t

where:
- z_t is the update gate
- r_t is the reset gate
- tilde{h}_t is the candidate hidden state

### Why GRU is popular

- fewer parameters than LSTM
- faster to train in many settings
- still handles long-range dependencies well

### Tradeoff

LSTM has more explicit memory control; GRU has a simpler design and often works well with less complexity.

### Interview-ready explanation

"The GRU is a simplified LSTM. It keeps only two gates: update and reset. This reduces complexity while still controlling how much previous information influences the current state."

---

## Attention Mechanism

### Concept

Attention allows a model to focus on the relevant parts of an input sequence instead of compressing everything into one fixed hidden state.

### Why vanilla RNNs struggled

The final hidden state of an RNN may not capture all important information from a long sequence. Attention helps by giving the model access to all relevant previous states rather than only the last one.

### Core idea

For each query, the model computes scores against all keys and then attends to the most relevant values.

### Standard formulation

Scores: s = Q K^T / sqrt(d_k)

Weights: α = softmax(s)

Context: c = α V

where:
- Q is the query vector
- K is the key matrix
- V is the value matrix
- α is the attention weight distribution

### Intuition

In translation, when generating the next word in the target sentence, the model may pay more attention to the relevant words in the source sentence rather than treating all words equally.

### Why attention matters

- handles long-range dependencies better than RNNs
- allows dynamic weighting of context
- foundation for modern transformer models

### Interview-ready explanation

"Attention is the mechanism that lets a model decide which tokens or positions are most relevant at each step. It computes a weighted combination of relevant values, which enables better context handling than relying only on a fixed hidden state."

---

## Transformer Architecture

### Concept

The transformer is built entirely on attention, without recurrence. It processes the whole sequence in parallel and captures relationships among all tokens using self-attention.

### Why transformers changed NLP

Transformers solve several limitations of RNN-style models:

- no sequential bottleneck
- better long-range modeling
- highly parallelizable at training time
- excellent scalability to large datasets

### Core building blocks

1. Input embeddings
   - each token is mapped to a dense vector

2. Positional encoding
   - adds information about token order because attention alone is order-agnostic

3. Self-attention
   - each token attends to every other token in the sequence

4. Multi-head attention
   - multiple attention mechanisms run in parallel, allowing different representational subspaces

5. Feed-forward network
   - adds nonlinearity and transformation after attention

6. Residual connections + layer normalization
   - stabilize training and improve gradient flow

### Architecture sketch

Input tokens -> embeddings -> positional encoding -> multi-head self-attention -> feed-forward -> output

### Why this matters in practice

Transformers power:
- BERT
- GPT
- T5
- translation models
- question answering systems
- summarization systems

### Interview-ready explanation

"A transformer uses self-attention to compute relationships between every pair of tokens in a sequence. It does not rely on recurrence, which makes it easier to parallelize and more powerful for modeling long-range context in language and other sequential domains."

---

## NLP Tasks

### 1. Text classification

Examples:
- sentiment analysis
- spam detection
- topic classification

Input: text sequence
Output: class label

Typical architecture:
- embedding layer
- transformer encoder
- classification head

### 2. Machine translation

Examples:
- English to Bangla
- English to French

Input: source sentence
Output: translated sentence

Typical architecture:
- encoder reads source sequence
- decoder generates target sequence
- attention aligns source and target words

### 3. Question answering

Examples:
- reading comprehension
- retrieval QA
- conversational question answering

Input: passage + question
Output: answer span or generated response

Typical architecture:
- encode question and passage together
- attention finds relevant passage segments
- predict answer span or produce final text

### 4. Sequence-to-sequence problems

These include:
- translation
- summarization
- text generation
- dialogue systems

### Why these tasks matter

They are the clearest examples of where order, memory, and contextual understanding are critical.

---

## Coding Patterns

### RNN example

```python
import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
```

### LSTM example

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

### GRU example

```python
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
```

### Attention example

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, hidden_dim):
        super().__init__()
        self.query = nn.Linear(embed_size, hidden_dim)
        self.key = nn.Linear(embed_size, hidden_dim)
        self.value = nn.Linear(embed_size, hidden_dim)
        self.out = nn.Linear(hidden_dim, embed_size)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)
        return self.out(context)
```

### Typical transformer block

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff = self.feed_forward(x)
        return self.norm2(x + ff)
```

---

## Interview Questions

### RNN
- What is the hidden state in an RNN?
- Why do vanilla RNNs struggle with long-term dependencies?
- How does sequence order influence the model?

### LSTM
- What are the roles of the forget, input, and output gates?
- Why does LSTM work better than a standard RNN for long sequences?
- What is the difference between hidden state and cell state?

### GRU
- Why is GRU simpler than LSTM?
- When do you prefer GRU over LSTM?
- What is the update gate doing mathematically?

### Attention
- Why is attention important?
- What is the role of query, key, and value?
- How does attention solve the bottleneck problem in RNNs?

### Transformer
- Why is positional encoding necessary?
- What is self-attention?
- Why is multi-head attention useful?
- Why are transformers parallelizable?

### NLP tasks
- What is the difference between text classification and machine translation?
- How would you use attention in QA systems?
- Why are transformers dominating modern NLP benchmarks?

---

## Recommended Study Roadmap

### Step 1: Build the foundation
- understand RNN basics
- learn why sequence modeling matters
- identify the vanishing-gradient problem

### Step 2: Learn gated memory
- study LSTM
- study GRU
- compare their behavior with standard RNNs

### Step 3: Understand attention
- Q, K, V formulation
- align tokens with relevance weights
- see how attention improves context handling

### Step 4: Understand the transformer
- embeddings
- positional encoding
- self-attention
- residual connections and normalization

### Step 5: Apply to NLP tasks
- classification
- translation
- QA
- summarization

### Step 6: Practice explaining in interview language

A strong answer usually looks like this:

- What is the problem?
- Why do classical methods fail?
- What architecture is used and why?
- What is the key mathematical idea?
- What are the trade-offs?

---

## Final takeaway

Sequence models are the reason modern AI can understand language, speech, and temporal patterns. The actual progression is simple to remember:

- RNN gives memory
- LSTM gives controlled memory
- GRU gives simpler memory control
- Attention gives dynamic relevance weighting
- Transformer gives scalable, parallel, attention-based modeling

This is the conceptual foundation behind almost every modern NLP system and large language model.

---

## Next recommended directions

After finishing this section, continue into:

- LSTM implementation and comparison notebooks
- attention mechanism notebooks
- transformer architecture notebooks
- beyond: BERT, GPT, T5, and large language models

The ideas in this folder are not just academic; they are the direct basis of modern generative AI and language systems.
