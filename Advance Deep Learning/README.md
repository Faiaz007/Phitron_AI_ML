# Advance Deep Learning

This folder introduces sequence modeling and advanced neural architectures. It is where the repository moves from feedforward networks to models designed for temporal structure, memory, and context-aware reasoning.

## Objective

The goal of this section is to understand how models process ordered data and how modern architectures capture long-range dependencies, contextual relationships, and language semantics.

## Core topics

- recurrent neural networks (RNN)
- long short-term memory (LSTM)
- gated recurrent units (GRU)
- attention mechanisms
- transformer architectures
- NLP tasks such as classification, translation, and question answering

## Notebook index

- [GK_Answering_system_using_RNN.ipynb](./GK_Answering_system_using_RNN.ipynb)

---

## Concept

A recurrent neural network processes time-ordered inputs one step at a time. It maintains a hidden state that carries information from earlier tokens into later decisions.

This is essential in tasks such as:
- language modeling
- machine translation
- sentiment analysis
- question answering
- time-series forecasting

## Intuition

Think of reading a sentence word by word. At each step, the model updates its internal understanding based on the current word and the memory from prior words. This allows the model to interpret context and dependencies across positions.

## Math

The core RNN recurrence is:

h_t = f(W_x x_t + W_h h_{t-1} + b)

Where:
- x_t is the current input
- h_{t-1} is the previous hidden state
- W_x and W_h are learnable parameters
- f is a nonlinear activation function

This creates a temporal memory effect that allows the model to use earlier information when predicting later outputs.

## Coding pattern

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

This model reads a sequence, produces hidden states for each step, and uses the last hidden state to generate the final prediction.

## Architectural design of sequence models

### RNN architecture
- input sequence enters step by step
- hidden state carries previous context
- output can be generated at each step or at the end

### LSTM architecture
LSTM improves on RNN by adding gates:
- forget gate
- input gate
- output gate
- cell state

This allows the model to decide what information to forget, preserve, and expose.

### GRU architecture
GRU is a simplified version of LSTM:
- fewer gates
- fewer parameters
- often faster and easier to train

### Attention and Transformer architecture
Attention allows the model to weigh the importance of different tokens dynamically. Instead of relying only on recurrence, the model can directly reason about which context is most relevant.

A transformer block typically contains:
- input embedding
- positional encoding
- multi-head self-attention
- feed-forward network
- residual connections and normalization

This architecture is the basis of large language models and modern NLP systems.

## Detailed comparison

### RNN
Pros:
- naturally handles ordered inputs
- conceptually simple
- useful for short sequences

Cons:
- struggles with long-term dependencies
- suffers from vanishing gradients

### LSTM
Pros:
- strong memory control
- better for long-range dependence
- widely used in sequence tasks

Cons:
- more parameters
- slower to train than GRU in some settings

### GRU
Pros:
- simpler than LSTM
- fewer parameters
- often efficient and effective

Cons:
- slightly less expressive in some tasks

### Attention / Transformer
Pros:
- capture dependencies across the entire sequence
- parallelizable during training
- foundational for modern NLP and multimodal models

Cons:
- computationally heavier at scale
- more architectural complexity

## NLP tasks related to this area

### Text classification
- sentiment analysis
- spam detection
- topic labeling

### Machine translation
- map one language sequence to another language sequence
- encoder-decoder framework is standard

### Question answering
- read passage and answer question using relevant context
- attention helps localize the answer span

### Sequence-to-sequence modeling
- translation, summarization, script generation, and dialogue tasks

## Common interview questions

- What is the difference between RNN, LSTM, and GRU?
- Why do RNNs struggle with long sequences?
- What is the role of the forget gate in LSTM?
- Why is attention important in sequence models?
- What problem does the transformer solve compared with RNNs?
- How does a transformer handle positional information?
- Which architecture is commonly used for NLP today and why?

## Interview cheat sheet

- RNNs use hidden state to remember past inputs
- LSTMs add gates to control memory and forgetting
- GRUs simplify LSTM but retain strong performance
- Attention lets the model focus on the most relevant tokens
- Transformers use self-attention instead of recurrence
- Modern NLP is dominated by transformers because they scale better and model long-range dependencies more effectively

## Recommended next steps

After this section, continue with:

- LSTM and GRU implementation notebooks
- attention mechanisms and contextual alignment
- transformer architecture and encoder-decoder design
- NLP tasks such as classification, translation, and question answering

This is the conceptual bridge between classical deep learning and modern large language models.
