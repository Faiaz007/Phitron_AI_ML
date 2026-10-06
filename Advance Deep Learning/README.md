# Advance Deep Learning

This folder focuses on sequence modeling and advanced deep learning ideas. It is the transition from feedforward neural networks to models built for temporal and contextual information.

## Core focus

- recurrent neural networks
- sequence learning
- context-aware question answering
- temporal memory and hidden states
- NLP-like modeling patterns

## Table of contents

- [GK_Answering_system_using_RNN.ipynb](./GK_Answering_system_using_RNN.ipynb)

## Learning path

### 1. Concept
A recurrent neural network processes sequences one step at a time, carrying forward information through a hidden state. This allows the model to incorporate earlier context when making later predictions.

### 2. Intuition
RNNs are like reading a sentence word by word: each new token updates your understanding. This is useful when the next output depends on the history of previous inputs.

### 3. Math
The main recurrence is:

```text
h_t = f(W_x x_t + W_h h_{t-1} + b)
```

Where:
- x_t is the current input
- h_t is the current hidden state
- h_{t-1} is prior memory
- W_x and W_h are learned parameters

### 4. Coding pattern
Typical RNN pipeline:

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

### 5. Common interview questions
- What is the role of the hidden state in an RNN?
- Why do RNNs struggle with long-term dependencies?
- What is the difference between an RNN and a feedforward network?
- Why are LSTMs and GRUs used in practice?
- How is sequence length handled in training and inference?

## Interview cheat sheet

- RNNs are designed for ordered data.
- They use a hidden state to carry memory from one time step to the next.
- They are useful for text, time series, and other sequential tasks.
- Long sequences cause vanishing gradients, which motivates LSTM/GRU.
- The final state or all states can be used for prediction depending on the task.

## Recommended next steps

After this folder, continue into:

- LSTM and GRU architectures
- attention and transformers
- NLP tasks such as classification, translation, and Q&A
