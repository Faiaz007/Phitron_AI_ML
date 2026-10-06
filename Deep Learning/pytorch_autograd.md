# PyTorch Autograd

## Why this notebook matters
Autograd is the mechanism that automatically computes gradients. This is one of the most important reasons PyTorch is widely used for deep learning.

## Core idea
When you perform operations on tensors with requires_grad=True, PyTorch records the computation graph. During backward(), it computes derivatives automatically.

## Mathematical intuition
If y = f(x), then autograd computes dy/dx by backpropagating through the graph. This saves developers from manually deriving and coding gradients.

For a scalar loss L, the gradient is:

\nabla_x L = \frac{\partial L}{\partial x}

## Intuition
Autograd turns differentiation into a software service. The network writes down its computations, and optimization takes care of the derivative work.

## Practical workflow in the notebook
- set requires_grad on tensors
- perform operations
- call backward()
- inspect gradient values

## Interview-ready explanation
"Autograd is PyTorch's automatic differentiation engine. It records the computational graph and computes gradients efficiently, which enables backpropagation in neural networks."

## Common pitfalls
- calling backward twice without resetting gradients
- expecting gradients for non-scalar outputs
- confusing gradient accumulation with parameter updates

## Takeaway
This is the core automation that makes deep learning practical. Without autograd, training models would be far more cumbersome and error-prone.
