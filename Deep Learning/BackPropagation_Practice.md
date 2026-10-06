# BackPropagation Practice

## Why this notebook matters
Backpropagation is the engine behind learning in neural networks. It tells the model how to change its weights so that the error decreases.

## Core idea
The model computes a loss, then propagates the error backward through the network. Each weight is updated in proportion to the gradient of the loss with respect to that weight.

## Mathematical intuition
If the loss is L and a weight is w, then:

w <- w - \eta \frac{\partial L}{\partial w}

where \eta is the learning rate.

This is gradient descent in action. The chain rule allows the network to propagate the error from the final layer to earlier ones.

## Intuition
If the network predicts badly, backpropagation answers: "Which weights should be adjusted, and by how much?" The answer comes from local sensitivity of the loss to each parameter.

## Practical workflow in the notebook
- define a simple network
- compute forward pass
- calculate loss
- backpropagate gradients
- update weights manually

## Interview-ready explanation
"Backpropagation applies the chain rule recursively to compute gradients for each parameter. It is how neural networks learn by reducing error across layers."

## Common pitfalls
- incorrect derivative sign conventions
- debugging by looking only at the final layer
- ignoring learning rate effects

## Takeaway
This notebook is essential because it turns the abstract idea of learning into a concrete computational process. If you understand backpropagation, most of neural network optimization becomes intuitive.
