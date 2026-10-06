# Dataset and Dataloader

## Why this notebook matters
In deep learning, raw data alone is not enough. We need a pipeline that prepares examples efficiently and feeds them to the model in a stable, scalable way.

## Core idea
A dataset object defines how samples are stored and accessed. A dataloader organizes them into mini-batches, shuffles them, and handles parallel loading.

## Mathematical intuition
Training is done on batches rather than the whole dataset. This yields:

- more stable gradients
- lower memory usage
- faster updates
- more regularization through stochasticity

If the dataset is D and batch size is B, then each optimization step sees a subset of size B rather than all D samples.

## Intuition
A dataloader is like a conveyor belt for training. It keeps the model supplied with samples while the network learns incrementally.

## Practical workflow in the notebook
- create dataset class or use PyTorch utilities
- load data samples
- transform features and labels
- define batch size and shuffle behavior
- iterate through mini-batches

## Interview-ready explanation
"The dataset and dataloader separate data management from model learning. They improve efficiency, support randomized training, and help scale learning to large data sources."

## Common pitfalls
- forgetting to normalize inputs
- misusing label shapes
- setting batch size too large for memory
- not shuffling training data when needed

## Takeaway
This is one of the most practical topics in deep learning because it turns raw data into a trainable optimization loop.
