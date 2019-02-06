# Linear Algebra for Deep Learning

This lecture is mainly an introduction into the notation that will be used for the rest of the course.

### Tensors

Datasets typically have more than one or two features, so a basic understanding of tensors is necessary. Apologies to any mathematicians or physicists that read these notes, as we often disregard notation and formalism. 

We reserve `X` to denote the __design matrix__, or the matrix containing the traning examples and features. We typically have $$\mathbf { X } \in \mathbb { R } ^ { n \times m }$$. 

It is typically advantageous (not only for simplicity, but also for storage, to treat rank-3 tensors as stacked matrices

![Rank 3 tensor](./images/tensor_visualization.png "Tensor visualization")

This scales to rank $n$ tensors (mainly for storage). 

#### Tensors in Python Modules

A `numpy.ndarray` is pretty much equivalent to a `pytorch.Tensor`. There are similar defaults for both modules, but `PyTorch` tends to be more finnicky about some type casting that `numpy` just does implicitly (ie adding `floats` to `ints`):

![Datatype table](./images/pytorch_datatypes.png "Datatype table")

### `PyTorch` seems like `numpy`, so what's the point?
* `PyTorch` plays nicely with GPUs!
* `PyTorch` has automatic differentiation
* `PyTorch` implements a lot of DL convenience functions (like convolutional layers and such)

If your computer has GPUs, it's easy to load data on to said GPU, 

```python
a = a.to(torch.device('cuda:0'))
# pass back to CPU
a = a.to(torch.device('cpu'))
```

`CUDA` is the library for Deep Learning stuff on NVIDIA GPUs. 