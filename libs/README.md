# GenT external libraries

These libraries are external dependencies that are used by GenT. 
We did not develop these libraries from scratch, but we made some compatability changes to make them suitable with GenT.

## CTGAN
This is a private and detached fork from https://github.com/sdv-dev/CTGAN/commit/2848a42651c440a01a987a27658722069c7bb9b1.

The core changes are:
* Added new functionality `continue_fit` to the `CTGAN` class
* Added keyword arguments to the constructor of `CTGAN`: `graph_index_to_edges, n_nodes, functional_loss, functional_loss_freq, with_gcn: bool, device: torch.device, name: str`
* Changed the logic of `sample` and adding `sample_one` to support the conditioned sampling that we need in GenT 
* Added the new `Noise` class in `ctgan/synthesizers/ctgan.py`
