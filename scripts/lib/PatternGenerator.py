# type-hinting
from __future__ import annotations

# python
from collections import OrderedDict

# stats
import torch


class Pattern_Generator(object):
    """
    Provides different subsets of nodes from activation pattern.

    - Same subsets of nodes across iteration for different images.
    - Takes care of different layers

    """

    def __init__(
        self, num_subsets: int, layer_shapes: OrderedDict, frac: float = 0.33
    ) -> None:
        """
        Input:
            num_iterations: number of node subsets to be drawn (e.g. 10 000, 100 000)
            net: Taskonomy network where the activation patterns will be from
            frac: Fraction of nodes of a layer selected for each subset.
        """
        # random generator seed for each different subset
        self.num_subsets = num_subsets
        self.seeds = torch.randint(int(1e9), (num_subsets,))
        self.layer_shapes = (
            layer_shapes  # dict: keys: layer names; value: tensors with layer shape
        )
        self.frac = frac

    def _generate_patterns(self, seed: int) -> OrderedDict:
        """
        Generates pattern for whole network
        with specified layer_shapes
        from the given seed
        """
        gen = torch.Generator().manual_seed(seed)
        return OrderedDict(
            (name, torch.rand(shape, generator=gen) > (1 - self.frac))
            for name, shape in self.layer_shapes.items()
        )

    def get_subset_pattern(self, subset_num):
        """Returns same subset of nodes each time it's called with the same subset num"""
        return self._generate_patterns(self.seeds[subset_num].item())
