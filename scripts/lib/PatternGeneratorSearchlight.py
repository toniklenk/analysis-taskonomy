# type-hinting
from __future__ import annotations

# python
from collections import OrderedDict

# stats
import torch


class PatternGeneratorSearchlight(object):
    """
    Provides patterns for searchlight analysis.

    - Same subsets of nodes across iteration for different images.
    - Takes care of different layers

    """

    def __init__(self, layer_shape: torch.Size, layer_name: str) -> None:
        """
        Input:
            num_iterations: number of node subsets to be drawn (e.g. 10 000, 100 000)
            net: Taskonomy network where the activation patterns will be from
            frac: Fraction of nodes of a layer selected for each subset.
        """
        # random generator seed for each different subset
        self.current_subset = 0
        self.num_subsets = torch.tensor(
            [dimsize - 2 for dimsize in layer_shape]
        ).cumprod(dim=0)[-1]
        self.layer_shape = layer_shape
        self.layer_name = layer_name

    @staticmethod
    def _posnum_to_3Dindex(posnum, layer_shape: torch.Size):
        """Get 3D coordinates from position number
        IMPORTANT: only for use with a 3x3x3 sliding window (because of the offset 1)
        """
        # get position
        d1, d2, d3 = layer_shape[-3], layer_shape[-2], layer_shape[-1]

        first_idx = (posnum // ((d2 - 2) * (d3 - 2))) + 1
        second_index = ((posnum % ((d2 - 2) * (d3 - 2))) // (d3 - 2)) + 1
        third_index = (posnum % (d3 - 2)) + 1

        # d1, d2, d3
        return first_idx, second_index, third_index

    def index_to_mask(self, idx: tuple):
        mask = torch.zeros(self.layer_shape, dtype=torch.bool)
        mask[
            idx[0] - 1 : idx[0] + 2, idx[1] - 1 : idx[1] + 2, idx[2] - 1 : idx[2] + 2
        ] = 1
        return mask

    def __iter__(self):
        self.current_subset = 0
        return self

    def __next__(self):
        if self.current_subset < self.num_subsets:
            # return 3D searchlight pattern
            idx = PatternGeneratorSearchlight._posnum_to_3Dindex(
                self.current_subset, self.layer_shape
            )
            self.current_subset += 1
            return self.current_subset - 1, self.index_to_mask(idx)

        else:
            self.current_subset = 0
            raise StopIteration

    def get_subset_pattern(self, subset_num: int):
        """Provide same interface as default Pattern_Generator class"""
        return OrderedDict(
            [
                (
                    self.layer_name,
                    self.index_to_mask(
                        PatternGeneratorSearchlight._posnum_to_3Dindex(
                            subset_num, self.layer_shape
                        )
                    ),
                )
            ]
        )
