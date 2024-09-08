# type-hinting
from __future__ import annotations

# python
from collections import OrderedDict
from dataclasses import dataclass

# stats
import numpy as np
from scipy.stats import pearsonr


@dataclass(frozen=True)
class Activation_Pattern(object):
    """
    Handles activation pattern of a network to an image

    - Takes care of layers
    - Selects subset of nodes (takes pattern from class Pattern_Generator)
    - Calculates integration values for different layers
    """

    activation_pattern: OrderedDict

    def __getitem__(self, layer_masks: OrderedDict):
        """
        Mask whole network with dict specifying tensor mask for each layer (as returned by class: Pattern_Generator)
        """
        return Activation_Pattern(
            OrderedDict(
                (layer_name, layer_activation[layer_mask])
                for (layer_name, layer_activation), layer_mask in zip(
                    self.activation_pattern.items(), layer_masks.values()
                )
            )
        )

    @staticmethod
    def average(first: Activation_Pattern, second: Activation_Pattern):
        """
        Calculate average of two activation patterns
        """
        return Activation_Pattern(
            OrderedDict(
                (layer_name, (act_first + act_second) / 2.0)
                for (layer_name, act_first), act_second in zip(
                    first.activation_pattern.items(), second.activation_pattern.values()
                )
            )
        )

    @staticmethod
    def calculate_integration_coeff(full: Activation_Pattern, avg: Activation_Pattern):
        """
        Calculate integration coeff in every layer from two acivation patterns:
        - One is the activation to the full image and one the average of the two halfes.
        - Returns a np.ndarray with shape (layers,) and the correlation coefficients

        """
        return np.array(
            [
                -pearsonr(layer_full.flatten(), layer_avg.flatten())[0]
                for layer_full, layer_avg in zip(
                    full.activation_pattern.values(), avg.activation_pattern.values()
                )
            ]
        )
