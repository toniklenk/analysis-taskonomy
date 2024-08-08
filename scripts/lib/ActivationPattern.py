# type-hinting
from __future__ import annotations

# python
from collections import OrderedDict
from dataclasses import dataclass




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
            OrderedDict((layer_name,layer_activation[layer_mask])
                        for (layer_name, layer_activation), layer_mask
                        in zip(self.activation_pattern.items(), layer_masks.values())))
    
    @staticmethod
    def average(first: Activation_Pattern, second: Activation_Pattern):
        """
        Calculate average of two activation patterns
        """
        return Activation_Pattern(
            OrderedDict((layer_name, (act_first + act_second) / 2.)
                        for (layer_name, act_first), act_second
                        in zip(first.activation_pattern.items(), second.activation_pattern.values())))