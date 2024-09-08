# type-hinting
from __future__ import annotations

# python
import os, pickle
from collections import OrderedDict

# stats
import torch
import numpy as np

# classes
from lib.PatternGenerator import Pattern_Generator


class NetworkScorer(object):
    """
    Maps back integration beauty score (from node subsets) onto nodes DNN.
    TODO: make this a simpler class by reducing its functionality to support only one layer,
    if needed, as class inhering from this containing multiple layers can be built later
    """

    def __init__(
        self, layer_shapes: OrderedDict, subset_iterations_start: int = 0
    ) -> None:
        """Init from activation architecture

        subset_iterations_count: counts the number of subsets who's score has been
                                 backprojected into the self.scores of this NetworkScorer
                                 (use subset_iteration_start to set a start value if
                                  there is a exists data loaded into this object)
        """

        self.scores = OrderedDict(
            (name, torch.zeros(shape, dtype=torch.float64))
            for name, shape in layer_shapes.items()
        )

        self.subset_iterations_count = subset_iterations_start

    def map_back_scores(self, scores: np.ndarray, patterns: Pattern_Generator):
        """map back scores
        from all subsets in scores (layer x subset)
        to all layers and of network

        The score can eiter be the integration-beauty correlation (score) of the whole dataset
        or the pure integration of a single image.
        """

        self.subset_iterations_count += scores.shape[1]

        # iterate subsets
        for idx_subset, scores_subset in enumerate(scores.T):
            pattern = patterns.get_subset_pattern(idx_subset)

            # iterate layers
            for layer_name, layer_score in zip(self.scores.keys(), scores_subset):
                # mask nodes, add score
                self.scores[layer_name][pattern[layer_name]] += layer_score

    def save(self, path: str):
        # save subset_iteration_count
        with open(os.path.join(path, "subset_iteration_count.pkl"), "wb") as file:
            pickle.dump(self.subset_iterations_count, file)

        # save scores
        for layer_name, layer_scores in self.scores.items():
            torch.save(layer_scores, os.path.join(path, layer_name + ".pt"))

    @classmethod
    def load(cls, path: str):
        # load subset_iteration count
        with open(os.path.join(path, "subset_iteration_count.pkl"), "rb") as file:
            subset_iterations_count = pickle.load(file)

        # load scores
        score_filenames = sorted(
            filter(lambda filename: filename.endswith(".pt"), os.listdir(path))
        )

        # get layer names from filenames
        layer_names = [filename[: -len(".pt")] for filename in score_filenames]

        # init empty NetworkScorer
        ns = cls(OrderedDict(), subset_iterations_start=subset_iterations_count)

        # load scores into NetworkScorer
        for layerfile_name, layer_name in zip(score_filenames, layer_names):
            ns.scores[layer_name] = torch.load(os.path.join(path, layerfile_name))

        return ns
