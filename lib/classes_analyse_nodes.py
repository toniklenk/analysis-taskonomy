from __future__ import annotations # this is not needed for the code to work, just for a type-hint in a class methods
from typing import Union # this is not needed for the code to work, just for the type-hint in the header

import os
import numpy as np
import pandas as pd
import pickle

import torch
from scipy.stats import pearsonr
from scipy.io import loadmat

from collections import OrderedDict
from dataclasses import dataclass

from scipy.stats import spearmanr
from taskonomy_network import TaskonomyDecoder, TaskonomyEncoder


class ImageDataset(object):
    """
    Handles preparing images for input into activation extractors:
        
        - Load images (matlab arrays) from subfolder,
            in alphanumerical order (corresponding to beauty ratings in file).
        
        - Transform into PyTorch format
    
    This class provides a iterator to do so.
    """
    def __init__(self, img_dir, beauty_ratings_path=None):

        dir_img_list    = list(f for f in os.listdir(os.path.join(img_dir, 'full')))
        self.img_dir    = img_dir
        self.img_list   = sorted(dir_img_list)
        self.img_count  = len(dir_img_list)
        if beauty_ratings_path is not None:
            self.beauty_ratings = pd.read_csv(beauty_ratings_path, header=None).mean(axis=1)

    def __iter__(self, transform = lambda x: x):
        self.img_pos = 0
        return self
    
    def __next__(self):
        if self.img_pos < self.img_count:
            # load arrays (transformed in matlab)
            img_full = loadmat(os.path.join(self.img_dir,'full', self.img_list[self.img_pos]))["im"]
            img_v1 = loadmat(os.path.join(self.img_dir,'version1', self.img_list[self.img_pos]))["imv1"]
            img_v2 = loadmat(os.path.join(self.img_dir,'version2', self.img_list[self.img_pos]))["imv2"]
            
            # convert to input format of Taskonomy models
            img_full = torch.tensor(img_full).permute([2, 0, 1]).unsqueeze(0)
            img_v1 = torch.tensor(img_v1).permute([2, 0, 1]).unsqueeze(0)
            img_v2 = torch.tensor(img_v2).permute([2, 0, 1]).unsqueeze(0)
            self.img_pos += 1
            return img_full, img_v1, img_v2
        else: # prepare for a possible next iteration
            self.img_pos = 0
            raise StopIteration


class Pattern_Generator(object):
    """
    Provides different subsets of nodes from activation pattern.

    - Same subsets of nodes across iteration for different images.
    - Takes care of different layers

    """
    def __init__(self, num_subsets: int, layer_shapes: OrderedDict, frac: float = .33) -> None:
        """
        Input:
            num_iterations: number of node subsets to be drawn (e.g. 10 000, 100 000)
            net: Taskonomy network where the activation patterns will be from
            frac: Fraction of nodes of a layer selected for each subset.
        """
        # random generator seed for each different subset
        self.num_subsets = num_subsets
        self.seeds = torch.randint(int(1E9), (num_subsets,))
        self.layer_shapes = layer_shapes # dict: keys: layer names; value: tensors with layer shape
        self.frac = frac

    def _generate_patterns(self, seed: int) -> OrderedDict:
        """
        Generates pattern for whole network
        with specified layer_shapes
        from the given seed
        """
        gen = torch.Generator().manual_seed(seed)
        return OrderedDict((name, torch.rand(shape, generator=gen) > (1-self.frac))
                           for name, shape in self.layer_shapes.items())
    
    def get_subset_pattern(self, subset_num):
        """Returns same subset of nodes each time it's called with the same subset num"""
        return self._generate_patterns(self.seeds[subset_num].item())


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


class NetworkScorer(object):
    """
    Maps back integration beauty score (from node subsets) onto nodes DNN.
    """

    def __init__(self, layer_shapes: OrderedDict, subset_iterations_start: int = 0) -> None:
        """Init from activation architecture
        
        subset_iterations_count: counts the number of subsets who's score has been
                                 backprojected into the self.scores of this NetworkScorer
                                 (use subset_iteration_start to set a start value if
                                  there is a exists data loaded into this object)
        """

        self.scores = OrderedDict((name, torch.zeros(shape,dtype=torch.float64))
                                  for name, shape in layer_shapes.items())
        
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
        with open(os.path.join(path, "subset_iteration_count.pkl"), 'wb') as file:
            pickle.dump(self.subset_iterations_count, file)
        
        # save scores
        for layer_name, layer_scores in self.scores.items():
            torch.save(layer_scores, os.path.join(path, layer_name + '.pt'))
    
    @classmethod
    def load(cls, path: str):
        # load subset_iteration count
        with open(os.path.join(path, "subset_iteration_count.pkl"), 'rb') as file:
            subset_iterations_count = pickle.load(file)
        
        # load scores
        score_filenames = sorted(filter(
            lambda filename: filename.endswith('.pt'),
            os.listdir(path)))
        
        # get layer names from filenames
        layer_names = [filename[:-len('.pt')]
                       for filename in score_filenames]
        
        # init empty NetworkScorer
        ns = cls(OrderedDict(), subset_iterations_start=subset_iterations_count)
        
        # load scores into NetworkScorer
        for layerfile_name, layer_name in zip(score_filenames, layer_names):
            ns.scores[layer_name] = torch.load(os.path.join(path, layerfile_name))
        
        return ns




def calculate_integration_coeff(full: Activation_Pattern, avg: Activation_Pattern):
    """
    Calculate integration coeff in every layer from two acivation patterns:
    - One is the activation to the full image and one the average of the two halfes.
    - Returns a np.ndarray with shape (layers,) and the correlation coefficients

    """
    return np.array([
        -pearsonr(layer_full.flatten(), layer_avg.flatten())[0]
        for layer_full, layer_avg
        in zip(full.activation_pattern.values(), avg.activation_pattern.values())])


def taskonomy_net_layer_shapes(net: Union[TaskonomyEncoder, TaskonomyDecoder]) -> OrderedDict:
    """
    Creates a dictionary with the shapes of
    all the convolutional and fully connected layers of
    a Taskonomy network 
    """
    return OrderedDict((name,layer.data.shape)
            for name, layer in net.named_parameters()
            if "conv" in name or 'fc' in name)


def taskonomy_activation_layer_shapes(net_activation: OrderedDict) -> OrderedDict:
    """
    Creates a dictionary with the shapes of
    all the convolutional and fully connected layers of
    a Taskonomy network 
    """
    return OrderedDict((name, layer_activation.shape)
                       for name, layer_activation in net_activation.items())

def correlate_integration_beauty(integration: np.ndarray, beauty_ratings: pd.Series):
    return np.apply_along_axis(lambda c: spearmanr(c, beauty_ratings)[0], 1, integration)

def calculate_dataset_metrics(ImageDataset_iterator, net):
    """Calculate metrics for whole dataset"""

    def calculate_image_metrics(net, img_full, img_v1, img_v2):
        """Calculate correlation coefficients for all layers between full and average activation pattern"""
        """Calculate image self-similarity"""
        """Calculate imgage L2-norm"""

        # activations for full image and image parts
        with torch.no_grad():
            act_full, act_v1, act_v2 = net(img_full), net(img_v1), net(img_v2)

        correlations, selfsimilarity, l2norm = {}, {}, {}

        for (layer, act_full_, act_v1_, act_v2_) in zip(act_full.keys(), act_full.values(), act_v1.values(), act_v2.values()):
            # average activation for image parts
            act_avg_ = torch.stack((act_v1_, act_v2_), dim=0).mean(dim=0).flatten()
            
            l2norm[layer] = act_full_.norm(p=2).item()

            act_v1_ = act_v1_.flatten()
            act_v2_ = act_v2_.flatten()
            act_full_ = act_full_.flatten()

            correlations[layer] = pearsonr(act_full_, act_avg_)[0]

            selfsimilarity[layer] = pearsonr(act_v1_, act_v2_)[0]


        return correlations, selfsimilarity, l2norm


    lst_correlation, lst_selfsimilarity, lst_l2norm = [],[],[]

    for img_full, img_v1, img_v2 in ImageDataset_iterator:
        correlation, selfsimilarity, l2norm = calculate_image_metrics(net, img_full, img_v1, img_v2)

        lst_correlation.append(correlation)
        lst_selfsimilarity.append(selfsimilarity)
        lst_l2norm.append(l2norm)
    
    column_names = list(net(torch.zeros(1,3,256,256)).keys())
    return pd.DataFrame(lst_correlation, columns=column_names), pd.DataFrame(lst_selfsimilarity, columns=column_names), pd.DataFrame(lst_l2norm, columns=column_names)