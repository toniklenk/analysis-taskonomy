# python
import os, pickle
from collections import OrderedDict
from dataclasses import dataclass

# type-hinting
from typing import Union
from __future__ import annotations

# stats
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# classes
from ActivationPattern import Activation_Pattern

# neural networks
from .taskonomy_network import TaskonomyDecoder, TaskonomyEncoder



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