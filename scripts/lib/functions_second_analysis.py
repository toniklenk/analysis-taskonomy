# type-hinting
from __future__ import annotations
from typing import Union, Dict

# python
import os, pickle
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations_with_replacement

# stats
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# classes
from .ActivationPattern import Activation_Pattern

# neural networks
from .taskonomy_network import TaskonomyDecoder, TaskonomyEncoder


def taskonomy_net_layer_shapes(
    net: Union[TaskonomyEncoder, TaskonomyDecoder]
) -> OrderedDict:
    """
    Creates a dictionary with the shapes of
    all the convolutional and fully connected layers of
    a Taskonomy network
    """
    return OrderedDict(
        (name, layer.data.shape)
        for name, layer in net.named_parameters()
        if "conv" in name or "fc" in name
    )


def taskonomy_activation_layer_shapes(net_activation: OrderedDict) -> OrderedDict:
    """
    Creates a dictionary with the shapes of
    all the convolutional and fully connected layers of
    a Taskonomy network
    """
    return OrderedDict(
        (name, layer_activation.shape)
        for name, layer_activation in net_activation.items()
    )


def correlate_integration_beauty(integration: np.ndarray, beauty_ratings: pd.Series):
    return np.apply_along_axis(
        lambda c: spearmanr(c, beauty_ratings)[0], 1, integration
    )


def flatten_concat(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Take a dict of DataFrames, flatten them,
    concat them into columns a single df and use dict keys as colnames

    """

    d = {key: pd.Series(df.values.flatten()) for key, df in d.items()}
    df = pd.concat(d.values(), axis=1)
    df.columns = d.keys()
    return df


def calculate_rdm(data: pd.DataFrame, correlation_type: str = "pearson"):
    """Calculate RDM with pearson/spearman correlation for every combination of columns

    Parameters
    ----------
    data: pd.DataFrame
        Input with data to correlate in the columns

    correlation_type: str
        Which correlation to use. "pearson" (default) or "spearman".


    Returns
    -------
    pd.DataFrame
        representational dissimilarity matrix of inputs' columns

    """
    num_columns = data.shape[1]

    # create empty matrix to store RDM
    # index and column labels are in order of input columns
    rdm = pd.DataFrame(
        np.full((num_columns, num_columns), np.nan),
        columns=data.columns,
        index=data.columns,
    )

    for col1, col2 in combinations_with_replacement(data.columns, 2):
        # there's one NaN in the autoencoding integration values, filter this here, don't know why that happens
        co11_col2 = data[[col1, col2]].dropna()

        # calculate correlation
        if correlation_type == "pearson":
            corr = pearsonr(co11_col2.values[:, 0], co11_col2.values[:, 1])[0]
        elif correlation_type == "spearman":
            corr = spearmanr(co11_col2.values[:, 0], co11_col2.values[:, 1])[0]

        # fill upper and lower triangular matrix
        rdm.loc[col1, col2] = corr
        rdm.loc[col2, col1] = corr
        rdm.loc[col1, col1] = 0.0

    return rdm
