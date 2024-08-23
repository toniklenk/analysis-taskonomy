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


NETS_SEMANTIC = ["class_object", "class_scene", "segment_semantic"]

# from radek paper missing: colorization (not downloadable from taskonomy)
NETS_2D = [
    "autoencoding",
    "denoising",
    "edge_texture",
    "inpainting",
    "keypoints2d",
    "segment_unsup2d",
]

# from radek paper missing: z-depth (missing from importing as well) and distance (but this is not a network after all)
NETS_3D = [
    "edge_occlusion",
    "keypoints3d",
    "segment_unsup25d",
    "reshading",
    "normal",
    "curvature",
]


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


def flatten_concat_rdms(rdms: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Take a dict of rdms as DataFrames, extract the upper triangle (without diagonal, as appropriate for rms),
    concat them into columns a single df and use dict keys as colnames, to prepare for use with calculate_rdm

    """

    example_rdm = rdms[list(rdms.keys())[0]]
    mask = np.triu(np.ones_like(example_rdm.values).astype(np.bool_), k=1)
    d = {mo: pd.Series(rdm.values[mask]) for mo, rdm in rdms.items()}
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


def compare_corr(r_yx1, r_yx2, X1, X2, correlation="pearson") -> float:
    """Compares two correlation coefficients that are non-independent because they're based on the same sample
    and returns a Z-statistic to perform a hypothesis test.

    This method is described in Meng, Rosenthal, Rubin; 1992; Psych. Bulletin

    """
    if correlation == "pearson":
        rx = pearsonr(X1, X2)[0]
    elif correlation == "spearman":
        rx = spearmanr(X1, X2)[0]

    r2bar = (r_yx1**2 + r_yx2**2) / 2
    f = (1 - rx) / (2 * (1 - r2bar))
    h = (1 - f * r2bar) / (1 - r2bar)
    N = len(X1)

    # fisher z-transform
    z_yx1, z_yx2 = np.arctanh(r_yx1), np.arctanh(r_yx2)

    Z = (z_yx1 - z_yx2) * np.sqrt((N - 3) / (2 * rx * h))

    return Z


def models_best_predicting_integration_from_block(
    block_num: int,
    df_model_ibcorr: pd.Series,
    df_model_integration,
    block_layer_mapping,
):
    """Returns layer num (nums starting with 0) from a block.

    ---
    block_num: equals the layer num in the blocked layers

    """
    mask = block_layer_mapping == block_num
    idx = df_model_ibcorr[mask].idxmax()
    return df_model_integration[idx]


def modelname2class(model_name):
    if model_name in NETS_SEMANTIC:
        return "semantic"
    elif model_name in NETS_2D:
        return "2d"
    elif model_name in NETS_3D:
        return "3d"
