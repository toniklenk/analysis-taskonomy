# type-hinting
from __future__ import annotations
from typing import Union, Dict, List

# python
import os, pickle
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations

# stats
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.api import OLS

# classes
from .ActivationPattern import Activation_Pattern

# neural networks
from .taskonomy_network import TaskonomyDecoder, TaskonomyEncoder
from lib.transforms import VisualPriorRepresentation
from torchvision.models.feature_extraction import (
    get_graph_node_names,
    create_feature_extractor,
)


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


# setup nets for extracting activations from best layer
def setup_singlelayer(model_name: str, layer_idx: int):
    """Setup activation extractor for a single layer of a tasnomomy network"""
    VisualPriorRepresentation._load_unloaded_nets([model_name])
    net = VisualPriorRepresentation.feature_task_to_net[model_name]

    _, eval_nodes = get_graph_node_names(net)
    return_nodes = {node: node for node in eval_nodes if "conv" in node or "fc" in node}

    layer_name = list(return_nodes.keys())[layer_idx]
    return (
        create_feature_extractor(net, return_nodes={layer_name: layer_name}),
        layer_name,
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


def rdm2vec(rdm):
    mask = np.triu(np.ones_like(rdm.values).astype(np.bool_), k=1)
    return rdm.values[mask]


def flatten_concat_rdms(rdms: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Take a dict of rdms as DataFrames, extract the upper triangle (without diagonal, as appropriate for rdms),
    concat them into columns a single df and use dict keys as colnames, to prepare for use with calculate_rdm

    """

    example_rdm = rdms[list(rdms.keys())[0]]
    mask = np.triu(np.ones_like(example_rdm.values).astype(np.bool_), k=1)
    d = {mo: pd.Series(rdm.values[mask]) for mo, rdm in rdms.items()}
    df = pd.concat(d.values(), axis=1)
    df.columns = d.keys()
    return df


def calculate_rdm(d: pd.DataFrame, ctype: str = "pearson", pval=False):
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
    if ctype not in ("pearson", "spearman", "absdiff"):
        raise ValueError("Kein gültiger Wert für ctype")

    rdm = pd.DataFrame(
        0,
        columns=d.columns,
        index=d.columns,
    )

    for col1, col2 in combinations(d.columns, 2):
        # there's one NaN in the autoencoding integration values, filter this here, don't know why that happens
        co11_col2 = d[[col1, col2]].dropna()

        if ctype == "pearson":
            c = pearsonr(co11_col2.values[:, 0], co11_col2.values[:, 1])
        elif ctype == "spearman":
            c = spearmanr(co11_col2.values[:, 0], co11_col2.values[:, 1])
        elif ctype == "absdiff":
            c = co11_col2.diff(axis=1).iloc[:, -1].abs().sum()

        if ctype in ("pearson", "spearman"):
            if pval:
                c = c[1]
            else:
                c = c[0]

        rdm.loc[col1, col2], rdm.loc[col2, col1] = c, c

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


def rdm2vec(rdm):
    mask = np.triu(np.ones_like(rdm.values).astype(np.bool_), k=1)
    return rdm.values[mask]


def correlate_rdms(rdm1, rdm2, correlation="pearson"):
    if type(rdm1) != pd.DataFrame or type(rdm2) != pd.DataFrame:
        raise TypeError("both RDMs need to be DataFrames")

    if correlation == "pearson":
        return pearsonr(rdm2vec(rdm1), rdm2vec(rdm2))

    if correlation == "spearman":
        return spearmanr(rdm2vec(rdm1), rdm2vec(rdm2))

    raise ValueError(
        "corrrelate_rdm muss für correlation pearson oder spearman bekommen"
    )


# variance partitioning
def predictors_r2(predictors: List[pd.DataFrame], target):
    predictors = np.stack([rdm2vec(_rdm).transpose() for _rdm in predictors], axis=1)
    predictors = sm.add_constant(predictors)
    model = sm.OLS(rdm2vec(target), predictors)
    results = model.fit()
    return results.rsquared
