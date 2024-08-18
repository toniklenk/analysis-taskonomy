""" Basic defintions & functions for analyses

"""

# python
import os, sys, pickle
from itertools import combinations_with_replacement, combinations, product
from collections import OrderedDict

# stats
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.api import OLS


DATASET_NAMES = ("places1", "places2", "oasis")
SCALE_NAMES = ("scale2", "scale4", "scale8", "scale16", "scale32")
STUDY_NAMES = ("short presentation", "long presentation", "complexity order")
BEHAVIOUR_NAMES = (
    "study1_places1_short.csv",
    "study2_places1.csv",
    "study3_places2.csv",
    "study4_oasis.csv",
)

# VisualPrior.viable_feature_tasks
MODEL_NAMES = [
    "autoencoding",
    "depth_euclidean",
    "jigsaw",
    "reshading",
    "edge_occlusion",
    "keypoints2d",
    "room_layout",  #'colorization' currently not working
    "curvature",
    "edge_texture",
    "keypoints3d",
    "segment_unsup2d",
    "class_object",
    "egomotion",
    "nonfixated_pose",
    "segment_unsup25d",
    "class_scene",
    "fixated_pose",
    "normal",
    "segment_semantic",
    "denoising",
    "inpainting",
    "point_matching",
    "vanishing_point",
]

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

NETS_ALL = NETS_SEMANTIC + NETS_2D + NETS_3D


def load_integration(
    path,
    models: list = MODEL_NAMES,
    datasets: list = DATASET_NAMES,
    scales: list = SCALE_NAMES,
):
    cl = []
    for mo, da, sc in product(models, datasets, scales):
        c = (
            pd.read_csv(
                os.path.join(path, mo, da, sc, "correlations.csv"),
                header=None,
            )
            .assign(scale=sc, dataset=da, model=mo)
            .reset_index()
            .rename({"index": "img"}, axis=1)
        )
        cl.append(c)

    # to integration
    dfi = -pd.concat(cl).set_index(["model", "dataset", "scale", "img"])
    dfi.columns.name = "layer"
    dfi = dfi.stack("layer").to_frame().rename({0: "integration"}, axis=1)
    return dfi


def load_ibcorr(
    path,
    models: list = MODEL_NAMES,
    studies: list = STUDY_NAMES,
    scales: list = SCALE_NAMES,
):
    cl = []
    for mo, st, sc in product(models, studies, scales):
        c = (
            pd.read_csv(
                os.path.join(path, mo, st, sc, "ib_correlations.csv"),
                header=None,
                names=["ibcorr"],
            )
            .assign(scale=sc, study=st, model=mo)
            .reset_index()
            .rename({"index": "layer"}, axis=1)
        )
        cl.append(c)

    dfibc = pd.concat(cl).set_index(["model", "study", "scale", "layer"])
    return dfibc


def load_pvalues(
    path,
    models: list = MODEL_NAMES,
    studies: list = STUDY_NAMES,
    scales: list = SCALE_NAMES,
):
    """Gets same path as ibcorr!"""
    pl = []
    for mo, st, sc in product(models, studies, scales):
        p = (
            pd.read_csv(
                os.path.join(path, mo, st, sc, "ib_correlations_pvalues.csv"),
                header=None,
                names=["pvalue"],
            )
            .assign(scale=sc, study=st, model=mo)
            .reset_index()
            .rename({"index": "layer"}, axis=1)
        )
        pl.append(p)

    dfp = pd.concat(pl).set_index(["model", "study", "scale", "layer"])
    return dfp


def load_ratings(path, behaviours = BEHAVIOUR_NAMES):
    beauty_ratings = {}
    for r in behaviours:

        data = (
            pd.read_csv(os.path.join(path, r), header=None)
            .mean(axis=1)
            .to_frame()
            .rename({0: "beauty rating"}, axis=1)
        )
        data.index.name = "img_id"

        # add name of study to index
        beauty_ratings[r] = pd.concat([data], names=["dataset"], keys=[r])
    
    return beauty_ratings