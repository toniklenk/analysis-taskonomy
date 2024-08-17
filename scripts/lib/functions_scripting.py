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
MODEL_NAMES = (
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
)


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
