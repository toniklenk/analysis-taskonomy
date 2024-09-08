""" Basic defintions & functions for analyses

"""

# python
import os
from itertools import product

# stats
import numpy as np
import pandas as pd

PATH_IMAGES = "../images and ratings/imageversions_256"
PATH_RATINGS = "../images and ratings/ratings"

# !! correlations, invert sign for integration
PATH_INTEGRATION = "../data csv/integration"
PATH_INTEGRATION_MAXPOOL = "../data csv/integration maxpool"
PATH_INTEGRATION_AVERAGE = "../data csv/integration average"

PATH_IBCORR = "../data csv/ibcorr"
PATH_IBCORR_AVERAGE = "../data csv/ibcorr average"
PATH_IBCORR_MAXPOOL = "../data csv/ibcorr maxpool"

PATH_RESULTS = "../results"
PATH_PLOTS = "../plots"


DATASET_NAMES = ("places1", "places2", "oasis")
SCALE_NAMES = ("scale2", "scale4", "scale8", "scale16", "scale32")
STUDY_NAMES = ("short presentation", "long presentation", "complexity order", "oasis")
BEHAVIOUR_NAMES = (
    "study1_places1_short.csv",
    "study2_places1.csv",
    "study3_places2.csv",
    "study4_oasis.csv",
)

# VisualPrior.viable_feature_tasks
MODEL_NAMES = [
    "autoencoding",
    "class_object",
    "class_scene",  #'colorization' currently not working
    "curvature",
    "denoising",
    "depth_euclidean",
    "edge_occlusion",
    "edge_texture",
    "egomotion",
    "fixated_pose",
    "inpainting",
    "jigsaw",
    "keypoints2d",
    "keypoints3d",
    "nonfixated_pose",
    "normal",
    "point_matching",
    "reshading",
    "room_layout",
    "segment_semantic",
    "segment_unsup25d",
    "segment_unsup2d",
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
    "curvature",
    "edge_occlusion",
    "keypoints3d",
    "normal",
    "reshading",
    "segment_unsup25d",
]

NETS_ALL = NETS_SEMANTIC + NETS_2D + NETS_3D

_models_ordered = [
    "vanishing_point",
    "segment_unsup2d",
    "normal",
    "room_layout",
    "keypoints3d",
    "nonfixated_pose",
    "autoencoding",
    "segment_unsup25d",
    "class_scene",
    "reshading",
    "curvature",
    "egomotion",
]


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
    dfi.columns = dfi.columns.astype(np.int16)
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


def load_ratings(path, behaviours=BEHAVIOUR_NAMES):
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


beauty_ratings = load_ratings(PATH_RATINGS)


def studyname2datasetname(studyname):
    if studyname in ("short presentation", "long presentation"):
        return "places1"
    elif studyname == "complexity order":
        return "places2"
    elif studyname == "oasis":
        return "oasis"


def studyname(studyid):
    return STUDY_NAMES[int(studyid[-1]) - 1]


def set_diagonal_to_zero(rdm):
    np.fill_diagonal(rdm.values, 0)
    return rdm


def studyratings(study):
    if study == "study1" or study == "short presentation":
        return beauty_ratings["study1_places1_short.csv"]
    if study == "study2" or study == "long presentation":
        return beauty_ratings["study2_places1.csv"]
    if study == "study3" or study == "complexity order":
        return beauty_ratings["study3_places2.csv"]
    if study == "study4" or study == "oasis":
        return beauty_ratings["study4_oasis.csv"]


def study2behaviour(st):
    return BEHAVIOUR_NAMES[STUDY_NAMES.index(st)]


def study2dataset(st):
    if st in (STUDY_NAMES[:2], "study1", "study2"):
        return DATASET_NAMES[0]
    if st in (STUDY_NAMES[2], "study3"):
        return DATASET_NAMES[1]
    if st in (STUDY_NAMES[3], "study4"):
        return DATASET_NAMES[2]
