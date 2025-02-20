""" Functions that perform basic integration beauty correlation analysis

"""

import pandas as pd

import torch
from scipy.stats import pearsonr, spearmanr


def calculate_image_metrics(net, img_full, img_v1, img_v2):
    """Correlation, self-similarity, L2-norm for one image"""
    with torch.no_grad():
        act_full, act_v1, act_v2 = net(img_full), net(img_v1), net(img_v2)

    correlations, selfsimilarity, l2norm = {}, {}, {}
    for layer, act_full_, act_v1_, act_v2_ in zip(
        act_full.keys(), act_full.values(), act_v1.values(), act_v2.values()
    ):
        act_avg_ = torch.stack((act_v1_, act_v2_), dim=0).mean(dim=0).flatten()
        act_v1_ = act_v1_.flatten()
        act_v2_ = act_v2_.flatten()
        act_full_ = act_full_.flatten()
        l2norm[layer] = act_full_.norm(p=2).item()
        correlations[layer] = pearsonr(act_full_, act_avg_)[0]
        selfsimilarity[layer] = pearsonr(act_v1_, act_v2_)[0]

    return correlations, selfsimilarity, l2norm


def calculate_dataset_metrics(images, net):
    """Correlation, self-similarity, L2-norm for whole dataset"""
    images = iter(images)
    l_cor, l_ssi, l_l2n = [], [], []
    for img_full, img_v1, img_v2 in images:
        cor, ssi, l2n = calculate_image_metrics(net, img_full, img_v1, img_v2)
        l_cor.append(cor)
        l_ssi.append(ssi)
        l_l2n.append(l2n)

    cols = list(net(torch.zeros(1, 3, 256, 256)).keys())
    return (
        pd.DataFrame(l_cor, columns=cols),
        pd.DataFrame(l_ssi, columns=cols),
        pd.DataFrame(l_l2n, columns=cols),
    )


# --- Correlation (-Integration)
def correlation_coeff(net, img_full, img_v1, img_v2):
    """Calculate correlation coefficients for all layers between full and average activation pattern"""
    # activations for full image and image parts
    with torch.no_grad():
        act_full, act_v1, act_v2 = net(img_full), net(img_v1), net(img_v2)

    correlations = {}
    for layer, act_full_, act_v1_, act_v2_ in zip(
        act_full.keys(), act_full.values(), act_v1.values(), act_v2.values()
    ):
        # average activation for image parts
        act_avg_ = torch.stack((act_v1_, act_v2_), dim=0).mean(dim=0).flatten()
        act_full_ = act_full_.flatten()
        correlations[layer] = pearsonr(act_full_, act_avg_)[0]

    return correlations


def calculate_dataset_correlation(ImageDataset_iterator, net):
    """Calculate integration for whole dataset"""
    lst = []
    for img_full, img_v1, img_v2 in ImageDataset_iterator:
        lst.append(correlation_coeff(net, img_full, img_v1, img_v2))

    column_names = list(net(torch.zeros(1, 3, 256, 256)).keys())
    return pd.DataFrame(lst, columns=column_names)


def correlate_integration_beauty(
    correlation_ratings: pd.DataFrame, beauty_ratings: pd.DataFrame
):
    return correlation_ratings.aggregate(
        lambda x: spearmanr(-x, beauty_ratings)[0], axis=0
    )


# --- Self similarity
def self_similarity(net, img_v1, img_v2):
    """Calculate image self-similarity"""
    # activations for image parts
    with torch.no_grad():
        act_v1, act_v2 = net(img_v1), net(img_v2)

    selfsimilarity = {}
    for layer, act_v1_, act_v2_ in zip(act_v1.keys(), act_v1.values(), act_v2.values()):
        act_v1_ = act_v1_.flatten()
        act_v2_ = act_v2_.flatten()

        selfsimilarity[layer] = pearsonr(act_v1_, act_v2_)[0]

    return selfsimilarity


def calculate_dataset_self_similarity(ImageDataset_iterator, net_tweaked):
    """Calculate self-similarity for whole dataset"""
    lst = []
    for _, img_v1, img_v2 in ImageDataset_iterator:
        lst.append(self_similarity(net_tweaked, img_v1, img_v2))

    column_names = list(net_tweaked(torch.zeros(1, 3, 256, 256)).keys())
    return pd.DataFrame(lst, columns=column_names)


# --- L2 norm
def l2_norm(net, img_full):
    """Calculate imgage L2-norm"""
    # activation for full image
    with torch.no_grad():
        act_full = net(img_full)

    l2norm = {}
    for layer, act_full_ in zip(act_full.keys(), act_full.values()):
        l2norm[layer] = act_full_.norm(p=2).item()

    return l2norm


def calculate_dataset_l2norm(ImageDataset_iterator, net_tweaked):
    """Calculate L2-norm for whole dataset"""
    lst = []
    for img_full, _, _ in ImageDataset_iterator:
        lst.append(l2_norm(net_tweaked, img_full))

    column_names = list(net_tweaked(torch.zeros(1, 3, 256, 256)).keys())
    return pd.DataFrame(lst, columns=column_names)


# only for testing:
# def correlate_integration_beauty(integration_ratings: pd.DataFrame, beauty_ratings: pd.DataFrame):
#    return integration_ratings.aggregate(lambda x: spearmanr(x, beauty_ratings)[0], axis= 0)
