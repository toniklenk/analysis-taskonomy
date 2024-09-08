import os, torch
import pandas as pd
from scipy.io import loadmat


class ImageDataset(object):
    """
    Handles preparing images for input into activation extractors:

        - Load images (matlab arrays) from subfolder,
            in alphanumerical order (corresponding to beauty ratings in file).

        - Transform into PyTorch format

    This class provides a iterator to do so.
    """

    def __init__(self, img_dir, beauty_ratings_path=None):

        dir_img_list = list(f for f in os.listdir(os.path.join(img_dir, "full")))
        self.img_dir = img_dir
        self.img_list = sorted(dir_img_list)
        self.img_count = len(dir_img_list)
        if beauty_ratings_path is not None:
            self.beauty_ratings = pd.read_csv(beauty_ratings_path, header=None).mean(
                axis=1
            )

    def __iter__(self, transform=lambda x: x):
        self.img_pos = 0
        return self

    def __next__(self):
        if self.img_pos < self.img_count:
            # load arrays (transformed in matlab)
            img_full = loadmat(
                os.path.join(self.img_dir, "full", self.img_list[self.img_pos])
            )["im"]
            img_v1 = loadmat(
                os.path.join(self.img_dir, "version1", self.img_list[self.img_pos])
            )["imv1"]
            img_v2 = loadmat(
                os.path.join(self.img_dir, "version2", self.img_list[self.img_pos])
            )["imv2"]

            # convert to input format of Taskonomy models
            img_full = torch.tensor(img_full).permute([2, 0, 1]).unsqueeze(0)
            img_v1 = torch.tensor(img_v1).permute([2, 0, 1]).unsqueeze(0)
            img_v2 = torch.tensor(img_v2).permute([2, 0, 1]).unsqueeze(0)
            self.img_pos += 1
            return img_full, img_v1, img_v2
        else:  # prepare for a possible next iteration
            self.img_pos = 0
            raise StopIteration
