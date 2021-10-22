# cxr_mask_dataset.py
# Chest XRay & Mask Dataset class for Pytorch

import os;
import glob;
import torch;
import random;
from torch.utils.data import Dataset;
from torchvision.io import read_image;

class CXRMaskDataset(Dataset):
    """
    USAGE:
        train_set = new CXRMaskDataset("data/x_rays/train", "data/masks/train");
        validation_set = new CXRMaskDataset("data/x_rays/validation", "data/masks/validation");
        test_set = new CXRMaskDataset("data/x_rays/test", "data/masks/test");
    """

    def __init__(self, img_dir, masks_dir, transform=None, target_transform=None):
        """
        img_dir : string - path to x-ray folder
        labels_file : string - path to CTR_Logs.txt (img name & 4 points labelled per line)
        transform : callback - whatever function you apply to each xray
        target_transform : callback - function that tranforms line from CTR_Logs.txt to mask
        """

        self.img_files = glob.glob(os.path.join(img_dir, "*.jpg")); # catch all jpg files in img_dir
        random.shuffle(self.img_files);
        self.mask_files = glob.glob(os.path.join(img_dir, "*.jpg"));
        random.shuffle(self.mask_files);
        self.transform = transform;
        self.target_transform = target_transform;

    def __len__(self):
        "return length of dataset"
        return len( img_files );

    def __getitem__(self, idx):
        """
        idx : int - get random data point

        returns (image : Tensor, mask : Tensor)
        """
        img = read_image(self.img_files[idx]);
        mask = read_image(self.mask_files[idx]);
        if transform:
            img = transform(img);
        if target_transform:
            mask = target_transform(mask);

        return(img, mask);

