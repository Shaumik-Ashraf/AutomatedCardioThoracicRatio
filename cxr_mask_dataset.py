# cxr_mask_dataset.py
# Chest XRay & Mask Dataset class for Pytorch

import os;
import cv2;
import glob;
import torch;
import random;
import numpy as np;
from torch.utils.data import Dataset;
#from torchvision.io import read_image;
import torchvision



def to_tensor(array):
    # numpy to tensor
    if( len(array.shape)==3 ):
        return(array.transpose(2, 0, 1).astype('float'));
    else:
        return(array.transpose(2, 0, 1, 3).astype('float'));

class CXRMaskDataset(Dataset):
    """
    USAGE:
        train_set = CXRMaskDataset("data/x_rays/train", "data/masks/train");
        validation_set = CXRMaskDataset("data/x_rays/validation", "data/masks/validation");
        test_set = CXRMaskDataset("data/x_rays/test", "data/masks/test");
    """
    DEFAULT_TRANSFORM = None;

    def __init__(self, img_dir, masks_dir, transform=DEFAULT_TRANSFORM, target_transform=DEFAULT_TRANSFORM, augmentations=None):
        """
        img_dir : string - path to x-ray folder
        labels_file : string - path to CTR_Logs.txt (img name & 4 points labelled per line)
        transform : callback - whatever function you apply to each xray
        target_transform : callback - function that tranforms line from CTR_Logs.txt to mask
        augmentations: albumentations.Compose
        """

        self.img_files = glob.glob(os.path.join(img_dir, "*")); # catch all files in img_dir
        random.shuffle(self.img_files);
        self.mask_files = glob.glob(os.path.join(img_dir, "*"));
        random.shuffle(self.mask_files);
        self.transform = transform;
        self.target_transform = target_transform;
        self.augmentations = augmentations;

    def __len__(self):
        "return length of dataset"
        return len( self.img_files );

    def __getitem__(self, idx):
        """
        idx : int - get random data point

        returns (image : Tensor, mask : Tensor)
        """
        img = cv2.imread(self.img_files[idx]);
        mask = cv2.imread(self.mask_files[idx]);

        # one-hot masks:             any   rc    rh    lh   lc
        masks = [(mask == v) for v in [0, 0.25, 0.50, 0.75, 1.0]]
        mask = np.stack(masks, axis=-1).astype('float');

        if self.transform:
            img = self.transform(img);
        else:
            img = to_tensor(img);

        if self.target_transform:
            mask = self.target_transform(mask);
        else:
            mask = to_tensor(mask);

        # one-hot after tensor
        # mask = torch.nn.functional.one_hot(mask, 5);

        if self.augmentations:
            sample = self.augmentations(image=img, mask=mask);
            img, mask = sample['image'], sample['mask']

        return(img, mask);
