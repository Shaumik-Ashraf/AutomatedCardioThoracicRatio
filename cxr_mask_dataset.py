# cxr_mask_dataset.py
# Chest XRay & Mask Dataset class for Pytorch

import os;
import cv2;
import glob;
import torch;
import random;
from torch.utils.data import Dataset;
#from torchvision.io import read_image;
from torchvision.transforms import ToTensor;

class CXRMaskDataset(Dataset):
    """
    USAGE:
        train_set = CXRMaskDataset("data/x_rays/train", "data/masks/train");
        validation_set = CXRMaskDataset("data/x_rays/validation", "data/masks/validation");
        test_set = CXRMaskDataset("data/x_rays/test", "data/masks/test");
    """

    def __init__(self, img_dir, masks_dir, transform=ToTensor, target_transform=ToTensor, augmentations=None):
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
        return len( img_files );

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
            img = transform(img);
        if self.target_transform:
            mask = target_transform(mask);

        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask);
            img, mask = sample['image'], sample['mask']

        return(img, mask);
