# cxr_mask_dataset.py
# Chest XRay & Mask Dataset class for Pytorch

import os;
#import cv2;
import glob;
import torch;
#import random;
import numpy as np;
from PIL import Image;
from torch.utils.data import Dataset;
from torchvision import transforms;
from torch.nn.functional import one_hot;
from torchvision.transforms.functional import to_tensor;

class OneHot:
    """
    one-hots tensors (dtype=int or long) only
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes;

    def __call__(self, tensor):
        #tensor = tensor.unsqueeze(0);
        tensor = tensor.to(torch.int64);
        #print(tensor.shape)
        return one_hot(tensor, num_classes = self.num_classes).permute(0,3,1,2);

class MaskToTensor:
    """
    converts numpy array H-W-C to tensor C-H-W
    """
    def __init__(self):
        pass;

    def __call__(self, array):
        print( array.shape );
        return torch.tensor(array).permute(3,0,1,2);

class CXRMaskDataset(Dataset):
    """
        Pytorch Dataloader class
        Assumes:
          - all images are 512x512
          - each cxr image is grayscale
          - each mask is 1 channel, with pixel values 0, 1, or 2
          - image & mask have same name

        Usage:
            train_set = CXRMaskDataset("train/cxr", "train/masks");
            train_loader = DataLoader(train_set, batch_size=4);
            # loop thru train_loader for batches
    """
    def __init__(self, cxr_folder, mask_folder):
        self.cxr_folder = cxr_folder
        self.mask_folder = mask_folder

        self.basenames = glob.glob(os.path.join(cxr_folder,'*'));
        assert len(self.basenames)!=0, "No images found"

        for i in range(len(self.basenames)):
            #self.basenames[i] = self.basenames[i][:-4] #rm last 4 chars
            self.basenames[i] = os.path.basename(self.basenames[i]);

        self.cxr_transforms = transforms.Compose([
            #transforms.PILToTensor(),
            #NumpyToTensor(),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ]);
        self.mask_transforms = transforms.Compose([
            #transforms.PILToTesnor(),
            #MaskToTensor(),
            transforms.ToTensor(),
            OneHot(3),
            transforms.ConvertImageDtype(torch.float)
        ]);


    def __len__(self):
        return len(self.basenames);

    def __getitem__(self, idx):
        cxr = self.read_img(os.path.join(self.cxr_folder, self.basenames[idx]));
        cxr = self.cxr_transforms(cxr);

        mask = self.read_img(os.path.join(self.mask_folder, self.basenames[idx]));
        mask = self.mask_transforms(mask); # does one-hot

        return(cxr, mask);

    def read_img(self, path):
        return np.array(Image.open(path));
        #return Image.open(path);

# test code
if __name__=="__main__":
    dataset = CXRMaskDataset("data/new/train_imgs", "data/new/train_masks");

    img1, mask1 = dataset.__getitem__(0);

    assert(img1 != None);
    print("image:", type(img1), img1.shape);
    print("======");

    assert(mask1 != None);
    print("mask:", type(mask1), mask1.shape);

