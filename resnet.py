# resnet.py
# use resnet to directly predict CTR

import torch;
import pandas as pd;
import numpy as np;
import torch.nn as nn;
import torch.nn.functional as F;
import torchvision.models as models;
from torchvision.transforms import ToTensor, Grayscale
from torhcvision.transforms.functional import to_grayscale
from torch.utils.data import Dataset, DataLoader;
from pathlib import Path;
from PIL import Image;
#from sklearn.metrics import ...


