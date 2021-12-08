# filter_2320x2828.py
# remove all images in the below directories that are not 2320 x 2828

DIRS = [
    "./data/preprocessed",
    "./data/masks"
    ]

import os;
import glob;
import numpy as np;
from matplotlib import pyplot as plt;

counter = 0;
for dir in DIRS:
    files = glob.glob(os.path.join(dir, "*"));
    for filename in files:
        img = plt.imread(filename);
        if( (img.shape[0] != 2320) or (img.shape[1] != 2828) ):
            print("removing", filename, img.shape);
            os.remove(filename);
            counter += 1;

print(f"{counter} images removed.");
