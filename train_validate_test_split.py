# train_validate_test_split.py
# split all files in a folder into subsets
#
# USAGE:
#     python train_validate_test_split.py
#

import os;
import glob;
import shutil;
from sklearn.model_selection import train_test_split;

# splits should add upto 1
# if we do cross validation, set validate fraction to 0
TRAIN_FRACTION = 0.7
VALIDATE_FRACTION = 0.1
TEST_FRACTION = 0.2

INPUT_DIR = "./x_rays/preprocessed"
INPUT_MASKS_DIR = "./masks/preprocessed"

OUTPUT_DIR = "./x_rays"       # will create x_rays/train, x_rays/validate, x_rays/test
OUTPUT_MASKS_DIR = "./masks"  # will create masks/train, masks/validate, masks/test

imgs = glob.glob(os.path.join(INPUT_DIR, "*"));
masks = glob.glob(os.path.join(INPUT_MASKS_DIR, "*"));

train_prefix = os.path.join(OUTPUT_DIR, "train");
validate_prefix = os.path.join(OUTPUT_DIR, "validate");
test_prefix = os.path.join(OUTPUT_DIR, "test");

train_masks_prefix = os.path.join(OUTPUT_MASKS_DIR, "train");
validate_masks_prefix = os.path.join(OUTPUT_MASKS_DIR, "validate");
test_masks_prefix = os.path.join(OUTPUT_MASKS_DIR, "test");

x_dev, y_dev, x_test, y_test = train_test_split(imgs, masks, test_size = TEST_FRACTION);
x_train, y_train, x_val, y_val = train_test_split(x_dev, y_dev, test_size = (1-TEST_FRACTION)*VALIDATE_FRACTION);

for set, prefix in [(x_train, train_prefix), 
                    (y_train, train_masks_prefix),
                    (x_validate, validate_prefix), 
                    (y_validate, validate_masks_prefix),
                    (x_test, test_prefix), 
                    (y_test, test_masks_prefix)
                   ]:
  for path in set:
    filename = os.path.basename(path);
    new_path = os.path.join(prefix, filename);
    shutil.copy(path, new_path);

