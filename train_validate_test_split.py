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

INPUT_DIR = os.path.join("data", "preprocessed");
INPUT_MASKS_DIR = os.path.join("data", "masks");

OUTPUT_DIR = os.path.join("data", "split", "preprocessed"); # will create x_rays/train, x_rays/validate, x_rays/test
OUTPUT_MASKS_DIR = os.path.join("data", "split", "masks");  # will create masks/train, masks/validate, masks/test

imgs = glob.glob(os.path.join(INPUT_DIR, "*"));
masks = glob.glob(os.path.join(INPUT_MASKS_DIR, "*"));

train_prefix = os.path.join(OUTPUT_DIR, "train");
validate_prefix = os.path.join(OUTPUT_DIR, "validate");
test_prefix = os.path.join(OUTPUT_DIR, "test");

train_masks_prefix = os.path.join(OUTPUT_MASKS_DIR, "train");
validate_masks_prefix = os.path.join(OUTPUT_MASKS_DIR, "validate");
test_masks_prefix = os.path.join(OUTPUT_MASKS_DIR, "test");

x_dev, x_test, y_dev, y_test = train_test_split(imgs, masks, test_size=TEST_FRACTION);
#print(imgs[0]);
#print(len(imgs));
#print(masks[0]);
#print(len(masks));
#print(x_dev[0]);
#print(len(x_dev));
#print(y_dev[0]);
#print(len(y_dev));
#print(x_test[0]);
print("test set x-rays:", len(x_test));
#print(y_test[0]);
print("test set masks:", len(y_test));

x_train, x_val, y_train, y_val = train_test_split(x_dev, y_dev, test_size=(1-TEST_FRACTION)*VALIDATE_FRACTION);
#print(x_train[0]);
print("train set x-rays:", len(x_train));
#print(y_train[0]);
print("train set masks:", len(y_train));
#print(x_val[0]);
print("validation set x-rays:", len(x_val));
#print(y_val[0]);
print("validation set masks:", len(y_val));


# generate dirs if they don't exist
for dir in [train_prefix, validate_prefix, test_prefix, train_masks_prefix, validate_masks_prefix, test_masks_prefix]:
    os.makedirs(dir, exist_ok = True);

# move files by split
for set, prefix in [(x_train, train_prefix),
                    (y_train, train_masks_prefix),
                    (x_val, validate_prefix),
                    (y_val, validate_masks_prefix),
                    (x_test, test_prefix),
                    (y_test, test_masks_prefix)
                   ]:
  for path in set:
    filename = os.path.basename(path);
    new_path = os.path.join(prefix, filename);
    shutil.copy(path, new_path);

