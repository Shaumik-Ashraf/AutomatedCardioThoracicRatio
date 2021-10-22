# train_validate_test_split.py
# split all files in a folder into subsets
#
# USAGE:
#     python train_validate_test_split.py
#

import os;
import glob;
import random;
import argparse;

# splits should add upto 1
# for cross validation, set validate fraction to 0
TRAIN_FRACTION = 0.7
VALIDATE_FRACTION = 0.1
TEST_FRACTION = 0.2

