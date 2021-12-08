# AutomatedCardioThoracicRatio
Survey different image segmentation techniques for computing CTR

**Work in progress**

## To split data:

Put preprocessed chest xrays in data/new/imgs

Put preprocessed masks in data/new/masks

Run `python train_test_split.py`, which will create:
 - data/new/train/imgs
 - data/new/train/masks
 - data/new/test/imgs
 - data/new/test/masks

