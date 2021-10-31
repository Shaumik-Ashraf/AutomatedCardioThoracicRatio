# coding: utf-8

# required downloads from demo: segmentation-models-pytorch albumentations

# libraries
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
# import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from cxr_mask_dataset import CXRMaskDataset
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms

print("Starting UNET segmentation...");

# helper function for data visualization
def visualize(savename, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    #plt.show()
    plt.savefig(savename)

""" We use our own dataset, but this here for reference
class Dataset(BaseDataset):
    ""CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    ""
    
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    def __init__(
            self, 
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
"""

# Lets look at data we have
dataset = CXRMaskDataset(os.path.join("data", "split", "preprocessed", "train"), os.path.join("data", "split", "masks", "train"));

image, mask = dataset[4] # get some sample
print("Saving example image");
visualize( "visualization_1.png", image=image, mask=mask.squeeze());

# Augmentations
def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(3200, 3200) # Must be padded?!?
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: torchvision.transforms.Compose

    """

    return transforms.Compose([
        preprocessing_fn,
        transforms.ToTensor,
    ]);


# Load dataset & model
ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['any', 'rc', 'rh', 'lh', 'lc']
ACTIVATION = 'softmax'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device {DEVICE}");

# create segmentation model with pretrained encoder
model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES), activation="softmax")
print("Initialized model");

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

x_train_dir = os.path.join("data", "split", "preprocessed", "train");
y_train_dir = os.path.join("data", "split", "masks", "train");

x_valid_dir = os.path.join("data", "split", "preprocessed", "validate");
y_valid_dir = os.path.join("data", "split", "masks", "validate");

x_test_dir = os.path.join("data", "split", "preprocessed", "test");
y_test_dir = os.path.join("data", "split", "masks", "test");

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    transform=get_preprocessing(preprocessing_fn),
    target_transform=get_preprocessing(preprocessing_fn),
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
print("Loaded train, validate datasets");

loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5) ]

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001) ])


# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True
)


# train model
max_score = 0
print("Training...");
for i in range(0, EPOCHS):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')

    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')


## Test best saved model

# load best saved checkpoint
best_model = torch.load('./best_model.pth')

# create test dataset
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    target=get_preprocessing(preprocessing_fn),
    target_transform=get_preprocessing(preprocessing_fn)
    #classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)

# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE
)

logs = test_epoch.run(test_dataloader)


# ## Visualize predictions

# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir
    #classes=CLASSES,
)


for i in range(5):
    n = np.random.choice(len(test_dataset))

    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]

    gt_mask = gt_mask.squeeze()

    # ToTensor transformation already done
    #x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    #pr_mask = best_model.predict(x_tensor)
    #pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    pr_mask = best_model.predict(image.to(DEVICE).unsqueeze(0));
    pr_mask = pr_mask.squeeze().cpu().numpy().round();

    visualize(
        "visualize_result.png",
        image=image_vis,
        ground_truth_mask=gt_mask,
        predicted_mask=pr_mask
    )

