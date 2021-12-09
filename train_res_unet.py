# train and run resnet on CXR images and guess continuous CTR

import os
import glob
import time
import torch
import argparse
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss as mean_square_error;
from torchvision import models
#from torchvision.datasets import MNIST;
from torchvision import transforms
#from torchvision.io import write_png;
#from sklearn.metrics import accuracy_score;
from matplotlib import pyplot as plt
from cxr_mask_dataset import CXRMaskDataset
from res_unet_model import ResnetUNet
# from torchsummary import summary
from loss import dice_loss
from collections import defaultdict

# BATCH_SIZE = 4
BATCH_SIZE = 16
WORKERS = 4
LEARNING_RATE = 1e-4
EPOCHS = 2
# EPOCHS = 1


TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Grayscale() # PIL reads as grayscale
    transforms.ConvertImageDtype(torch.float),
    # Normalization?
])


parser = argparse.ArgumentParser(description="Train and test RESNET18 on CXRs to predict CTR (with MSE loss and Adam optimizer).");
parser.add_argument("--test-only", action="store_true", help="skip training, load model, and test")


class ToFloat():
    def __init__(self):
        self.target_dtype = torch.float

    def __call__(self, scalar):
        return torch.tensor(scalar, dtype=self.target_dtype)

class CTRData(Dataset):
    def __init__(self, cxr_dir, ctr_file, transform = None, target_transform = None):
        super(CTRData, self).__init__()
        self.imgs = glob.glob(os.path.join(cxr_dir, "*"))

        f = open(ctr_file, "r")
        lines = f.read().split("\n")
        f.close()

        self.ctr_map = {}
        for line in lines:
            tokens = line.split()
            if len(tokens) > 0:
                self.ctr_map[tokens[0]] = float(tokens[2])

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        x = Image.open(self.imgs[i])
        y = self.ctr_map[ os.path.basename(self.imgs[i]) ]

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.imgs)

    def debug(self):
        print(self.imgs)
        print(self.ctr_map)


class Resnet(nn.Module):
    "Resnet18 adapted for CXR and CTR prediction"

    def __init__(self):
        super(Resnet, self).__init__()
        self.resnet18 = models.resnet18(); # random weights
        self.resnet18.conv1 = nn.Conv2d(1, 64, 7, stride = 2, padding = 3); # first layer, grayscale input
        # self.resnet18.fc = nn.Linear(512, 1, bias=True); # last computation layer
        self.resnet18.fc = nn.Linear(512, 3, bias=True)
        self.flatten = nn.Flatten(0); # make output scalar (1D array for batch)

    def forward(self, x):
        return self.flatten( self.resnet18(x) )

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

# def train(loader, model, loss_func, optim, epochs): modified
def train(loader, model, optim, epochs):
    steps = len(loader)
    model.train()

    for epoch in range(epochs):

        metrics = defaultdict(float)
        epoch_samples = 0

        for i, (x_batch, y_batch) in enumerate(loader):

            #print(x_batch);
            #print("---");
            #print(y_batch);
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # print(f'device: {device}, x_batch type: {type(x_batch)}, y_batch type: {type(y_batch)}')

            y_pred = model(x_batch)

            #print("---");
            #print(y_pred);

            # loss = loss_func(y_pred, y_batch)
            # print(f'y_pred shape: {y_pred.shape}, y_batch shape: {y_batch.shape}')
            y_pred = torch.squeeze(y_pred)
            y_batch = torch.squeeze(y_batch)
            loss = calc_loss(y_pred, y_batch, metrics)

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_samples += x_batch.size(0)

            if( True or (i+1)%100 == 0 ): # print everything atm
                # print(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{steps}, Loss {loss.item()}")
                print(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{steps}, Loss {loss}")
        
        # print_metrics(metrics, epoch_samples, 'train')
        # epoch_loss = metrics['loss'] / epoch_samples

def test(loader, model):
    model.eval()
    scores = []

    with torch.no_grad():
        #correct = 0;
        #total = 0;
        metrics = defaultdict(float)
        # epoch_samples = 0

        for images, y_batch in loader:
            
            y_pred = model(images)
            y_pred = torch.squeeze(y_pred)
            y_batch = torch.squeeze(y_batch)
            loss = calc_loss(y_pred, y_batch, metrics)
                        
            # y_pred = model(images)
            # mse = mean_square_error(y_batch, y_pred)
            # scores.append(mse)
            scores.append(loss)

    # print(f"Micro-averaged Mean Squared Error {sum(scores)/len(scores)}")
    print(f"average loss {sum(scores)/len(scores)}")


if __name__ == "__main__":

    args = parser.parse_args()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f'Using {device}')

    print("===")
    print("Loading & Initializing Data")
    print("===")

    # labels_file = os.path.join('data', 'CTR_Logs.txt')

    train_cxr_folder = os.path.join('data', 'processed', 'train', 'imgs')
    test_cxr_folder = os.path.join('data', 'processed', 'test', 'imgs')

    train_mask_folder = os.path.join('data', 'processed', 'train', 'masks')
    test_mask_folder = os.path.join('data', 'processed', 'test', 'masks')

    train_set = CXRMaskDataset(train_cxr_folder, train_mask_folder)
    test_set = CXRMaskDataset(test_cxr_folder, test_mask_folder)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    # resnet = Resnet() modified
    resunet = ResnetUNet(3)
    resunet.to(device)
    # summary(resunet, input_size=(1,512,512))
    # mse_loss = nn.MSELoss()
    adam = torch.optim.Adam(resunet.parameters(), lr = LEARNING_RATE)

    if not args.test_only:
        print("===")
        print("Training Model")
        print("===")
        start = time.time()
        # train(train_loader, resunet, mse_loss, adam, EPOCHS) modified
        train(train_loader, resunet, adam, EPOCHS)
        print(f"Finished in {(time.time() - start) / 60} minutes")
        torch.save(resunet.state_dict(), "res_unet.pt"); # give better name?

    print("===")
    print("Testing")
    print("===")

    if args.test_only:
        resunet.load_state_dict(torch.load("res_unet.pt"))

    test(test_loader, resunet)
