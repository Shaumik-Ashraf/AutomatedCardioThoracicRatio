# train and run resnet on CXR images and guess continuous CTR

import os;
import glob;
import time;
import torch;
import argparse;
from PIL import Image;
from torch.utils.data import Dataset;
from torch.utils.data import DataLoader;
from torch import nn;
from torch.nn.functional import mse_loss as mean_square_error;
import torch.nn.functional as F
from torchvision import models;
#from torchvision.datasets import MNIST;
from torchvision import transforms;
#from torchvision.io import write_png;
#from sklearn.metrics import accuracy_score;
from matplotlib import pyplot as plt;
from cxr_mask_dataset import CXRMaskDataset
from res_unet_model import ResnetUNet, UNet2 # mod'd
# from torchsummary import summary
from loss import dice_loss
from collections import defaultdict


BATCH_SIZE = 4;
WORKERS = 4;
LEARNING_RATE = 0.0001; # Try: 0.01
EPOCHS = 5
NAME = "unet_val"
OUTPUT_PATH = os.path.join('data', 'predicted_unet_val', 'masks')
BCE_WEIGHT = 0.1

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Grayscale() # PIL reads as grayscale
    transforms.ConvertImageDtype(torch.float),
    # Normalization?
]);


parser = argparse.ArgumentParser(description="Train and test RESNET18 on CXRs to predict CTR (with MSE loss and Adam optimizer).");
parser.add_argument("--test-only", action="store_true", help="skip training, load model, and test");
parser.add_argument("--print-model", action="store_true", help="print model, and then train/test");


class ToFloat():
    def __init__(self):
        self.target_dtype = torch.float

    def __call__(self, scalar):
        return torch.tensor(scalar, dtype=self.target_dtype);

def calc_loss(pred, target, metrics, bce_weight=0.5): # try weight = 0.01
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    # later we calc CTR and then MSE
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train(loader, valloader, model, optim, epochs):
    steps = len(loader);
    train_loss_per_epoch = [];
    val_loss_per_epoch = [];

    for epoch in range(epochs):
        running_loss = 0.0;
        model.train();
        metrics = defaultdict(float)
        epoch_samples = 0
        for i, (x_batch, y_batch) in enumerate(loader):

            y_pred = model(x_batch);
            
            y_pred = torch.squeeze(y_pred)
            y_batch = torch.squeeze(y_batch)
            loss = calc_loss(y_pred, y_batch, metrics, bce_weight=BCE_WEIGHT)

            optim.zero_grad();
            loss.backward();
            optim.step();

            running_loss += loss.item();

            epoch_samples += x_batch.size(0)
            if( (i+1)%10 == 0 or (i+1)==steps ):
                print(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{steps}, Loss {loss}");

        print_metrics(metrics, epoch_samples, 'train')
        train_loss_per_epoch.append( running_loss / steps );
        val_loss_per_epoch.append( test(valloader, model, True) )

        # plot masks for each epoch

    return( train_loss_per_epoch, val_loss_per_epoch );

def test(loader, model, is_validation = False):
    model.eval();
    scores = [];

    with torch.no_grad():
        #correct = 0;
        #total = 0;
        metrics = defaultdict(float)
        for images, y_batch in loader:
            y_pred = model(images);

            y_pred = torch.squeeze(y_pred)
            y_batch = torch.squeeze(y_batch)
            loss = calc_loss(y_pred, y_batch, metrics, bce_weight=BCE_WEIGHT)
            scores.append(loss);

    if is_validation:
        print(f"Val Micro-averaged Mean Squared Error {sum(scores)/len(scores)}")
    else:
        print(f"Test Micro-averaged Mean Squared Error {sum(scores)/len(scores)}")

    return( sum(scores) / len(scores) );

def plot_loss(train_loss, val_loss):
    """
    Plot loss over epoch.
    """
    n_epochs = range(1, len(train_loss)+1)

    plt.plot(n_epochs, train_loss, 'r', label='Training Loss')
    plt.plot(n_epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(NAME + f'_loss_{EPOCHS}_{BCE_WEIGHT}.png', dpi=300);


if __name__ == "__main__":

    args = parser.parse_args();


    print("=== Loading & Initializing Data ===");

    train_cxr_folder = os.path.join('data', 'train1', 'imgs');
    val_cxr_folder = os.path.join('data', 'validate1', 'imgs');
    test_cxr_folder = os.path.join('data', 'test1', 'imgs');
    # labels_file = os.path.join('data', 'CTR_Logs.txt');

    train_mask_folder = os.path.join('data', 'train1', 'masks')
    val_mask_folder = os.path.join('data', 'validate1', 'masks')
    test_mask_folder = os.path.join('data', 'test1', 'masks')

    train_set = CXRMaskDataset(train_cxr_folder, train_mask_folder)
    val_set = CXRMaskDataset(val_cxr_folder, val_mask_folder)
    test_set = CXRMaskDataset(test_cxr_folder, test_mask_folder)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS);
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS);
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS);


    resnet = UNet2();
    # mse_loss = nn.MSELoss();
    adam = torch.optim.Adam(resnet.parameters(), lr = LEARNING_RATE);

    if args.print_model:
        print(resnet);

    if not args.test_only:
        print("=== Training Model ===");
        start = time.time();

        train_loss_per_epoch, val_loss_per_epoch = train(train_loader, val_loader, resnet, adam, EPOCHS);
        plot_loss(train_loss_per_epoch, val_loss_per_epoch);

        print(f"Finished in {(time.time() - start) / 60} minutes");
        # torch.save(resnet.state_dict(), NAME + ".pt");
        # Save the model
        i = 0
        while os.path.exists(f'{NAME}{i}.pt'):
            i += 1
        model_name = f'{NAME}{i}.pt'
        # torch.save(resunet.state_dict(), "res_unet.pt"); # give better name?
        torch.save(resnet.state_dict(), model_name)

    print("=== Testing ===");

    if args.test_only:
        resnet.load_state_dict(torch.load(NAME + ".pt"));

    test(test_loader, resnet);
