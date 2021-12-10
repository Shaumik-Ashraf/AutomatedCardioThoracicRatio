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
from torchvision import models;
#from torchvision.datasets import MNIST;
from torchvision import transforms;
#from torchvision.io import write_png;
#from sklearn.metrics import accuracy_score;
from matplotlib import pyplot as plt;


BATCH_SIZE = 4;
WORKERS = 4;
LEARNING_RATE = 0.000001;
EPOCHS = 10;
NAME = "resnet_val"

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

class CTRData(Dataset):
    def __init__(self, cxr_dir, ctr_file, transform = None, target_transform = None):
        super(CTRData, self).__init__();
        self.imgs = glob.glob(os.path.join(cxr_dir, "*"));

        f = open(ctr_file, "r");
        lines = f.read().split("\n");
        f.close();

        self.ctr_map = {};
        for line in lines:
            tokens = line.split();
            if len(tokens) > 0:
                self.ctr_map[tokens[0]] = float(tokens[2]);

        self.transform = transform;
        self.target_transform = target_transform;

    def __getitem__(self, i):
        x = Image.open(self.imgs[i]);
        y = self.ctr_map[ os.path.basename(self.imgs[i]) ];

        if self.transform:
            x = self.transform(x);

        if self.target_transform:
            y = self.target_transform(y);

        return x, y

    def __len__(self):
        return len(self.imgs);

    def debug(self):
        print(self.imgs);
        print(self.ctr_map);


class Resnet(nn.Module):
    "Resnet18 adapted for CXR and CTR prediction"

    def __init__(self):
        super(Resnet, self).__init__();
        self.resnet18 = models.resnet18(); # random weights
        self.resnet18.conv1 = nn.Conv2d(1, 64, 7, stride = 2, padding = 3); # first layer, grayscale input
        self.resnet18.fc = nn.Linear(512, 1, bias=True); # last computation layer
        self.flatten = nn.Flatten(0); # make output scalar (1D array for batch)

    def forward(self, x):
        return self.flatten( self.resnet18(x) );


def train(loader, valloader, model, loss_func, optim, epochs):
    steps = len(loader);
    train_loss_per_epoch = [];
    val_loss_per_epoch = [];

    for epoch in range(epochs):
        running_loss = 0.0;
        model.train();
        for i, (x_batch, y_batch) in enumerate(loader):

            #print(x_batch);
            #print("---");
            #print(y_batch);

            y_pred = model(x_batch);

            #print("---");
            #print(y_pred);

            loss = loss_func(y_pred, y_batch);

            optim.zero_grad();
            loss.backward();
            optim.step();

            running_loss += loss.item();
            if( (i+1)%10 == 0 or (i+1)==steps ):
                print(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{steps}, Loss {loss.item()}");

        train_loss_per_epoch.append( running_loss / steps );
        val_loss_per_epoch.append( test(valloader, model, True) )

    return( train_loss_per_epoch, val_loss_per_epoch );

def test(loader, model, is_validation = False):
    model.eval();
    scores = [];

    with torch.no_grad():
        #correct = 0;
        #total = 0;

        for images, y_batch in loader:
            y_pred = model(images);
            mse = mean_square_error(y_batch, y_pred);
            scores.append(mse);

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
    plt.savefig(NAME + '_loss.png', dpi=200);


if __name__ == "__main__":

    args = parser.parse_args();


    print("=== Loading & Initializing Data ===");

    train_cxr_folder = os.path.join('data', 'new', 'train1', 'imgs');
    val_cxr_folder = os.path.join('data', 'new', 'validate1', 'imgs');
    test_cxr_folder = os.path.join('data', 'new', 'test1', 'imgs');
    labels_file = os.path.join('data', 'CTR_Logs.txt');

    train_set = CTRData(train_cxr_folder, labels_file, transform=TRANSFORM, target_transform=ToFloat());
    val_set = CTRData(val_cxr_folder, labels_file, transform=TRANSFORM, target_transform=ToFloat());
    test_set = CTRData(test_cxr_folder, labels_file, transform=TRANSFORM, target_transform=ToFloat());

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS);
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS);
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS);


    resnet = Resnet();
    mse_loss = nn.MSELoss();
    adam = torch.optim.Adam(resnet.parameters(), lr = LEARNING_RATE);

    if args.print_model:
        print(resnet);

    if not args.test_only:
        print("=== Training Model ===");
        start = time.time();

        train_loss_per_epoch, val_loss_per_epoch = train(train_loader, val_loader, resnet, mse_loss, adam, EPOCHS);
        plot_loss(train_loss_per_epoch, val_loss_per_epoch);

        print(f"Finished in {(time.time() - start) / 60} minutes");
        torch.save(resnet.state_dict(), NAME + ".pt");

    print("=== Testing ===");

    if args.test_only:
        resnet.load_state_dict(torch.load(NAME + ".pt"));

    test(test_loader, resnet);
