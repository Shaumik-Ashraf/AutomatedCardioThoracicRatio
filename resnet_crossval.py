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
LEARNING_RATE = 0.0001;
EPOCHS = 2;
NAME = "resnet_xval"

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Grayscale() # PIL reads as grayscale
    transforms.ConvertImageDtype(torch.float),
    # Normalization?
]);


parser = argparse.ArgumentParser(description="Train and test RESNET18 on CXRs to predict CTR (with MSE loss and Adam optimizer).");
parser.add_argument("--print-model", action="store_true", help="print model architecture");
parser.add_argument("--test-only", action="store_true", help="skip training, load model, and test");


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

def train(loader, model, loss_func, optim, epochs):
    steps = len(loader);
    model.train();

    for epoch in range(epochs):
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

            if( (i+1)%10 == 0 or (i+1) == steps ):
                print(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{steps}, Loss {loss.item()}");

# mse built in
def test(loader, model):
    model.eval();
    scores = [];

    with torch.no_grad():
        #correct = 0;
        #total = 0;

        for images, y_batch in loader:
            y_pred = model(images);
            mse = mean_square_error(y_batch, y_pred);
            scores.append(mse);

    print(f"Micro-averaged Mean Squared Error {sum(scores)/len(scores)}")

# credits to Skipper
def crossvalid(model=None,criterion=None,optimizer=None,dataset=None,k_fold=3):

    train_loss = [];
    val_loss = [];

    total_size = len(dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        print(f"--- Fold {i+1} ---");
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        # print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)"
        #        % (trll,trlr,trrl,trrr,vall,valr))

        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))

        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))

        train_set = torch.utils.data.dataset.Subset(dataset,train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset,val_indices)

        # print(len(train_set),len(val_set))
        # print()

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

        train_mse = train(train_loader, model, criterion, optimizer, EPOCHS)
        train_loss.append(train_mse)

        val_mse = test(val_loader, model)
        val_loss.append(val_mse)

    return train_loss, val_loss

def plot_loss(train_loss, val_loss):
    """
    Plot loss over epoch.
    """
    n_epochs = range(1, len(train_loss)+1)

    plt.plot(n_epochs, train_loss, 'r', label='training loss')
    plt.plot(n_epochs, val_loss, 'b', label='validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(NAME + '_loss.png', dpi=300)


if __name__ == "__main__":

    args = parser.parse_args();

    print("\n=== Loading & Initializing Data ===\n");

    train_cxr_folder = os.path.join('data', 'new', 'train', 'imgs');
    test_cxr_folder = os.path.join('data', 'new', 'test', 'imgs');
    labels_file = os.path.join('data', 'CTR_Logs.txt');

    train_set = CTRData(train_cxr_folder, labels_file, transform=TRANSFORM, target_transform=ToFloat());
    test_set = CTRData(test_cxr_folder, labels_file, transform=TRANSFORM, target_transform=ToFloat());

    #train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS);

    resnet = Resnet();
    mse_loss = nn.MSELoss();
    adam = torch.optim.Adam(resnet.parameters(), lr = LEARNING_RATE);

    if args.print_model:
        print(resnet);

    if not args.test_only:
        print("\n=== Training Model with Validation ===\n");
        start = time.time();
        #train(train_loader, resnet, mse_loss, adam, EPOCHS);
        train_score, val_score = crossvalid(resnet, mse_loss, adam, train_set);

        print(f"Finished in {(time.time() - start) / 60} minutes");

        plot_loss(train_score, val_score);
        torch.save(resnet.state_dict(), NAME + ".pt");


    print("\n=== Testing ===\n");

    if args.test_only:
        resnet.load_state_dict(torch.load(NAME + ".pt"));

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS);
    test(test_loader, resnet);
