# probe models with dummy inputs for understanding

from cxr_mask_dataset import CXRMaskDataset;
from resnet_val import ToFloat, CTRData, Resnet;
from vit_val import VisionTransformer;
from torch.utils.data import DataLoader;
from torchvision import transforms as T;
from torch.nn.functional import mse_loss;
import torch;
import os;

img_dir = os.path.join("data", "probe", "imgs");
mask_dir = os.path.join("data", "probe", "masks");
ctr_file = os.path.join("data", "CTR_Logs.txt");

xforms = T.Compose([T.ToTensor(), T.ConvertImageDtype(torch.float)]);
txforms = ToFloat();

probeset = CTRData(img_dir, ctr_file, xforms, txforms);
loader = DataLoader(probeset, batch_size = 1, shuffle = False, num_workers = 0);

resnet = Resnet();
resnet.load_state_dict(torch.load("resnet.pt"));
print("--------- resnet ------------");
print("Sample | GT | Prediction | MSE");
for i, (x, y_true) in enumerate(loader):
    y_pred = resnet(x);
    mse = mse_loss(y_true, y_pred);
    print(f"{i} | {y_true} | {y_pred} | {mse}");


vit = VisionTransformer(); # lg
vit.load_state_dict(torch.load("vit_val_lg.pt"));
print("--------- transformer ------------");
print("Sample | GT | Prediction | MSE");
for i, (x, y_true) in enumerate(loader):
    y_pred = vit(x);
    mse = mse_loss(y_true, y_pred);
    print(f"{i} | {y_true} | {y_pred} | {mse}");


