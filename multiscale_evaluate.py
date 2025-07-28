import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader

# ─── CONFIG ─────────────────────────────────────────────────────────
DATA_DIR    = "./VOC2012"
BATCH_SIZE  = 8
NUM_CLASSES = 21
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "multiscale_unet_pascalvoc.pth"
SCALES      = [0.5, 1.0, 1.5]

# ─── MODEL DEFINITION ─────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.down1 = DoubleConv(3,  64)
        self.pool  = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128,256)
        self.up2   = nn.ConvTranspose2d(256,128,2,2)
        self.conv2 = DoubleConv(256,128)
        self.up1   = nn.ConvTranspose2d(128,64,2,2)
        self.conv1 = DoubleConv(128,64)
        self.outc  = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool(c1)
        c2 = self.down2(p1)
        p2 = self.pool(c2)
        c3 = self.down3(p2)
        u2 = self.up2(c3)
        c2u= torch.cat([u2, c2], dim=1); c4 = self.conv2(c2u)
        u1 = self.up1(c4)
        c1u= torch.cat([u1, c1], dim=1); c5 = self.conv1(c1u)
        return self.outc(c5)

class MultiScaleSeg(nn.Module):
    def __init__(self, base_model, scales):
        super().__init__()
        self.base = base_model
        self.scales = scales

    def forward(self, x):
        B, C, H, W = x.shape
        logits = 0
        for s in self.scales:
            h, w = int(H*s), int(W*s)
            xi = F.interpolate(x, size=(h,w), mode="bilinear", align_corners=False)
            yi = self.base(xi)
            yi = F.interpolate(yi, size=(H,W), mode="bilinear", align_corners=False)
            logits += yi
        return logits / len(self.scales)

# ─── DATA ───────────────────────────────────────────────────────────────
common_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_ds = VOCSegmentation(DATA_DIR, year="2012", image_set="val", download=False, transforms=common_tf)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ─── LOADING MODEL ─────────────────────────────────────────────────────────
model = MultiScaleSeg(UNet(NUM_CLASSES), SCALES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ─── EVALUATION ─────────────────────────────────────────────────────────
conf_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
with torch.no_grad():
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        # flatten
        t = masks.view(-1)
        p = preds.view(-1)
        # filter out ignore index 255
        valid = (t < NUM_CLASSES)
        inds = valid.nonzero().squeeze()
        t = t[inds]; p = p[inds]
        # accumulate
        conf_matrix[t, p] += 1

# pixel accuracy
correct = conf_matrix.diag().sum().item()
total   = conf_matrix.sum().item()
pixel_acc = correct / total
# per-class IoU
tp = conf_matrix.diag().float()
fn = conf_matrix.sum(dim=1).float() - tp
fp = conf_matrix.sum(dim=0).float() - tp
iou = tp / (tp + fp + fn + 1e-6)
mean_iou = iou.mean().item()

print(f"Pixel Accuracy: {pixel_acc:.4f}")
print(f"Mean IoU:      {mean_iou:.4f}")

# ─── PLOTS ──────────────────────────────────────────────────────────────
os.makedirs("evaluation_plots", exist_ok=True)

# Per-class IoU bar plot
torch_classes = [f"cls_{i}" for i in range(NUM_CLASSES)]
plt.figure(figsize=(12,4))
plt.bar(torch_classes, iou.cpu().numpy())
plt.xticks(rotation=90)
plt.ylabel("IoU")
plt.title(f"Per-class IoU (mIoU={mean_iou:.3f})")
plt.tight_layout()
plt.savefig("evaluation_plots/per_class_iou.png")

# input, GT, prediction
import random
batch = next(iter(val_loader))
imgs, masks = batch
imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
preds = model(imgs).argmax(dim=1)

def to_numpy(x): return x.cpu().permute(1,2,0).numpy()

grid_n = min(4, imgs.size(0))
plt.figure(figsize=(12,8))
for i in range(grid_n):
    plt.subplot(grid_n,3,i*3+1); plt.imshow(to_numpy(imgs[i])); plt.axis('off'); plt.title('Input')
    plt.subplot(grid_n,3,i*3+2); plt.imshow(masks[i].cpu(), cmap='gray'); plt.axis('off'); plt.title('GT')
    plt.subplot(grid_n,3,i*3+3); plt.imshow(preds[i].cpu(), cmap='gray'); plt.axis('off'); plt.title('Pred')
plt.tight_layout()
plt.savefig("evaluation_plots/examples.png")

print("Saved evaluation plots in ./evaluation_plots/")
