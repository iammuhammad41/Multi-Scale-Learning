import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR    = "./VOC2012"
BATCH_SIZE  = 8
NUM_CLASSES = 21
LR          = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALES      = [0.5, 1.0, 1.5]
NUM_EPOCHS  = 20

# ─── DATASET & DATALOADER ─────────────────────────────────────────────────
class VOCMultiScale(VOCSegmentation):
    def __init__(self, root, image_set="train", transforms=None):
        super().__init__(root, year="2012", image_set=image_set,
                         download=True, transforms=transforms)

    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)
        # mask: PIL image with 0..20 labels
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask

common_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

train_ds = VOCMultiScale(DATA_DIR, image_set="train", transforms=common_tf)
val_ds   = VOCMultiScale(DATA_DIR, image_set="val",   transforms=common_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ─── UNET BACKBONE ─────────────────────────────────────────────────────────
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

# ─── MULTI‑SCALE WRAPPER ──────────────────────────────────────────────────
class MultiScaleSeg(nn.Module):
    def __init__(self, base_model, scales):
        super().__init__()
        self.base = base_model
        self.scales = scales

    def forward(self, x):
        B, C, H, W = x.shape
        logits = 0
        for s in self.scales:
            # resize
            h, w = int(H*s), int(W*s)
            xi = F.interpolate(x, size=(h,w), mode="bilinear", align_corners=False)
            yi = self.base(xi)
            yi = F.interpolate(yi, size=(H,W), mode="bilinear", align_corners=False)
            logits += yi
        return logits / len(self.scales)

# ─── TRAINING & EVAL ─────────────────────────────────────────────────────────
model = MultiScaleSeg(UNet(NUM_CLASSES), SCALES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=255)

def train_epoch():
    model.train()
    total_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        loss  = criterion(preds, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def eval_epoch():
    model.eval()
    tot, correct = 0, 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == masks).sum().item()
            tot     += masks.numel()
    return correct / tot

for epoch in range(1, NUM_EPOCHS+1):
    loss = train_epoch()
    acc  = eval_epoch()
    print(f"Epoch {epoch:02d} — train loss {loss:.4f},  val pixel‑acc {acc:.4f}")

# ─── SAVING MODEL ───────────────────────────────────────────────────────────
torch.save(model.state_dict(), "multiscale_unet_pascalvoc.pth")
