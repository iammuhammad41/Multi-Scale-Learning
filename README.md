# Multi-Scale-Learning
Multi-scale learning involves processing data at different levels of resolution to capture both fine details and global context. This is particularly useful in tasks like semantic segmentation and object detection, where understanding context is as important as precise feature localization.

```markdown
# Multi‑Scale Learning for Semantic Segmentation

This repository implements a **multi‑scale U‑Net** for semantic segmentation, processing images at multiple resolutions to capture both fine details and global context. We demonstrate it on the **Pascal VOC 2012** dataset.

## 📂 Project Structure
```

.
├── data/
│   ├── VOCdevkit/VOC2012/             # download & unzip Pascal VOC 2012 here
│   └── …
├── models/
│   └── multiscale\_unet.py             # model definition
├── train\_multiscale.py                # training script
├── multiscale\_evaluate.py             # evaluation & plotting
├── requirements.txt                   # Python dependencies
└── README.md

````

## 🔍 Task
> **Multi‑Scale Learning:**  
> We build a U‑Net that ingests each image at several resolutions (e.g. 256×256, 128×128, 64×64), fuses their feature maps, and decodes to a full‑resolution segmentation mask. This improves boundary localization and context awareness.

## ⚙️ Requirements
```bash
pip install -r requirements.txt
# requirements.txt includes:
#   torch, torchvision, tqdm, matplotlib, numpy, opencv-python, scikit-learn, pillow
````

## 🚀 Usage

1. **Prepare Pascal VOC 2012**
   Download & extract under `data/VOCdevkit/VOC2012/`.

2. **Train**

   ```bash
   python train_multiscale.py \
     --data_root data/VOCdevkit/VOC2012/ \
     --batch_size 8 \
     --epochs 50 \
     --lr 1e-3 \
     --save_dir models/
   ```

3. **Evaluate & Plot**

   ```bash
   python multiscale_evaluate.py \
     --data_root data/VOCdevkit/VOC2012/ \
     --model_path models/best_multiscale_unet.pth \
     --out_dir evaluation_plots/
   ```

   This will compute **pixel accuracy** and **mean IoU**, save:

   * `evaluation_plots/per_class_iou.png`
   * `evaluation_plots/sample_predictions.png`

## 📈 Results

After training, you should see:

* **Mean IoU** across 21 Pascal VOC classes
* **Per‑class IoU** bar chart
* **Random sample grid** of input / GT mask / predicted mask

