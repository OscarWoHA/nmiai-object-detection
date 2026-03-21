"""Train a timm crop classifier on GT bounding box crops.

Uses mobilenetv3_large_100 (small, fast, pre-installed in sandbox via timm 0.9.12).
Trains on crops extracted from training images with 10% padding.
Class-weighted loss for imbalanced categories.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from timm.data import resolve_data_config, create_transform
from PIL import Image
from pathlib import Path
from collections import Counter
import os
import random

random.seed(42)
torch.manual_seed(42)

DATA = Path(os.path.expanduser("~/data"))
RUNS = Path(os.path.expanduser("~/runs"))
RUNS.mkdir(exist_ok=True)

with open(DATA / "train" / "annotations.json") as f:
    coco = json.load(f)
img_map = {img["id"]: img for img in coco["images"]}
img_dir = DATA / "train" / "images"

NUM_CLASSES = max(a["category_id"] for a in coco["annotations"]) + 1
print(f"Classes: {NUM_CLASSES}, Annotations: {len(coco['annotations'])}")


class CropDataset(Dataset):
    def __init__(self, annotations, img_map, img_dir, transform, pad=0.10):
        self.annotations = [a for a in annotations if a["bbox"][2] >= 15 and a["bbox"][3] >= 15]
        self.img_map = img_map
        self.img_dir = img_dir
        self.transform = transform
        self.pad = pad
        print(f"  CropDataset: {len(self.annotations)} crops")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        info = self.img_map[ann["image_id"]]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")

        x, y, w, h = ann["bbox"]
        # Add padding
        x1 = max(0, int(x - w * self.pad))
        y1 = max(0, int(y - h * self.pad))
        x2 = min(info["width"], int(x + w + w * self.pad))
        y2 = min(info["height"], int(y + h + h * self.pad))

        crop = img.crop((x1, y1, x2, y2))

        # Pad to square
        cw, ch = crop.size
        if cw != ch:
            side = max(cw, ch)
            padded = Image.new("RGB", (side, side), (128, 128, 128))
            padded.paste(crop, ((side - cw) // 2, (side - ch) // 2))
            crop = padded

        tensor = self.transform(crop)
        label = ann["category_id"]
        return tensor, label


# Create model
print("Creating mobilenetv3_large_100...")
model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=NUM_CLASSES)
model = model.cuda()

data_config = resolve_data_config(model=model)
train_transform = create_transform(**data_config, is_training=True)
val_transform = create_transform(**data_config, is_training=False)

# Split annotations 90/10
all_anns = coco["annotations"]
random.shuffle(all_anns)
split = int(len(all_anns) * 0.9)
train_anns = all_anns[:split]
val_anns = all_anns[split:]

train_ds = CropDataset(train_anns, img_map, img_dir, train_transform)
val_ds = CropDataset(val_anns, img_map, img_dir, val_transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# Class-weighted loss
cat_counts = Counter(a["category_id"] for a in train_anns)
weights = torch.zeros(NUM_CLASSES)
for c, n in cat_counts.items():
    weights[c] = 1.0 / np.sqrt(max(n, 1))
weights = weights / weights.sum() * NUM_CLASSES
criterion = nn.CrossEntropyLoss(weight=weights.cuda(), label_smoothing=0.1)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# Train
best_acc = 0
for epoch in range(15):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_y)
        correct += (logits.argmax(1) == batch_y).sum().item()
        total += len(batch_y)

    scheduler.step()

    # Validate
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            logits = model(batch_x)
            val_correct += (logits.argmax(1) == batch_y).sum().item()
            val_total += len(batch_y)

    train_acc = correct / total
    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/15: loss={total_loss/total:.3f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), str(RUNS / "crop_classifier.pt"))
        print(f"  Saved best model (val_acc={val_acc:.3f})")

print(f"\nBest val accuracy: {best_acc:.3f}")
print(f"Model size: {(RUNS / 'crop_classifier.pt').stat().st_size / 1e6:.1f} MB")
