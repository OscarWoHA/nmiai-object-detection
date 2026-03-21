"""Train crop classifier v2 — longer training, more augmentation, EfficientNet."""
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

with open(DATA / "train" / "annotations.json") as f:
    coco = json.load(f)
img_map = {img["id"]: img for img in coco["images"]}
img_dir = DATA / "train" / "images"

NUM_CLASSES = max(a["category_id"] for a in coco["annotations"]) + 1


class CropDataset(Dataset):
    def __init__(self, annotations, img_map, img_dir, transform, pad=0.10):
        self.annotations = [a for a in annotations if a["bbox"][2] >= 15 and a["bbox"][3] >= 15]
        self.img_map = img_map
        self.img_dir = img_dir
        self.transform = transform
        self.pad = pad

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        info = self.img_map[ann["image_id"]]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")

        x, y, w, h = ann["bbox"]
        # Random jitter on padding (augmentation)
        pad = self.pad * (0.5 + random.random())
        x1 = max(0, int(x - w * pad))
        y1 = max(0, int(y - h * pad))
        x2 = min(info["width"], int(x + w + w * pad))
        y2 = min(info["height"], int(y + h + h * pad))

        crop = img.crop((x1, y1, x2, y2))
        cw, ch = crop.size
        if cw != ch:
            side = max(cw, ch)
            padded = Image.new("RGB", (side, side), (128, 128, 128))
            padded.paste(crop, ((side - cw) // 2, (side - ch) // 2))
            crop = padded

        return self.transform(crop), ann["category_id"]


# Try efficientnet_b0 — good balance of accuracy and size
print("Creating tf_efficientnet_b0...")
model = timm.create_model("tf_efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
model = model.cuda()

data_config = resolve_data_config(model=model)
train_transform = create_transform(**data_config, is_training=True)
val_transform = create_transform(**data_config, is_training=False)

all_anns = list(coco["annotations"])
random.shuffle(all_anns)
split = int(len(all_anns) * 0.9)

train_ds = CropDataset(all_anns[:split], img_map, img_dir, train_transform)
val_ds = CropDataset(all_anns[split:], img_map, img_dir, val_transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

cat_counts = Counter(a["category_id"] for a in all_anns[:split])
weights = torch.zeros(NUM_CLASSES)
for c, n in cat_counts.items():
    weights[c] = 1.0 / np.sqrt(max(n, 1))
weights = weights / weights.sum() * NUM_CLASSES

criterion = nn.CrossEntropyLoss(weight=weights.cuda(), label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_acc = 0
for epoch in range(30):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(batch_y)
        correct += (logits.argmax(1) == batch_y).sum().item()
        total += len(batch_y)
    scheduler.step()

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            val_correct += (model(batch_x).argmax(1) == batch_y).sum().item()
            val_total += len(batch_y)

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/30: loss={total_loss/total:.3f} train_acc={correct/total:.3f} val_acc={val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), str(RUNS / "crop_classifier_v2.pt"))
        print(f"  Saved (val_acc={val_acc:.3f})")

print(f"\nBest: {best_acc:.3f}, Size: {(RUNS / 'crop_classifier_v2.pt').stat().st_size / 1e6:.1f} MB")
