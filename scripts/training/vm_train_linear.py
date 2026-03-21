"""Train a linear classifier on DINOv2 embeddings of all GT crops."""
import torch
import numpy as np
import json
from pathlib import Path
from collections import Counter
import timm
from timm.data import resolve_data_config, create_transform
from PIL import Image
import os

DATA = Path(os.path.expanduser("~/data"))
RUNS = Path(os.path.expanduser("~/runs"))
RUNS.mkdir(exist_ok=True)

with open(DATA / "train" / "annotations.json") as f:
    coco = json.load(f)
img_map = {img["id"]: img for img in coco["images"]}
img_dir = DATA / "train" / "images"

model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
model = model.half().cuda().eval()
data_config = resolve_data_config(model=model)
transform = create_transform(**data_config, is_training=False)

print("Computing embeddings for all GT crops...")
all_embs, all_labels = [], []
for i, ann in enumerate(coco["annotations"]):
    info = img_map[ann["image_id"]]
    x, y, w, h = ann["bbox"]
    if w < 15 or h < 15:
        continue
    pad = 0.08
    x1, y1 = max(0, x - w * pad), max(0, y - h * pad)
    x2, y2 = min(info["width"], x + w + w * pad), min(info["height"], y + h + h * pad)
    try:
        img = Image.open(img_dir / info["file_name"]).convert("RGB")
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        cw, ch = crop.size
        if cw != ch:
            side = max(cw, ch)
            padded = Image.new("RGB", (side, side), (128, 128, 128))
            padded.paste(crop, ((side - cw) // 2, (side - ch) // 2))
            crop = padded
        tensor = transform(crop).unsqueeze(0).cuda().half()
        with torch.no_grad():
            emb = model(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu().float().numpy()[0])
            all_labels.append(ann["category_id"])
    except:
        pass
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(coco['annotations'])}")

print(f"Total crops: {len(all_embs)}")
X = torch.tensor(np.array(all_embs, dtype=np.float32)).cuda()
y = torch.tensor(np.array(all_labels, dtype=np.int64)).cuda()

num_classes = int(y.max().item()) + 1
linear = torch.nn.Linear(384, num_classes).cuda()
optimizer = torch.optim.AdamW(linear.parameters(), lr=3e-4, weight_decay=1e-2)

counts = Counter(all_labels)
weights = torch.zeros(num_classes).cuda()
for c, n in counts.items():
    weights[c] = 1.0 / np.sqrt(n)
weights = weights / weights.sum() * num_classes
criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

print("Training linear head...")
for epoch in range(20):
    perm = torch.randperm(len(X))
    total_loss, correct = 0, 0
    for i in range(0, len(X), 1024):
        bx, by = X[perm[i:i + 1024]], y[perm[i:i + 1024]]
        logits = linear(bx)
        loss = criterion(logits, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == by).sum().item()
    print(f"  Epoch {epoch + 1}/20: loss={total_loss:.3f} acc={correct / len(X):.3f}")

torch.save(linear.cpu().state_dict(), str(RUNS / "linear_head.pt"))
print(f"Saved: {(RUNS / 'linear_head.pt').stat().st_size / 1e6:.2f} MB")
