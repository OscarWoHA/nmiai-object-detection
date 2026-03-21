#!/bin/bash
# VM2: Train YOLOv8l nc=356 + DINOv2 embeddings in parallel with VM1
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Installing packages ==="
sudo apt-get update -qq && sudo apt-get install -y -qq unzip libgl1-mesa-glx libglib2.0-0
pip install ultralytics==8.1.0 timm==0.9.12 -q

echo "=== Downloading data ==="
mkdir -p ~/data && cd ~/data

[ -f coco_dataset.zip ] && echo "exists" || curl -L -o coco_dataset.zip 'https://storage.googleapis.com/nmaichamps-participant-data/norgesgruppen-data/NM_NGD_coco_dataset.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=1065880881946-compute%40developer.gserviceaccount.com%2F20260319%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260319T185007Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&X-Goog-Signature=22297b8ab52e57069aa61a95cad1c79bbe6718042236dec93c4a869a4dff9ca7dcfb39d18591b7cf1cfe5b486af1791dc89bb25ecf72b5ccff6f356fc696a7f8814e803c9a545f56e637dd6e5e51429cdc1c29c8e81b882772565b3119802d40c24a66cdf7e4e032fa83c0d7ed53c7f5d9c25ea30fc0cbf825635a4ba5f3a201883b82814f824cd937455c9d5d85ef8a12851b5adb55d94caae16c054621d6fa8b19cdfa094a5e79196db8c241b34e68fee11b94a4eafd61c4d48b65a2d3fafd30adc31fc0bbf8048200d6965e1102654a0755977ac14e3337475c4c1fb53d4750ba89aaa0a2468f6bcd65f5a8553929246ad96968b2c8c2f56af6593f315c4f'

[ -f product_images.zip ] && echo "exists" || curl -L -o product_images.zip 'https://storage.googleapis.com/nmaichamps-participant-data/norgesgruppen-data/NM_NGD_product_images.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=1065880881946-compute%40developer.gserviceaccount.com%2F20260319%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260319T185007Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&X-Goog-Signature=05eaf10cc1be437a2fdb6927b86be2242eb22b796c08eac8e0721a4f999a20b25a8b8755cd00d67f1f581a1497c13cb0a1bb76596cfd60dadbcae0533364e3c0f4452156263c765ea58c1525c0b5ac012d29d0848c0ae826a347b64a7089119f93e95ac829ed39f622624cd3891075a67570205e655644303607e645d3227c514ea833badcb8b88b5142ff9f72d282c3c299451b692353e8f4b1aedc4ef674f2e20068a47f8bcd08d28eae5d62694100223c756a3f4ef81176d876afab3664d4e37e7fac085428ee046980a841026e104d9e861f0c01c124293bf5275cc7df030174d2142999f43ddb1ead4aefa06259aef1b799eed427f94b1c8b86426b149a'

echo "=== Unpacking ==="
unzip -qo coco_dataset.zip
unzip -qo product_images.zip

echo "=== Converting COCO to YOLO ==="
python3 << 'PYEOF'
import json, random, os
from pathlib import Path
from collections import Counter, defaultdict

random.seed(42)
DATA = Path(os.path.expanduser("~/data"))

ann_path = list(DATA.rglob("annotations.json"))[0]
img_candidates = [d for d in DATA.rglob("images") if d.is_dir() and any(d.glob("*.jpg"))]
img_dir = img_candidates[0]

with open(ann_path) as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]
img_map = {img["id"]: img for img in images}

img_cats = defaultdict(set)
for ann in annotations:
    img_cats[ann["image_id"]].add(ann["category_id"])

cat_counts = Counter(ann["category_id"] for ann in annotations)
rare_cats = {c for c, n in cat_counts.items() if n < 10}
rare_images = {img_id for img_id, cats in img_cats.items() if cats & rare_cats}

all_ids = [img["id"] for img in images]
non_rare = [i for i in all_ids if i not in rare_images]
random.shuffle(non_rare)
val_count = min(max(1, int(len(all_ids) * 0.15)), len(non_rare))
val_ids = set(non_rare[:val_count])
train_ids = set(all_ids) - val_ids

labels = defaultdict(list)
for ann in annotations:
    img = img_map[ann["image_id"]]
    x, y, w, h = ann["bbox"]
    cx = max(0, min(1, (x + w/2) / img["width"]))
    cy = max(0, min(1, (y + h/2) / img["height"]))
    nw = max(0, min(1, w / img["width"]))
    nh = max(0, min(1, h / img["height"]))
    labels[ann["image_id"]].append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

for split, ids in [("train", train_ids), ("val", val_ids)]:
    lbl_dir = DATA / "labels_nc356" / split / "labels"
    img_dst = DATA / "labels_nc356" / split / "images"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    img_dst.mkdir(parents=True, exist_ok=True)
    for img_id in ids:
        info = img_map[img_id]
        src = img_dir / info["file_name"]
        dst = img_dst / info["file_name"]
        if not dst.exists() and src.exists():
            os.symlink(str(src), str(dst))
        lbl_file = lbl_dir / (Path(info["file_name"]).stem + ".txt")
        lines = labels.get(img_id, [])
        lbl_file.write_text("\n".join(lines) + "\n" if lines else "")

cat_names = [c["name"] for c in sorted(categories, key=lambda c: c["id"])]
with open(DATA / "data_nc356.yaml", "w") as f:
    f.write(f"path: {DATA}\ntrain: labels_nc356/train/images\nval: labels_nc356/val/images\n\nnc: {len(cat_names)}\nnames: {cat_names}\n")

# barcode mapping
cat_name_to_id = {c["name"]: c["id"] for c in categories}
meta_files = list(DATA.rglob("metadata.json"))
if meta_files:
    with open(meta_files[0]) as f:
        meta = json.load(f)
    mapping = {}
    for product in meta["products"]:
        bc = product["product_code"]
        name = product["product_name"]
        if name in cat_name_to_id:
            mapping[bc] = cat_name_to_id[name]
        else:
            for cn, ci in cat_name_to_id.items():
                if cn.strip().upper() == name.strip().upper():
                    mapping[bc] = ci
                    break
    with open(DATA / "barcode_to_category.json", "w") as f:
        json.dump(mapping, f)

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
PYEOF

echo "=== Computing DINOv2 embeddings ==="
python3 << 'PYEOF'
import json, numpy as np, torch, timm
from timm.data import resolve_data_config, create_transform
from PIL import Image
from pathlib import Path
import os

DATA = Path(os.path.expanduser("~/data"))
RUNS = Path(os.path.expanduser("~/runs"))
RUNS.mkdir(exist_ok=True)

barcode_to_cat = json.loads((DATA / "barcode_to_category.json").read_text())
model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
model = model.half().cuda().eval()
data_config = resolve_data_config(model=model)
transform = create_transform(**data_config, is_training=False)

torch.save(model.cpu().state_dict(), str(RUNS / "dinov2_vits14.pt"))
model = model.cuda()
print(f"DINOv2 weights: {(RUNS / 'dinov2_vits14.pt').stat().st_size / 1e6:.1f} MB")

multiangle_candidates = list(DATA.rglob("images_multiangle"))
multiangle = [d for d in multiangle_candidates if d.is_dir()][0]

all_emb, all_ids = [], []
for d in sorted(multiangle.iterdir()):
    if not d.is_dir() or d.name == "__MACOSX": continue
    bc = d.name
    if bc not in barcode_to_cat: continue
    angle_embs = []
    for p in sorted(d.glob("*.jpg")):
        try:
            img = Image.open(p).convert("RGB")
            t = transform(img).unsqueeze(0).cuda().half()
            with torch.no_grad():
                e = model(t)
                e = e / e.norm(dim=-1, keepdim=True)
                angle_embs.append(e.cpu().float().numpy()[0])
        except Exception as ex:
            print(f"Error: {p}: {ex}")
    if angle_embs:
        avg = np.mean(angle_embs, axis=0)
        avg = avg / np.linalg.norm(avg)
        all_emb.append(avg)
        all_ids.append(barcode_to_cat[bc])

torch.save({"embeddings": np.array(all_emb, dtype=np.float32), "category_ids": np.array(all_ids, dtype=np.int64)}, str(RUNS / "reference_embeddings.pt"))
print(f"Saved {len(all_emb)} product embeddings")
PYEOF

echo "=== Training YOLOv8l nc=356 ==="
python3 -c "
import torch; _L=torch.load; torch.load=lambda *a,**k: _L(*a,**{**k,'weights_only':False})
from ultralytics import YOLO
model = YOLO('yolov8l.pt')
model.train(
    data='$HOME/data/data_nc356.yaml',
    imgsz=1280, batch=4, epochs=200, patience=40,
    lr0=0.001, lrf=0.01, cos_lr=True, warmup_epochs=5,
    mosaic=1.0, close_mosaic=15, mixup=0.15,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.5,
    translate=0.1, scale=0.4, fliplr=0.5, flipud=0.0, degrees=0.0,
    erasing=0.1, weight_decay=0.0005, max_det=500,
    project='$HOME/runs', name='yolov8l_nc356',
    save=True, plots=True, device=0,
)
"

echo "=== DONE ==="
