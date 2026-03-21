#!/bin/bash
# Set up VM3 as eval server: val split + eval harness + run submissions on GPU
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Installing packages ==="
sudo apt-get update -qq && sudo apt-get install -y -qq unzip libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1
pip install ultralytics==8.1.0 timm==0.9.12 ensemble-boxes -q 2>&1 | tail -1

echo "=== Preparing data ==="
cd ~/data
[ -f coco_dataset.zip ] || { echo "ERROR: no data zip"; exit 1; }
[ -d train ] || unzip -qo coco_dataset.zip
[ -d metadata.json ] || unzip -qo product_images.zip 2>/dev/null || true

echo "=== Creating val split ==="
python3 << 'PYEOF'
import json, random, os
from pathlib import Path
from collections import defaultdict, Counter

random.seed(42)
DATA = Path(os.path.expanduser("~/data"))
EVAL = Path(os.path.expanduser("~/eval"))
EVAL.mkdir(exist_ok=True)

with open(DATA / "train" / "annotations.json") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
img_map = {img["id"]: img for img in images}
img_dir = DATA / "train" / "images"

# Group by image
img_anns = defaultdict(list)
for ann in annotations:
    img_anns[ann["image_id"]].append(ann)

img_cats = {img["id"]: set(a["category_id"] for a in img_anns[img["id"]]) for img in images}
cat_counts = Counter(a["category_id"] for a in annotations)

# Iterative stratification: rare categories first
val_ids = set()
cat_val_counts = Counter()
sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1])

for cat_id, count in sorted_cats:
    if count < 2 or cat_val_counts.get(cat_id, 0) > 0:
        continue
    candidates = [i for i in img_map if i not in val_ids and cat_id in img_cats.get(i, set())]
    if candidates:
        best = max(candidates, key=lambda i: len(img_cats[i] - set(c for c in cat_val_counts if cat_val_counts[c] > 0)))
        val_ids.add(best)
        for c in img_cats[best]:
            cat_val_counts[c] += len([a for a in img_anns[best] if a["category_id"] == c])

# Fill to 20%
unassigned = [i for i in img_map if i not in val_ids]
random.shuffle(unassigned)
target = int(len(images) * 0.20)
while len(val_ids) < target and unassigned:
    img_id = unassigned.pop()
    val_ids.add(img_id)

train_ids = set(img_map.keys()) - val_ids

# Create val image dir
val_dir = EVAL / "val_images"
val_dir.mkdir(exist_ok=True)
for img_id in val_ids:
    src = img_dir / img_map[img_id]["file_name"]
    dst = val_dir / img_map[img_id]["file_name"]
    if not dst.exists() and src.exists():
        os.symlink(str(src), str(dst))

# Save val ground truth
val_anns = [a for a in annotations if a["image_id"] in val_ids]
val_imgs = [img for img in images if img["id"] in val_ids]
with open(EVAL / "val_gt.json", "w") as f:
    json.dump({"images": val_imgs, "annotations": val_anns, "categories": coco["categories"]}, f)

# Save train COCO (for retraining)
train_anns = [a for a in annotations if a["image_id"] in train_ids]
train_imgs = [img for img in images if img["id"] in train_ids]
with open(EVAL / "train_annotations.json", "w") as f:
    json.dump({"images": train_imgs, "annotations": train_anns, "categories": coco["categories"]}, f)

covered = len([c for c in cat_counts if cat_val_counts.get(c, 0) > 0])
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
print(f"Val annotations: {len(val_anns)}, categories covered: {covered}/{len(cat_counts)}")
PYEOF

echo "=== Writing eval scorer ==="
cat << 'PYEOF' > ~/eval/score.py
"""Score a predictions.json against val ground truth. Competition metric."""
import json, sys, numpy as np
from pathlib import Path
from collections import defaultdict

def compute_iou(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xa,ya = max(x1,x2), max(y1,y2)
    xb,yb = min(x1+w1,x2+w2), min(y1+h1,y2+h2)
    inter = max(0,xb-xa)*max(0,yb-ya)
    union = w1*h1+w2*h2-inter
    return inter/union if union>0 else 0

def compute_ap(recs, precs):
    mrec = np.concatenate(([0.],recs,[1.]))
    mpre = np.concatenate(([1.],precs,[0.]))
    for i in range(len(mpre)-1,0,-1): mpre[i-1]=max(mpre[i-1],mpre[i])
    return sum(max(mpre[mrec>=t]) if len(mpre[mrec>=t])>0 else 0 for t in np.linspace(0,1,101))/101

def score(pred_path, gt_path):
    preds = json.loads(Path(pred_path).read_text())
    gt_data = json.loads(Path(gt_path).read_text())
    gts = [{"image_id":a["image_id"],"category_id":a["category_id"],"bbox":a["bbox"]} for a in gt_data["annotations"]]

    pb = defaultdict(list); gb = defaultdict(list)
    for p in preds: pb[p["image_id"]].append(p)
    for g in gts: gb[g["image_id"]].append(g)
    all_ids = set(list(pb.keys())+list(gb.keys()))

    # Detection mAP (class-agnostic)
    dp=[]; dtg=0
    for iid in all_ids:
        ps=sorted(pb[iid],key=lambda x:-x["score"]); gs=list(gb[iid]); dtg+=len(gs); m=[False]*len(gs)
        for p in ps:
            bi,bj=0,-1
            for j,g in enumerate(gs):
                if m[j]: continue
                iou=compute_iou(p["bbox"],g["bbox"])
                if iou>bi: bi,bj=iou,j
            if bi>=0.5 and bj>=0: m[bj]=True; dp.append((p["score"],True))
            else: dp.append((p["score"],False))
    dp.sort(key=lambda x:-x[0]); tp=0; dr,dpr=[],[]
    for i,(s,t) in enumerate(dp):
        if t: tp+=1
        dpr.append(tp/(i+1)); dr.append(tp/dtg if dtg>0 else 0)
    det_map = compute_ap(dr,dpr) if dtg>0 else 0

    # Classification mAP (per-category)
    gt_cats = set(g["category_id"] for g in gts); cas=[]
    for cid in gt_cats:
        cp=[]; ctg=0
        for iid in all_ids:
            ps=sorted([p for p in pb[iid] if p["category_id"]==cid],key=lambda x:-x["score"])
            gs=[g for g in gb[iid] if g["category_id"]==cid]; ctg+=len(gs); m=[False]*len(gs)
            for p in ps:
                bi,bj=0,-1
                for j,g in enumerate(gs):
                    if m[j]: continue
                    iou=compute_iou(p["bbox"],g["bbox"])
                    if iou>bi: bi,bj=iou,j
                if bi>=0.5 and bj>=0: m[bj]=True; cp.append((p["score"],True))
                else: cp.append((p["score"],False))
        if ctg==0: continue
        cp.sort(key=lambda x:-x[0]); tp=0; r,pr=[],[]
        for i,(s,t) in enumerate(cp):
            if t: tp+=1
            pr.append(tp/(i+1)); r.append(tp/ctg)
        cas.append(compute_ap(r,pr))
    cls_map = np.mean(cas) if cas else 0

    combined = 0.7*det_map + 0.3*cls_map
    return {"combined": round(combined,4), "det_mAP": round(det_map,4), "cls_mAP": round(cls_map,4), "preds": len(preds), "gt": len(gts)}

if __name__=="__main__":
    r = score(sys.argv[1], sys.argv[2])
    print(json.dumps(r, indent=2))
PYEOF

echo "=== Writing auto-eval runner ==="
cat << 'PYEOF' > ~/eval/auto_eval.py
"""Auto-eval: watches ~/eval/submissions/ for new submission dirs, runs and scores them."""
import json, subprocess, time, os, sys
from pathlib import Path

EVAL = Path(os.path.expanduser("~/eval"))
SUBS = EVAL / "submissions"
RESULTS = EVAL / "results"
VAL_IMGS = EVAL / "val_images"
VAL_GT = EVAL / "val_gt.json"
SUBS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

def run_and_score(sub_dir):
    name = sub_dir.name
    result_dir = RESULTS / name
    result_dir.mkdir(exist_ok=True)
    out_path = result_dir / "predictions.json"
    run_py = sub_dir / "run.py"

    if not run_py.exists():
        return {"error": "no run.py"}

    print(f"\n=== Evaluating {name} ===")
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, str(run_py.resolve()), "--input", str(VAL_IMGS.resolve()), "--output", str(out_path.resolve())],
            capture_output=True, text=True, timeout=600
        )
        elapsed = time.time() - t0
        if r.returncode != 0:
            print(f"FAILED: {r.stderr[-300:]}")
            return {"error": r.stderr[-300:], "runtime": round(elapsed,1)}
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT", "runtime": 600}

    # Score
    r2 = subprocess.run([sys.executable, str(EVAL / "score.py"), str(out_path), str(VAL_GT)], capture_output=True, text=True)
    scores = json.loads(r2.stdout)
    scores["runtime"] = round(elapsed, 1)
    scores["name"] = name

    with open(result_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    print(f"  Combined: {scores['combined']:.4f}  Det: {scores['det_mAP']:.4f}  Cls: {scores['cls_mAP']:.4f}  Runtime: {scores['runtime']}s")
    return scores

# Watch loop
print(f"Auto-eval watching {SUBS}")
print(f"Drop submission directories into {SUBS} to evaluate them.")
evaluated = set()
while True:
    for d in sorted(SUBS.iterdir()):
        if d.is_dir() and d.name not in evaluated:
            scores = run_and_score(d)
            evaluated.add(d.name)
            # Print comparison
            print(f"\n=== All Results ===")
            all_results = []
            for rd in sorted(RESULTS.iterdir()):
                sf = rd / "scores.json"
                if sf.exists():
                    all_results.append(json.loads(sf.read_text()))
            all_results.sort(key=lambda x: -x.get("combined", 0))
            for r in all_results:
                print(f"  {r.get('name','?'):<30s} {r.get('combined',0):.4f}  det={r.get('det_mAP',0):.4f}  cls={r.get('cls_mAP',0):.4f}  {r.get('runtime',0):.0f}s")
    time.sleep(10)
PYEOF

echo "=== DONE ==="
echo "Eval VM ready. To evaluate a submission:"
echo "  1. Upload submission dir to ~/eval/submissions/<name>/"
echo "  2. auto_eval.py will pick it up and score it"
echo "  3. Results in ~/eval/results/<name>/scores.json"
