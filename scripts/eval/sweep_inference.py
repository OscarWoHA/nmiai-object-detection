"""PATH 1: Sweep inference params on v5 model to find optimal config.

Tests all combinations of:
- WBF iou_thr: [0.45, 0.50, 0.55, 0.60, 0.65]
- conf threshold: [0.005, 0.01, 0.02]
- scales: [2-scale, 3-scale, 4-scale]
- conf_type: [max, avg]

Runs on eval VM, reports best combo.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image
import itertools
import time
import sys


def run_scale(model, img_path, device, imgsz, conf, w, h):
    results = model(str(img_path), device=device, verbose=False,
                    imgsz=imgsz, conf=conf, iou=0.5, max_det=500)
    boxes, scores, labels = [], [], []
    for r in results:
        if r.boxes is None: continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            boxes.append([x1/w, y1/h, x2/w, y2/h])
            scores.append(float(r.boxes.conf[i].item()))
            labels.append(int(r.boxes.cls[i].item()))
    return boxes, scores, labels


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


def score_predictions(predictions, ground_truths):
    from collections import defaultdict
    pb = defaultdict(list); gb = defaultdict(list)
    for p in predictions: pb[p["image_id"]].append(p)
    for g in ground_truths: gb[g["image_id"]].append(g)
    all_ids = set(list(pb.keys())+list(gb.keys()))

    # Detection
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

    # Classification
    gt_cats = set(g["category_id"] for g in ground_truths); cas=[]
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
    return 0.7*det_map + 0.3*cls_map, det_map, cls_map


def main():
    import os
    EVAL = Path(os.path.expanduser("~/eval"))
    VAL_IMGS = EVAL / "val_images"
    VAL_GT = EVAL / "val_gt.json"

    with open(VAL_GT) as f:
        gt_data = json.load(f)
    ground_truths = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"]} for a in gt_data["annotations"]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(os.path.expanduser("~/eval/submissions/v5_yolov8l_nc356_ep46/yolo_best.pt"))

    # Pre-compute detections at all scales
    print("Pre-computing detections at all scales...")
    img_paths = sorted([p for p in VAL_IMGS.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    scale_configs = [
        (640, 0.03),
        (800, 0.02),
        (960, 0.02),
        (1088, 0.015),
        (1280, 0.01),
        (1440, 0.01),
        (1536, 0.015),
    ]

    # Cache all scale results
    cached = {}
    for imgsz, conf in scale_configs:
        key = f"{imgsz}_{conf}"
        cached[key] = {}
        t0 = time.time()
        for img_path in img_paths:
            img = Image.open(img_path)
            w, h = img.size
            image_id = int(img_path.stem.split("_")[-1])
            boxes, scores, labels = run_scale(model, img_path, device, imgsz, conf, w, h)
            cached[key][image_id] = (boxes, scores, labels, w, h)
        print(f"  Scale {imgsz} conf={conf}: {time.time()-t0:.1f}s")

    # Now sweep WBF parameters
    print("\nSweeping WBF parameters...")
    best_score = 0
    best_config = None

    scale_combos = [
        [("960_0.02", 1.0), ("1280_0.01", 2.0), ("1536_0.015", 1.5)],  # current best
        [("1280_0.01", 2.0), ("1536_0.015", 1.5)],  # 2-scale
        [("960_0.02", 0.8), ("1088_0.015", 1.0), ("1280_0.01", 2.0), ("1536_0.015", 1.5)],  # 4-scale
        [("800_0.02", 0.8), ("1088_0.015", 1.2), ("1280_0.01", 2.0), ("1440_0.01", 1.2)],  # alt 4-scale
        [("960_0.02", 1.0), ("1280_0.01", 2.5), ("1536_0.015", 1.0)],  # reweighted 3-scale
    ]

    wbf_ious = [0.45, 0.50, 0.55, 0.60]
    conf_types = ["max", "avg"]
    skip_thrs = [0.005, 0.01, 0.02]

    total = len(scale_combos) * len(wbf_ious) * len(conf_types) * len(skip_thrs)
    tested = 0

    for scales in scale_combos:
        for wbf_iou in wbf_ious:
            for conf_type in conf_types:
                for skip_thr in skip_thrs:
                    tested += 1
                    predictions = []

                    for img_path in img_paths:
                        image_id = int(img_path.stem.split("_")[-1])
                        all_b, all_s, all_l, weights = [], [], [], []

                        for scale_key, weight in scales:
                            if image_id not in cached[scale_key]:
                                continue
                            boxes, scores, labels, w, h = cached[scale_key][image_id]
                            if boxes:
                                all_b.append(np.array(boxes, dtype=np.float32))
                                all_s.append(np.array(scores, dtype=np.float32))
                                all_l.append(np.array(labels, dtype=np.float32))
                                weights.append(weight)

                        if not all_b:
                            continue

                        mb, ms, ml = weighted_boxes_fusion(
                            all_b, all_s, all_l,
                            weights=weights, iou_thr=wbf_iou,
                            skip_box_thr=skip_thr, conf_type=conf_type,
                        )

                        for i in range(len(mb)):
                            box = mb[i]
                            predictions.append({
                                "image_id": image_id,
                                "category_id": int(ml[i]),
                                "bbox": [round(box[0]*w,1), round(box[1]*h,1),
                                         round((box[2]-box[0])*w,1), round((box[3]-box[1])*h,1)],
                                "score": round(float(ms[i]), 4),
                            })

                    combined, det, cls = score_predictions(predictions, ground_truths)

                    if combined > best_score:
                        best_score = combined
                        scale_names = "+".join(s[0] for s in scales)
                        best_config = f"scales={scale_names} wbf_iou={wbf_iou} conf_type={conf_type} skip={skip_thr}"
                        print(f"  NEW BEST [{tested}/{total}]: {combined:.4f} (det={det:.4f} cls={cls:.4f}) | {best_config}")

    print(f"\n=== BEST CONFIG ===")
    print(f"Score: {best_score:.4f}")
    print(f"Config: {best_config}")


if __name__ == "__main__":
    main()
