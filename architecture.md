# How Our Object Detection Pipeline Works

## The Big Picture

We have **248 photos of grocery store shelves**. Each photo has ~92 products on it.
Our job: find every product (draw a box around it) and identify what it is (assign a category).

**Scoring:** 70% for finding products + 30% for identifying them correctly.

---

## Training (happens on GCP VMs, takes hours)

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING DATA                         │
│                                                          │
│  248 shelf photos (2000×1500 pixels each)                │
│  22,700 labeled boxes ("this box is Corn Flakes")        │
│  356 product categories (some have 400+ examples,        │
│  others have just 1 — huge imbalance!)                   │
│                                                          │
│  + 327 reference product photos (multi-angle studio      │
│    shots of individual products, organized by barcode)   │
└─────────────┬───────────────────────┬────────────────────┘
              │                       │
              ▼                       ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│   Model A Training   │  │     Model B Training          │
│                      │  │                                │
│  YOLOv8l (large)     │  │  YOLOv8l (large)              │
│  "GPT recipe":       │  │  "Original recipe":            │
│  • No horizontal     │  │  • Default ultralytics         │
│    flip (because     │  │    augmentation                │
│    mirrored text     │  │  • Standard hyperparams        │
│    doesn't exist     │  │                                │
│    in real life)     │  │  Trained for ~46 epochs        │
│  • Low mixup         │  │                                │
│  • Rare class        │  │  This model is our "bread      │
│    oversampling      │  │  and butter" — trained the     │
│  • Longer training   │  │  normal way, works great       │
│                      │  │                                │
│  85 MB weights       │  │  85 MB weights                 │
└──────────┬───────────┘  └──────────────┬─────────────────┘
           │                              │
           │  These two models learned    │
           │  DIFFERENT things because    │
           │  they were trained with      │
           │  different augmentation.     │
           │  That's the whole point!     │
           │                              │
           ▼                              ▼
      model_a.pt                    model_b.pt
      (85 MB)                       (85 MB)
```

---

## Inference (happens in competition sandbox, must finish in 300 seconds)

```
                    ┌─────────────────┐
                    │   Test Image     │
                    │   (2000×1500)    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Model A  │  │ Model A  │  │ Model A  │    Model A runs
        │ @1088px  │  │ @1280px  │  │ @1440px  │    at 3-4 different
        │          │  │          │  │          │    resolutions
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │
             │    ┌────────┼─────────┐   │
             │    ▼        ▼         ▼   │
             │  ┌──────┐ ┌──────┐ ┌──────┐
             │  │Mod B │ │Mod B │ │Mod B │    Model B also runs
             │  │@1088 │ │@1280 │ │@1440 │    at multiple resolutions
             │  └──┬───┘ └──┬───┘ └──┬───┘
             │     │        │        │
             ▼     ▼        ▼        ▼
        ┌─────────────────────────────────┐
        │                                  │
        │   ALL predictions collected      │
        │   (hundreds of boxes per image   │
        │    from all model+scale combos)  │
        │                                  │
        │   Many boxes overlap because     │
        │   multiple models/scales found   │
        │   the same product               │
        │                                  │
        └───────────────┬──────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────┐
        │                                  │
        │   WEIGHTED BOX FUSION (WBF)      │
        │                                  │
        │   Smart merging algorithm:       │
        │   • Groups overlapping boxes     │
        │   • Averages their coordinates   │
        │   • Picks best class by vote     │
        │   • Keeps confidence scores      │
        │                                  │
        │   Parameters (sweep-optimized):  │
        │   • iou_thr=0.5                  │
        │   • conf_type='avg'              │
        │   • skip_box_thr=0.005           │
        │                                  │
        │   Result: ~100 clean boxes       │
        │   per image, one per product     │
        │                                  │
        └───────────────┬──────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────┐
        │   predictions.json               │
        │                                  │
        │   For each detected product:     │
        │   • image_id (which image)       │
        │   • category_id (what product)   │
        │   • bbox [x, y, width, height]   │
        │   • score (how confident)        │
        └───────────────┬──────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────┐
        │          COMPETITION             │
        │          SCORING                 │
        │                                  │
        │  "Did you FIND products?"        │
        │   → Detection mAP@0.5           │
        │   → Worth 70% of score           │
        │   → Just checks: is there a      │
        │     predicted box overlapping     │
        │     a real product? (IoU ≥ 0.5)  │
        │   → Doesn't care about class     │
        │                                  │
        │  "Did you IDENTIFY products?"    │
        │   → Classification mAP@0.5      │
        │   → Worth 30% of score           │
        │   → Checks: box overlaps AND     │
        │     predicted the right category │
        │                                  │
        │  Final = 0.7 × det + 0.3 × cls  │
        └──────────────────────────────────┘
```

---

## Why Two Models Beat One

```
Single model on training images:  val mAP = 0.94  →  test = 0.77  (ratio: 0.82)
Two-model ensemble:               val mAP = 0.97  →  test = 0.91  (ratio: 0.94)
```

A single model memorizes the 248 training images. On unseen test images it drops 18%.

Two models trained differently make DIFFERENT mistakes. When we merge their
predictions, the mistakes cancel out but the correct detections reinforce.
Result: only 6% drop instead of 18%.

---

## Copy-Paste Augmentation (what we're trying now)

```
Problem: 41 categories have just 1 training example!
         The model can't learn what it's never seen.

Solution: We have studio photos of 327 products.
          Cut them out and paste onto shelf images.

Before:  "Rare Coffee Brand" → 1 training box
After:   "Rare Coffee Brand" → 20+ training boxes (synthetic)

┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐
│ Reference    │     │ Random shelf │     │ Augmented shelf      │
│ product      │  +  │ image        │  =  │ image with product   │
│ photo        │     │              │     │ pasted on it         │
│ (studio)     │     │              │     │                      │
└─────────────┘     └──────────────┘     └──────────────────────┘
```

---

## Current Scores

| Submission | Val Score | Test Score | What |
|-----------|-----------|------------|------|
| v2 | 0.8695 | 0.7413 | Single YOLOv8m |
| v10 | 0.9403 | 0.7661 | Single YOLOv8l, tuned WBF |
| **v11** | **0.9672** | **0.9079** | **2-model ensemble (BEST)** |
| v14 | 0.9693 | 0.9050 | 3-model ensemble (worse!) |
| Leader | — | 0.9255 | Gap: 0.018 |
