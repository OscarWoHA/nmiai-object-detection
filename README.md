# NM i AI 2026 - Object Detection

Grocery shelf product detection for the Norwegian Championship in AI (NM i AI 2026).

**Final score: 0.9190** | Leader: 0.9266 | Scoring: `0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5`

## The Challenge

Detect and classify grocery products on store shelves. 248 training images (2000x1500), ~22,700 COCO annotations, 356 product categories with severe class imbalance (some categories have just 1 training example).

Sandbox constraints: ultralytics 8.1.0, L4 GPU, 8GB RAM, 300s timeout, 420MB max model size.

## Score Progression

| Version | Test Score | Architecture | Key Change |
|---------|-----------|-------------|------------|
| v1 | 0.5583 | YOLOv8m detection-only | Baseline, no classification |
| v2 | 0.7413 | YOLOv8m nc=356 + TTA | Added classification |
| v10 | 0.7661 | YOLOv8l nc=356 + sweep-tuned WBF | Better model + WBF parameter sweep |
| v11 | 0.9079 | 2x YOLOv8l ensemble | **Breakthrough**: two differently-trained models |
| v15 | 0.9098 | Original + copy-paste augmented model | Copy-paste with reference product images |
| v19 | 0.9123 | Original + stage-2 fine-tuned model | No-mosaic fine-tune on clean high-res data |
| v22 | 0.9146 | v19 + ConvNeXt tiebreaker | Heavy crop classifier fixes uncertain predictions |
| v25 | 0.9152 | v22 + sweep-tuned ConvNeXt params | Stricter thresholds, more crop padding |
| v26 | 0.9190 | Nuclear: 5 scales per model + ConvNeXt | Max compute, 10 YOLO passes per image |

## Architecture (v26 - Final)

```
                    +------------------+
                    |   Test Image     |
                    |   (2000x1500)    |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |    Model A       |          |    Model B       |
     |    YOLOv8l       |          |    YOLOv8l       |
     |    (Original     |          |    (Copy-paste   |
     |     training)    |          |     + stage-2    |
     |    85 MB .pt     |          |     fine-tune)   |
     |                  |          |    85 MB .pt     |
     +--------+---------+          +--------+---------+
              |                             |
              |  5 scales each:             |  5 scales each:
              |  640, 800, 1088,            |  640, 800, 1088,
              |  1280, 1440                 |  1280, 1440
              |                             |
              +----------+    +-------------+
                         |    |
                    +----v----v----+
                    |              |
                    |   Weighted   |
                    |   Box        |
                    |   Fusion     |
                    |              |
                    |  iou=0.5     |
                    |  conf=avg    |
                    |  skip=0.005  |
                    |              |
                    +------+-------+
                           |
                           v
                    +------+-------+
                    |  For each    |
                    |  detection   |
                    |  with YOLO   |
                    |  conf < 0.8  |
                    +------+-------+
                           |
                    +------v-------+
                    |  ConvNeXt    |
                    |  Small       |
                    |  (93.5% acc) |
                    |  100 MB FP16 |
                    |              |
                    |  Crop with   |
                    |  10% padding |
                    |  Pad to      |
                    |  square      |
                    +------+-------+
                           |
                    +------v-------+
                    |  Tiebreaker  |
                    |  Logic       |
                    |              |
                    |  If YOLO<0.4 |
                    |  & Conv>0.85:|
                    |  use ConvNeXt|
                    |              |
                    |  If YOLO<0.8 |
                    |  & Conv>0.9: |
                    |  use ConvNeXt|
                    +------+-------+
                           |
                    +------v-------+
                    | predictions  |
                    | .json        |
                    +--------------+
```

## All Architectures Tried

### Single Model (v1-v10)
```
Image -> YOLOv8 (single model, single or multi-scale) -> predictions
```
Best: v10 at 0.7661. Limited by single model overfitting.

### Two-Model Ensemble (v11-v20)
```
Image -> Model A (multi-scale) -+
                                 +-> WBF merge -> predictions
Image -> Model B (multi-scale) -+
```
Breakthrough at v11 (0.9079). Different training = different errors = ensemble cancels them out.

Variants tested:
- v11: Original + GPT-recipe (different augmentation)
- v14: 3-model ensemble (worse, too much noise)
- v15: Original + copy-paste augmented
- v16: 2x copy-paste models (worse, lost original's real-data signal)
- v19: Original + stage-2 fine-tuned (best two-model: 0.9123)
- v20: 2x stage-2 models (worse, same problem as v16)

### Two-Model + ConvNeXt Tiebreaker (v22-v26)
```
Image -> Model A (multi-scale) -+
                                 +-> WBF merge -> ConvNeXt tiebreaker -> predictions
Image -> Model B (multi-scale) -+                      |
                                               Crop uncertain boxes,
                                               classify with ConvNeXt,
                                               override if confident
```
v22 (0.9146) to v26 (0.9190). ConvNeXt reads product text/logos that YOLO misses.

### Approaches That Failed
- **DINOv2 prototype matching** (v4: 0.6863): Reference image matching can't compete with end-to-end YOLO classification
- **Crop classifier replacement** (v8: 0.9272 val, not submitted): mobilenetv3 too weak to replace YOLO's classification
- **Agnostic WBF + class voting** (v7/v13): Hurt classification on val
- **Spatial row voting**: Products don't repeat as consistently as expected
- **Test-time BN adaptation**: Corrupted batch norm stats at wrong resolution
- **Sweep-optimized WBF on ensemble**: Params overfit to val set

## Key Learnings

1. **The original model is irreplaceable.** Every submission that dropped it scored lower. It provides authentic signal from real shelf data that augmented models can't replicate.

2. **Ensembles generalize dramatically better.** Single model val-to-test ratio: ~0.82. Ensemble ratio: ~0.94. Different training errors cancel out.

3. **Val score is a poor predictor of test score.** We hit 0.99+ on val but only 0.92 on test. The 49 val images overlap with training data.

4. **Copy-paste augmentation helps the partner model**, not the primary. Pasting reference product images onto shelf images boosted rare classes (41 categories with 1 example).

5. **Stage-2 fine-tuning on clean data** improves generalization. Training on original images at 1536 resolution without mosaic/mixup "makes training look like test."

6. **ConvNeXt tiebreaker > ConvNeXt replacement.** A 93% accurate crop classifier can't replace YOLO's end-to-end classification (which sees shelf context), but it can fix YOLO's uncertain predictions.

7. **Use your full runtime budget.** v26 used 10 YOLO passes (5 scales x 2 models) vs v22's 6 passes, and scored +0.004 higher.

## Project Structure

```
.
├── README.md                    # This file
├── CLAUDE.md                    # Competition rules and constraints
├── architecture.md              # Detailed architecture diagrams
├── data/                        # Category mappings, run logs
├── scripts/
│   ├── training/                # Model training scripts
│   ├── eval/                    # Evaluation harness and sweeps
│   ├── gcp/                     # GCP VM setup and training
│   └── monitoring/              # Dashboard, autopilot, overnight automation
├── submissions/                 # All 26+ submission run.py files
│   ├── submission_v1/           # Detection-only baseline
│   ├── submission_v11_ensemble/ # Breakthrough 2-model ensemble
│   ├── submission_v25_tuned/    # Best tuned submission
│   ├── submission_v26_nuclear/  # Final "nuclear" submission
│   └── ...                      # All other variants
└── logs/                        # Monitoring alerts
```

## Model Weights

Weights are hosted on HuggingFace: [oscarwoha/nmiai-object-detection-weights](https://huggingface.co/oscarwoha/nmiai-object-detection-weights)

## Infrastructure

- 4 GCP L4 GPU VMs running in parallel
- Live web dashboard (WebSocket) tracking training progress and leaderboard
- Automated eval harness matching competition scoring
- Overnight autopilot script for monitoring and auto-submission
- Consulted GPT-5.4 (Codex CLI) and Gemini 3.1 Pro for architecture advice

## Team

**Paralov - studs.gg** (NM i AI 2026)

Built with [Claude Code](https://claude.ai/claude-code) (Claude Opus 4.6, 1M context).
