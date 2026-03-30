# Development Log — Traffic Congestion Classification
**Team:** Road Rangers
**Task:** 3-class congestion classification (Low / Medium / High) from drone intersection footage

---

## Table of Contents
1. [Project Setup & Dataset Exploration](#1-project-setup--dataset-exploration)
2. [Labeling Pipeline](#2-labeling-pipeline)
3. [Frame Extraction & Splits](#3-frame-extraction--splits)
4. [Dataset Statistics (Final)](#4-dataset-statistics-final)
5. [Training Configuration](#5-training-configuration)
6. [Experiment 1 — First Training Run (Overlays Present)](#6-experiment-1--first-training-run-overlays-present)
7. [Overlay Discovery & Removal](#7-overlay-discovery--removal)
8. [Experiment 2 — Retrain on Clean Frames (Final Results)](#8-experiment-2--retrain-on-clean-frames-final-results)
9. [Model Comparison: Before vs After Overlay Removal](#9-model-comparison-before-vs-after-overlay-removal)
10. [Key Findings & Analysis](#10-key-findings--analysis)

---

## 1. Project Setup & Dataset Exploration

**Dataset:** Waterloo Multi-Agent Traffic Dataset
**Source:** University of Waterloo — drone footage of urban intersections
**Format:** `.avi` video + `.db` SQLite annotation file per recording pair

**Pairs used:** 14 intersection pairs — IDs: 769, 770, 771, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785

**Video properties:**
- Frame rate: ~29.97 FPS
- Duration: ~5 minutes per pair
- Perspective: top-down drone view

**SQLite schema (relevant tables):**
- `TRACKS` — trajectory rows: `track_id`, `frame_id`, `x_center`, `y_center`, `speed`, `type`
- `tracksMeta` — track metadata: `id`, `type`, `first_appearance`, `last_appearance`

**Vehicle types included:** Car, Bus, Heavy Vehicle, Medium Vehicle, Motorcycle
**Excluded:** Pedestrian, Bicycle (not relevant to congestion)

---

## 2. Labeling Pipeline

Labels are derived programmatically from trajectory metadata — no manual annotation.

### Windowing
- Each recording split into non-overlapping **5-second windows**
- Window boundary: `frame_id` grouped by `floor(frame_id / (fps × 5))`

### Per-window features
| Feature | Description |
|---|---|
| `vehicle_count` | Number of unique vehicles (by track_id) active in window |
| `avg_speed` | Mean speed (m/s) across all trajectory rows in window |
| `stop_proxy` | Fraction of rows where `speed < 0.5 m/s` |

### Composite score
```
score = 0.4 × norm_count + 0.4 × (1 − norm_speed) + 0.2 × stop_proxy
```

- `norm_count` and `norm_speed` are **percentile-normalised within each pair** (5th–95th percentile range) to account for inter-pair differences in intersection size and traffic volume
- `stop_proxy` is already in [0, 1], used directly

### Class thresholds
| Class | Score range |
|---|---|
| Low | score ≤ 0.33 |
| Medium | 0.33 < score < 0.67 |
| High | score ≥ 0.67 |

**Output per pair:** `data/labels/per_pair/{pair_id}_window_labels_v1.csv`
**Combined output:** `data/labels/window_labels_v1_all.csv`

---

## 3. Frame Extraction & Splits

### Frame extraction
- **Strategy:** `multi` — 3 frames per window at evenly-spaced interior positions
- **Positions:** 25%, 50%, 75% through window duration
- **Resize:** 224×224 pixels (RGB, JPEG quality 95)
- **Output path:** `data/processed/frames/{pair_id}/{window_id}_f{00|01|02}.jpg`

**Rationale for 3 frames:** Single middle-frame extraction risks capturing an atypical moment (e.g., traffic light phase transition). Three frames give temporal coverage without tripling storage. Near-duplicate risk is mitigated by the window-level split strategy below.

### Train/Val/Test split
- **Method:** Window-level stratified split
- **Split key:** composite `(pair_id, window_id)` — all 3 frames from the same window are always in the same split
- **Rationale:** Frame-level split would allow near-identical frames (same window, 1–2 seconds apart) to appear in both train and test, causing data leakage
- **Ratio:** 70% train / 15% val / 15% test
- **Stratification:** by label class, ensuring class distribution is consistent across all splits

**Why not temporal split:** Congestion drifts continuously within a recording — a purely temporal split (first 70% = train) would cause entire congestion levels to be absent from val/test in some pairs.

---

## 4. Dataset Statistics (Final)

### Overall
| Metric | Value |
|---|---|
| Pairs | 14 |
| Total windows | 769 |
| Total samples (frames) | 2,296 |
| Frames per window | 3 |
| Image size | 224 × 224 |

### Class distribution (all splits combined)
| Class | Count | Percentage |
|---|---|---|
| Low | 398 | 17.3% |
| Medium | 1,292 | 56.3% |
| High | 606 | 26.4% |

### Split sizes
| Split | Samples | Windows |
|---|---|---|
| Train | 1,605 | ~538 |
| Val | 343 | ~115 |
| Test | 348 | ~116 |

**Note:** Class imbalance is real — Low is significantly underrepresented. This is addressed in training via class-weighted loss (see §5).

---

## 5. Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| LR scheduler | CosineAnnealingLR (`T_max=50`) |
| Loss | CrossEntropyLoss with class weights |
| Seed | 42 |
| Device | MPS (Apple Silicon) |

### Class weights
Inverse-frequency weighting computed from training set distribution. Upweights Low class (most underrepresented) to counteract the 17% / 56% / 27% imbalance.

### Data augmentation (train only)
- Random horizontal flip
- Random rotation ±10°
- Color jitter: brightness ±0.2, contrast ±0.2, saturation ±0.1

### Evaluation transforms (val/test)
- Resize to 224×224
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet stats)

### Best checkpoint selection
Saved when `val_acc` improves. Final evaluation uses best checkpoint, not last epoch.

---

## 6. Experiment 1 — First Training Run (Overlays Present)

**Dataset state:** Raw extracted frames — annotation overlays still present (see §7)
**All 4 models trained and evaluated on test set (348 samples)**

### Results

| Model | Test Acc | Macro F1 | Low F1 | Med F1 | High F1 | Params |
|---|---|---|---|---|---|---|
| Baseline CNN | 0.7816 | 0.7325 | 0.524 | 0.801 | 0.872 | 619K |
| MobileNetV2 | 0.7615 | 0.7218 | 0.551 | 0.787 | 0.828 | 2.2M |
| ResNet-50 | **0.7845** | **0.7615** | **0.634** | **0.795** | 0.856 | 23.5M |
| EfficientNet-B0 | 0.7672 | 0.7383 | 0.585 | 0.791 | **0.839** | 4.0M |

**→ Best model at this stage: ResNet-50** (highest test acc and macro F1)

### Observations
- ResNet-50 leads on Low F1 (0.634) — better at recovering the minority class
- Baseline CNN is competitive despite having 37× fewer parameters than ResNet-50
- MobileNetV2 underperforms relative to its architecture capacity
- All models struggle with Low class (F1 0.52–0.63)

---

## 7. Overlay Discovery & Removal

### Discovery
During frame inspection, it was identified that the source `.avi` files contain **red bounding boxes baked into the video frames** by the dataset provider's annotation tool. These are not a separate overlay layer — they are permanently composited into the pixel data.

**Visual signature:** High-saturation red rectangles (HSV hue ≈ 0°–8° and 172°–180°, saturation > 150, value > 150) drawn around each tracked vehicle in every frame.

**Problem:** Overlay density directly correlates with vehicle count, which correlates with congestion class. Models could achieve above-chance accuracy by simply counting red pixels rather than learning genuine visual traffic patterns. This is a form of dataset leakage through an annotation artefact.

**Evidence (post-removal):** Baseline CNN dropped most significantly after overlay removal (0.7816 → 0.7241, −0.0575 test acc), suggesting it relied heavily on overlay density. MobileNetV2 improved most (0.7615 → 0.7874, +0.0259), suggesting it was negatively affected by the overlays masking the underlying visual features it was trying to learn.

### Removal method
**Algorithm:** HSV colour masking + TELEA inpainting
**Script:** `src/preprocessing/remove_overlays.py`

```python
def remove_red_overlay(img_bgr):
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Red wraps around hue=0/180 in HSV, so two masks needed
    mask1 = cv2.inRange(hsv, np.array([0,   150, 150]), np.array([8,   255, 255]))
    mask2 = cv2.inRange(hsv, np.array([172, 150, 150]), np.array([180, 255, 255]))
    mask  = cv2.bitwise_or(mask1, mask2)
    mask  = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    return cv2.inpaint(img_bgr, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
```

- Dilation (2 iterations, 3×3 kernel) ensures box edges are fully covered before inpainting
- TELEA inpainting reconstructs masked regions using surrounding pixel statistics
- Applied in-place to all 2,296 frames; frames re-saved at JPEG quality 95

**All models retrained from scratch on cleaned frames.**

---

## 8. Experiment 2 — Retrain on Clean Frames (Final Results)

**Dataset state:** All 2,296 frames with overlays removed via HSV inpainting
**Training config:** identical to Experiment 1
**This is the canonical, final result set**

### Results

| Model | Test Acc | Macro F1 | Low F1 | Med F1 | High F1 | Params |
|---|---|---|---|---|---|---|
| Baseline CNN | 0.7241 | 0.7210 | 0.618 | 0.707 | 0.837 | 619K |
| **MobileNetV2** ★ | **0.7874** | **0.7659** | **0.650** | **0.800** | 0.847 | 2.2M |
| ResNet-50 | 0.7615 | 0.7380 | 0.598 | 0.774 | 0.842 | 23.5M |
| EfficientNet-B0 | 0.7759 | 0.7503 | 0.580 | 0.793 | **0.878** | 4.0M |

**→ Best model: MobileNetV2** (highest test acc 78.74% and macro F1 0.766)

### Per-class analysis (MobileNetV2, final)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | — | — | 0.650 | ~60 |
| Medium | — | — | 0.800 | ~196 |
| High | — | — | 0.847 | ~92 |
| **Macro avg** | | | **0.766** | 348 |

---

## 9. Model Comparison: Before vs After Overlay Removal

| Model | Acc (before) | Acc (after) | Δ Acc | F1 (before) | F1 (after) | Δ F1 |
|---|---|---|---|---|---|---|
| Baseline CNN | 0.7816 | 0.7241 | **−0.0575** | 0.7325 | 0.7210 | −0.0115 |
| MobileNetV2 | 0.7615 | 0.7874 | **+0.0259** | 0.7218 | 0.7659 | +0.0441 |
| ResNet-50 | 0.7845 | 0.7615 | −0.0230 | 0.7615 | 0.7380 | −0.0235 |
| EfficientNet-B0 | 0.7672 | 0.7759 | +0.0087 | 0.7383 | 0.7503 | +0.0120 |

### Ranking change
| | Before | After |
|---|---|---|
| 1st | ResNet-50 (0.7845) | **MobileNetV2 (0.7874)** |
| 2nd | Baseline CNN (0.7816) | EfficientNet-B0 (0.7759) |
| 3rd | EfficientNet-B0 (0.7672) | ResNet-50 (0.7615) |
| 4th | MobileNetV2 (0.7615) | Baseline CNN (0.7241) |

---

## 10. Key Findings & Analysis

### 10.1 Overlay leakage
The largest finding of the project. Red bounding boxes baked into source video constituted a spurious cue that models could exploit — higher overlay density = more vehicles = higher congestion. Baseline CNN (a shallow, 3-block network) dropped most after removal, indicating it was learning this shortcut rather than genuine visual features. MobileNetV2's improvement confirms it was being degraded by the overlays rather than helped.

### 10.2 Low class is consistently hardest
Across all models and both experiments, Low F1 is always the weakest class (0.52–0.65). Root cause: class imbalance — Low accounts for only 17.3% of samples (398 out of 2,296). Class-weighted loss partially compensates, but a larger Low-class sample pool would further improve this.

### 10.3 MobileNetV2 efficiency
MobileNetV2 achieves the best test accuracy (78.74%) with only 2.2M parameters — 10× fewer than ResNet-50 (23.5M, 76.15%). This suggests the depthwise-separable convolution structure in MobileNetV2 is well-suited to the spatial patterns present in top-down drone footage at this scale.

### 10.4 EfficientNet-B0 High F1
EfficientNet-B0 achieves the highest High F1 (0.878), above MobileNetV2 (0.847). If the application were focused solely on detecting severe congestion (e.g., triggering emergency response), EfficientNet-B0 would be preferred. For balanced general-purpose classification, MobileNetV2 wins on macro F1.

### 10.5 Why temporal split was not used
An early consideration was to split temporally (first 70% of frames in time = train). This was rejected because congestion is not uniformly distributed over time within a recording — peak periods cluster together, meaning a temporal split can leave entire congestion levels absent from val/test. Window-level stratified split is more statistically sound for this dataset.

### 10.6 Window-level split prevents leakage
3 frames extracted from the same 5-second window are near-identical (1–2 seconds apart). A frame-level random split would place these near-duplicates across train/test, inflating test accuracy. Keeping all frames from the same window in the same split prevents this.

---

## Appendix: Model Architectures

| Model | Type | Key layers | GradCAM target | Spatial resolution |
|---|---|---|---|---|
| Baseline CNN | Custom 3-block CNN | Conv→BN→ReLU×3, GAP, FC | `features[-1]` | 28×28 |
| MobileNetV2 | Depthwise-separable CNN | 19 bottleneck blocks, GAP | `features[13]` | 14×14 |
| ResNet-50 | Residual CNN | 4 residual stages, GAP | `layer3[-1]` | 14×14 |
| EfficientNet-B0 | Compound-scaled CNN | MBConv blocks, SE | `features[5]` | 14×14 |

All transfer models initialised with ImageNet pretrained weights; full backbone fine-tuned (no frozen layers).

---

## Appendix: File Structure

```
src/
├── preprocessing/
│   ├── discover_pairs.py       # Scans data/raw/ for valid (video + db) pairs
│   ├── generate_labels.py      # Step 2: congestion labels from SQLite per pair
│   ├── extract_frames.py       # Step 3: JPEG frame extraction (3/window, 224×224)
│   ├── build_splits.py         # Step 4: window-stratified 70/15/15 split
│   ├── process_pairs.py        # Orchestrator: runs steps 2–4 for all 14 pairs
│   └── remove_overlays.py      # HSV masking + TELEA inpainting (run once)
├── models/
│   ├── baseline_cnn.py         # Custom 3-block CNN
│   └── transfer_models.py      # MobileNetV2, ResNet-50, EfficientNet-B0
├── training/
│   └── train.py                # Training loop, cosine LR, class weighting, checkpoint
├── evaluation/
│   └── evaluate.py             # Test set eval, confusion matrix, classification report
└── gui/
    └── app.py                  # Gradio demo: classify + GradCAM + compare models
```
