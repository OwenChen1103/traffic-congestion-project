# Development Log — Traffic Congestion Classification
**Team:** Road Rangers
**Task:** 3-class congestion classification (Low / Medium / High) from intersection footage

---

## Table of Contents
1. [Project Setup & Dataset Exploration](#1-project-setup--dataset-exploration)
2. [Labeling Pipeline](#2-labeling-pipeline)
3. [Frame Extraction & Splits](#3-frame-extraction--splits)
4. [Dataset Statistics (Drone Dataset)](#4-dataset-statistics-drone-dataset)
5. [Training Configuration](#5-training-configuration)
6. [Experiment 1 — First Training Run (Overlays Present)](#6-experiment-1--first-training-run-overlays-present)
7. [Overlay Discovery & Removal](#7-overlay-discovery--removal)
8. [Experiment 2 — Retrain on Clean Frames](#8-experiment-2--retrain-on-clean-frames-final-results)
9. [Model Comparison: Before vs After Overlay Removal](#9-model-comparison-before-vs-after-overlay-removal)
10. [Experiment 3 — Ensemble + Test-Time Augmentation](#10-experiment-3--ensemble--test-time-augmentation-no-retraining)
11. [Experiment 4 — WeightedRandomSampler + Stronger Augmentation (Failed)](#11-experiment-4--weightedrandomsampler--stronger-augmentation-failed-reverted)
12. [Experiment 5 — Live NSW Traffic Camera Pipeline](#12-experiment-5--live-nsw-traffic-camera-pipeline)
13. [Key Findings & Analysis](#13-key-findings--analysis)

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

## 4. Dataset Statistics (Drone Dataset)

### Overall
| Metric | Value |
|---|---|
| Pairs | 14 |
| Total windows | 777 |
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
**Script:** `src/drone_pipeline/remove_overlays.py`

```python
def remove_red_overlay(img_bgr):
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
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
**This is the canonical, final result set for the drone dataset**

### Results

| Model | Test Acc | Macro F1 | Low F1 | Med F1 | High F1 | Params |
|---|---|---|---|---|---|---|
| Baseline CNN | 0.7241 | 0.7210 | 0.618 | 0.707 | 0.837 | 619K |
| **MobileNetV2** ★ | **0.7874** | **0.7659** | **0.650** | **0.800** | 0.847 | 2.2M |
| ResNet-50 | 0.7615 | 0.7380 | 0.598 | 0.774 | 0.842 | 23.5M |
| EfficientNet-B0 | 0.7759 | 0.7503 | 0.580 | 0.793 | **0.878** | 4.0M |

**→ Best model: MobileNetV2** (highest test acc 78.74% and macro F1 0.766)

---

## 9. Model Comparison: Before vs After Overlay Removal

| Model | Acc (before) | Acc (after) | Δ Acc | F1 (before) | F1 (after) | Δ F1 |
|---|---|---|---|---|---|---|
| Baseline CNN | 0.7816 | 0.7241 | **−0.0575** | 0.7325 | 0.7210 | −0.0115 |
| MobileNetV2 | 0.7615 | 0.7874 | **+0.0259** | 0.7218 | 0.7659 | +0.0441 |
| ResNet-50 | 0.7845 | 0.7615 | −0.0230 | 0.7615 | 0.7380 | −0.0235 |
| EfficientNet-B0 | 0.7672 | 0.7759 | +0.0087 | 0.7383 | 0.7503 | +0.0120 |

---

## 10. Experiment 3 — Ensemble + Test-Time Augmentation (No Retraining)

**Goal:** Improve accuracy without retraining by combining all 4 existing checkpoints and applying TTA.

### Ensemble
Average softmax probabilities from all 4 models before taking argmax. No retraining required.

### Test-Time Augmentation (TTA)
Each image is augmented 5 ways at inference time; probabilities are averaged:
1. Original (no augmentation)
2. Horizontal flip
3. Vertical flip
4. +5° rotation
5. −5° rotation

### Results

| Mode | Test Acc | Macro F1 | Low F1 | Med F1 | High F1 |
|---|---|---|---|---|---|
| MobileNetV2 (single, no TTA) | 0.7874 | 0.7659 | 0.650 | 0.800 | 0.847 |
| Ensemble (4 models, no TTA) | 0.8046 | 0.7845 | 0.672 | 0.817 | 0.864 |
| **Ensemble + TTA ⚡** | **0.8218** | **0.7992** | **0.667** | **0.839** | **0.893** |

**→ Ensemble + TTA: 82.18% test accuracy** — +3.44% over best single model, no retraining needed.

---

## 11. Experiment 4 — WeightedRandomSampler + Stronger Augmentation (Failed, Reverted)

**Goal:** Improve Low class F1 by balancing training batches and applying stronger augmentation.

### Results

| Model | Test Acc (Exp 2) | Test Acc (Exp 4) | Δ | Notes |
|---|---|---|---|---|
| Baseline CNN | 0.7241 | **0.4971** | −0.227 | Collapsed |
| MobileNetV2 | 0.7874 | 0.7787 | −0.009 | Minor drop |
| ResNet-50 | 0.7615 | **0.5690** | −0.193 | Collapsed |
| EfficientNet-B0 | 0.7759 | 0.7644 | −0.012 | Minor drop |

### Failure analysis
WeightedRandomSampler trains on an approximately uniform 33%/33%/33% class distribution, but the test set retains the natural 17%/56%/27% distribution. The model's learned decision boundaries are calibrated for a balanced world that does not exist at inference time. Medium, which accounts for 56% of real traffic states, was systematically underweighted in training relative to its importance at test time.

**Decision:** Reverted all changes. This negative result demonstrates that naive oversampling of minority classes can be counter-productive when the test distribution is imbalanced.

---

## 12. Experiment 5 — Live NSW Traffic Camera Pipeline

### Motivation
The drone dataset (2,296 samples, top-down view, controlled conditions) is limited in diversity. To validate generalisation and build a more realistic system, a second pipeline was built using **live NSW traffic camera footage** via the TfNSW Open Data API.

Key differences from the drone dataset:
- Street-level fixed camera perspective (not top-down drone)
- Real-world conditions: weather, glare, day/night, varying traffic volumes
- Labels derived from YOLO vehicle detection (not SQLite annotations)
- 10 cameras across Sydney and Wollongong covering different road types

---

### 12.1 Camera Selection

| Camera | Role | Region | Road Type |
|---|---|---|---|
| parramatta_road_camperdown | train | SYD_MET | Urban arterial |
| hume_highway_bankstown | train | SYD_SOUTH | Wide arterial with median |
| anzac_parade_moore_park | train | SYD_MET | Wider urban road |
| james_ruse_drive_rosehill | train | SYD_WEST | 4+ lane highway |
| princes_highway_st_peters_n | train* | SYD_MET | Excluded (severe glare) |
| city_road_newtown | train | SYD_MET | Urban tram road |
| king_georges_road_hurstville | train | SYD_SOUTH | Suburban 2-lane |
| 5_ways_miranda | train | SYD_SOUTH | 5-way intersection |
| memorial_drive_towradgi | **test** | REG_WOLLONGONG | Regional highway |
| shellharbour_road_warilla | **test** | REG_WOLLONGONG | Regional arterial |

\* `princes_highway_st_peters_n` collected but excluded from training — YOLO blind due to severe lens glare.

Wollongong cameras used as test set to evaluate cross-region generalisation (Sydney → Wollongong).

---

### 12.2 Data Collection

**Script:** `src/live_pipeline/collect.py`
**Method:** TfNSW camera API polled every 15 seconds → 4 frames per 1-minute window per camera

**Collection sessions:**
| Date | Time | Duration | Notes |
|---|---|---|---|
| 3 Apr 2026 (Sun) | ~08:00–18:00 | ~10h | Weekend baseline, low traffic |
| 7 Apr 2026 (Mon) | 07:30–09:00 | 90 min | Weekday morning peak |
| 7 Apr 2026 (Mon) | 11:30–13:00 | 90 min | Weekday midday off-peak |
| 7 Apr 2026 (Mon) | 17:00–18:00 | 60 min | Weekday afternoon peak (delayed start) |

**Total frames collected:** 19,919 across 10 cameras (~500 windows per camera)

---

### 12.3 Vehicle Detection

**Script:** `src/live_pipeline/detect.py`
**Model:** YOLOv8n (conf threshold = 0.3)
**Vehicle classes (COCO):** 2=car, 3=motorcycle, 5=bus, 7=truck

**Features per frame:**
| Feature | Description |
|---|---|
| `vehicle_count` | Vehicles detected within valid ROI |
| `bbox_area_ratio` | Sum of bbox areas / frame area |
| `bottom_roi_count` | Vehicles in bottom third of frame |
| `mean_brightness` | Mean pixel intensity (0–255), used for nighttime filtering |

#### Camera ROI and Exclusion Zones

**Problem encountered:** `5_ways_miranda` has a car dealership visible in the frame. Parked dealership vehicles were being counted as road traffic, inflating vehicle counts and polluting labels.

**Solution attempts:**
1. X-axis cutoff at `x < 0.28` — insufficient, dealership still counted
2. X-axis cutoff at `x < 0.38` — too aggressive, legitimate road vehicles on left lane were excluded (visible as red dashed boxes in manual review tool)
3. **Final solution:** Corner exclusion zone `(x < 0.28, y < 0.55)` — excludes only the top-left rectangle where the dealership sits, preserving left-lane road vehicles

```python
CAMERA_EXCLUDE = {
    "5_ways_miranda": [(0.0, 0.0, 0.28, 0.55)],  # top-left corner = dealership
}
```

#### Filters applied
- **Nighttime filter:** frames with `mean_brightness < 80` excluded
- **Time filter:** only 06:00–18:00 retained (removes pre-dawn and post-dusk)
- After filters: **15,881 / 19,919 frames** retained (79.7%)

---

### 12.4 Labeling Design Decision: Per-Frame vs Window-Average

**Initial approach:** Window-level averaging (same as drone pipeline)
- Aggregate 4 frames into window features → assign one label per window → all 4 frames share that label

**Problem discovered:** Label-image mismatch. A window could have `avg_vehicle_count = 5` (low) but contain a frame with `vehicle_count = 16`. The review tool showed `LOW v=16` — visually wrong and misleading for the CNN which sees only individual frames.

**Root cause:** A CNN image classifier sees one frame at inference time. Its input-output pair should be (this frame's pixels) → (this frame's congestion). Window averaging is a temporal construct that doesn't align with the model's view of the world.

**Decision:** Switch to **per-frame labeling** — apply camera thresholds directly to each frame's `vehicle_count`.

```
if vehicle_count ≤ low_max  → "low"
if vehicle_count ≥ high_min → "high"
else                        → "medium"
```

This ensures the label assigned to each training image reflects what is actually visible in that image.

---

### 12.5 Per-Camera Threshold Calibration

Thresholds set by visual inspection using `src/live_pipeline/manual_label.py` — a camera review tool that shows sample LOW / MEDIUM / HIGH frames per camera with YOLO bounding boxes and vehicle counts.

| Camera | low_max | high_min | Notes |
|---|---|---|---|
| james_ruse_drive_rosehill | 8 | 20 | 4+ lane highway, high capacity |
| hume_highway_bankstown | 6 | 15 | Wide arterial |
| 5_ways_miranda | 7 | 15 | ROI-corrected counts |
| parramatta_road_camperdown | 7 | 17 | Urban arterial |
| king_georges_road_hurstville | 3 | 10 | Suburban 2-lane, low capacity |
| city_road_newtown | 5 | 15 | Urban tram road |
| anzac_parade_moore_park | 5 | 15 | Wider urban road |
| memorial_drive_towradgi | 5 | 13 | Regional highway |
| shellharbour_road_warilla | 7 | 15 | Regional arterial |
| princes_highway_st_peters_n | 5 | 15 | Excluded from train |

**Known issue:** `city_road_newtown` never reached `high_min=15` in the collected data — 0 high frames. This camera appears to have low baseline traffic volume. Awaiting more weekday peak-hour data to confirm.

---

### 12.6 Dataset Statistics (Live Dataset, Final)

**Total frames:** 19,919
**After filters:** 15,881
**Train cameras:** 8 (excluding princes_highway)
**Test cameras:** 2 (Wollongong region)

| Split | Frames | Low | Medium | High |
|---|---|---|---|---|
| Train | 8,507 | 2,288 | 3,983 | 2,236 |
| Val | 1,505 | 412 | 702 | 391 |
| Test | 3,894 | 1,150 | 1,510 | 1,234 |

Overall distribution after thresholding: low 31% / med 43% / high 26% — substantially more balanced than the drone dataset (17% / 56% / 27%).

---

### 12.7 Split Strategy

Train cameras are split **window-level** (85% train / 15% val top-up) to avoid leakage — all frames from a window stay together. Majority label per window used for stratification. Test cameras are never seen during training, providing a true cross-camera generalisation test.

---

### 12.8 Training & Results

All 4 models trained from scratch on the live dataset using the same hyperparameter configuration as Experiments 1–2. Class-weighted CrossEntropyLoss applied.

#### Bug fix: evaluate.py split directory
`evaluate.py` was hardcoded to read from `data/processed/splits` (drone dataset). When evaluating live-trained models without `--split-dir`, test accuracy appeared as ~40% (model trained on live data, evaluated on drone data). Fixed by adding `--split-dir` argument.

```bash
python src/evaluation/evaluate.py --model efficientnet_b0 --split-dir data/live/splits
```

#### Results

| Model | Test Acc | Macro F1 | Low F1 | Med F1 | High F1 |
|---|---|---|---|---|---|
| Baseline CNN | 0.7201 | 0.7246 | 0.822 | 0.636 | 0.716 |
| MobileNetV2 | 0.8143 | 0.8204 | 0.880 | 0.764 | 0.818 |
| ResNet-50 | 0.7686 | 0.7778 | 0.845 | 0.715 | 0.773 |
| EfficientNet-B0 | 0.8290 | 0.8340 | 0.899 | 0.768 | 0.835 |
| Ensemble (4 models) | 0.8249 | 0.8308 | 0.891 | 0.776 | 0.825 |
| **EfficientNet-B0 + TTA** ⚡ | **0.8331** | **0.8375** | 0.898 | 0.774 | **0.841** |

**→ Best model: EfficientNet-B0 + TTA, 83.31% test accuracy**

#### Key observations
- EfficientNet-B0 is the best single model — reversed from the drone dataset where MobileNetV2 led
- Ensemble does not outperform EfficientNet-B0 alone; Baseline CNN and ResNet50 drag the average down
- TTA gives +0.4% consistent improvement across all classes
- Medium remains the hardest class (lowest F1) — boundary with Low and High is inherently ambiguous
- High class dramatically improved: from 15% of train data (weekend-only collection) to 26% after adding weekday peak sessions. Precision improved from 0.678 → 0.809 once more high samples were available.

---

### 12.9 Bugs & Issues Encountered

| Issue | Root Cause | Fix |
|---|---|---|
| `evaluate.py` showing 40% on live-trained model | Default split dir pointed to drone dataset splits | Added `--split-dir` CLI argument |
| `manual_label.py` KeyError: `image_path` | Frame labels CSV uses `file_path`, old code used `image_path` | Renamed all references |
| YOLO boxes showing dealership vehicles despite ROI | `manual_label.py` drew all boxes without applying ROI mask | Added ROI-aware box drawing: green=counted, red dashed=excluded |
| 5_ways_miranda x-axis ROI too aggressive | Cutoff at 38% excluded left-lane road vehicles | Replaced with corner exclusion zone (x<0.28, y<0.55) |
| Window-average labels mismatching visible frame content | Temporal averaging obscures per-frame reality | Switched to per-frame labeling |
| `city_road_newtown` high=0 | high_min=15 exceeds this camera's observed traffic volume | Awaiting more weekday peak data |

---

## 13. Key Findings & Analysis

### 13.1 Overlay leakage (drone dataset)
The largest finding from the drone pipeline. Red bounding boxes baked into source video constituted a spurious cue. Baseline CNN dropped most after removal, indicating it had learned to count red pixels rather than visual traffic patterns.

### 13.2 Per-frame labeling is more appropriate for CNNs
Window-level averaging (designed for time-series analysis) does not align with how a CNN classifier works. A CNN receives one image and outputs one prediction — the label should describe what is visible in that image, not the temporal average of a surrounding window.

### 13.3 Real-world camera data is harder to label than drone data
Drone data has SQLite trajectory annotations with precise vehicle counts and speeds. Live camera data requires YOLO detection as a proxy, which introduces its own errors (missed vehicles, false positives, viewpoint-dependent detection rates). Per-camera threshold calibration by visual inspection is necessary to produce clean labels.

### 13.4 EfficientNet-B0 generalises best to street-level cameras
On the drone dataset, MobileNetV2 led (78.7%). On the live camera dataset, EfficientNet-B0 leads (83.3%). This suggests EfficientNet's compound scaling and squeeze-excitation attention is better suited to the more variable, street-level visual conditions.

### 13.5 Class balance matters more than model architecture
The High class accuracy gap between weekend-only collection (18 high frames for 5_ways_miranda, precision 0.678) and weekday+weekend collection (756 high frames, precision 0.809) shows that data distribution is the primary driver of class-specific performance.

### 13.6 Ensemble only helps when constituent models are diverse and comparably strong
On the live dataset, including Baseline CNN and ResNet50 in the ensemble pulled accuracy below the EfficientNet-B0 single-model result. Ensemble is most effective when all constituent models are near the performance ceiling.

### 13.7 Window-level split prevents leakage in both pipelines
Both the drone pipeline (3 frames/window) and live pipeline (4 frames/window) maintain window integrity during train/val/test splitting. Near-identical frames from the same temporal window should not appear in both train and test.

---

## Appendix: Model Architectures

| Model | Type | Key layers | Params |
|---|---|---|---|
| Baseline CNN | Custom 3-block CNN | Conv→BN→ReLU×3, GAP, FC | 619K |
| MobileNetV2 | Depthwise-separable CNN | 19 bottleneck blocks, GAP | 2.2M |
| ResNet-50 | Residual CNN | 4 residual stages, GAP | 23.5M |
| EfficientNet-B0 | Compound-scaled CNN | MBConv blocks, SE attention | 4.0M |

All transfer models initialised with ImageNet pretrained weights; full backbone fine-tuned.

---

## Appendix: File Structure

```
src/
├── drone_pipeline/             # Drone dataset preprocessing pipeline
│   ├── discover_pairs.py       # Scans data/raw/ for valid (video + db) pairs
│   ├── generate_labels.py      # Congestion labels from SQLite per pair
│   ├── extract_frames.py       # JPEG frame extraction (3/window, 224×224)
│   ├── build_splits.py         # Window-stratified 70/15/15 split
│   ├── process_pairs.py        # Orchestrator: runs steps 2–4 for all 14 pairs
│   └── remove_overlays.py      # HSV masking + TELEA inpainting (run once)
├── live_pipeline/              # Live NSW camera pipeline
│   ├── collect.py              # TfNSW API poller (15s interval, 10 cameras)
│   ├── detect.py               # YOLOv8n vehicle detection + feature extraction
│   ├── label.py                # Per-frame congestion labeling (absolute thresholds)
│   ├── build_dataset.py        # Window-level split → train/val/test CSVs
│   ├── manual_label.py         # Visual review tool (ROI-aware YOLO overlay)
│   └── preview_predictions.py  # Model prediction preview with confidence + boxes
├── models/
│   ├── baseline_cnn.py
│   └── transfer_models.py
├── training/
│   └── train.py
├── evaluation/
│   └── evaluate.py             # Supports --split-dir for live/drone dataset switching
└── gui/
    └── app.py
```
