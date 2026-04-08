# Traffic Congestion Classification from Intersection Footage

Visual traffic congestion classification from intersection footage, with a rule-based signal timing recommendation prototype. Three congestion levels are predicted — **Low**, **Medium**, and **High** — using CNN-based image classifiers trained on two datasets: the Waterloo Multi-Agent Traffic Dataset (drone footage) and live NSW traffic camera streams.

---

## Results

### Live NSW Camera Dataset (Primary)

10 cameras across Sydney and Wollongong. Street-level fixed cameras. Labels derived from YOLOv8n vehicle detection with per-camera calibrated thresholds. Train/test split by camera region (Sydney → trained, Wollongong → tested).

| Model | Test Acc | Macro F1 | Low F1 | Med F1 | High F1 |
|---|---|---|---|---|---|
| Baseline CNN | 0.7201 | 0.7246 | 0.822 | 0.636 | 0.716 |
| MobileNetV2 | 0.8143 | 0.8204 | 0.880 | 0.764 | 0.818 |
| ResNet-50 | 0.7686 | 0.7778 | 0.845 | 0.715 | 0.773 |
| EfficientNet-B0 | 0.8290 | 0.8340 | 0.899 | 0.768 | 0.835 |
| Ensemble (4 models) | 0.8249 | 0.8308 | 0.891 | 0.776 | 0.825 |
| **EfficientNet-B0 + TTA** ⚡ | **0.8331** | **0.8375** | 0.898 | 0.774 | **0.841** |

**Best model: EfficientNet-B0 + TTA — 83.31% test accuracy**

19,919 frames collected across 3 sessions (weekend + weekday morning/midday/afternoon peak). 15,881 retained after brightness and nighttime filters. Window-level stratified split (4 frames/window kept together).

---

### Drone Dataset (Waterloo, Reference)

| Model | Test Acc | Macro F1 | Low F1 | Med F1 | High F1 | Params |
|---|---|---|---|---|---|---|
| Baseline CNN | 0.7241 | 0.7210 | 0.618 | 0.707 | 0.837 | 619K |
| **MobileNetV2** ★ | **0.7874** | **0.7659** | **0.650** | **0.800** | 0.847 | 2.2M |
| ResNet-50 | 0.7615 | 0.7380 | 0.598 | 0.774 | 0.842 | 23.5M |
| EfficientNet-B0 | 0.7759 | 0.7503 | 0.580 | 0.793 | **0.878** | 4.0M |
| **Ensemble + TTA** | **0.8218** | **0.7992** | **0.667** | **0.839** | **0.893** | 4 models |

2,296 samples across 777 five-second windows. Annotation overlays removed via HSV inpainting prior to training.

---

## Quickstart

### Install

```bash
git clone https://github.com/OwenChen1103/traffic-congestion-project.git
cd traffic-congestion-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the demo GUI

```bash
python src/gui/app.py
```

Opens at `http://localhost:7860`. Upload any intersection frame to get:
- Predicted congestion class + per-class confidence
- Rule-based signal timing recommendation
- GradCAM heatmap

### Evaluate models (live dataset)

```bash
python src/evaluation/evaluate.py --model efficientnet_b0 --split-dir data/live/splits
python src/evaluation/evaluate.py --model mobilenet_v2 --split-dir data/live/splits
python src/evaluation/evaluate.py --ensemble --split-dir data/live/splits
python src/evaluation/evaluate.py --model efficientnet_b0 --tta --split-dir data/live/splits
```

### Preview model predictions

```bash
python src/live_pipeline/preview_predictions.py --model efficientnet_b0
```

Shows sample LOW / MEDIUM / HIGH predictions with YOLO bounding boxes, vehicle counts, confidence scores, and ground truth labels. Press `n` for a new random batch.

---

## Live Pipeline

### Collect new camera data

```bash
python src/live_pipeline/collect.py --duration 90   # collect for 90 minutes
```

Polls 10 TfNSW cameras every 15 seconds. Automatically resumes from the last collected window. Requires `TFNSW_API_KEY` in `.env`.

### Run the full pipeline after collection

```bash
python src/live_pipeline/detect.py          # YOLOv8n vehicle detection on all frames
python src/live_pipeline/label.py           # Per-frame congestion labeling
python src/live_pipeline/build_dataset.py   # Train/val/test split CSVs
python src/training/train.py --model efficientnet_b0 --split-dir data/live/splits
```

### Visual label review

```bash
python src/live_pipeline/manual_label.py
```

Review tool showing 9 images per camera (3 per class). YOLO boxes shown in real time — green = counted towards vehicle count, red dashed = excluded by camera ROI.

---

## Repository Structure

```
traffic-congestion-project/
│
├── src/
│   ├── config/
│   │   ├── config.yaml              # Central configuration
│   │   └── settings.py
│   │
│   ├── drone_pipeline/              # Waterloo drone dataset pipeline
│   │   ├── discover_pairs.py
│   │   ├── process_pairs.py
│   │   ├── extract_frames.py
│   │   ├── build_splits.py
│   │   ├── generate_labels.py
│   │   ├── inspect_dataset.py
│   │   └── remove_overlays.py       # HSV inpainting to remove annotation overlays
│   │
│   ├── live_pipeline/               # NSW live camera pipeline
│   │   ├── collect.py               # TfNSW API data collection
│   │   ├── detect.py                # YOLOv8n vehicle detection
│   │   ├── label.py                 # Per-frame congestion labeling
│   │   ├── build_dataset.py         # Train/val/test split builder
│   │   ├── manual_label.py          # Visual label review tool
│   │   └── preview_predictions.py   # Model prediction visualiser
│   │
│   ├── datasets/
│   │   └── congestion_dataset.py    # PyTorch Dataset class
│   │
│   ├── models/
│   │   ├── baseline_cnn.py
│   │   └── transfer_models.py
│   │
│   ├── training/
│   │   └── train.py                 # Training loop, cosine LR, class weighting
│   │
│   ├── evaluation/
│   │   └── evaluate.py              # Test eval, ensemble, TTA, --split-dir support
│   │
│   └── gui/
│       └── app.py                   # Gradio demo with GradCAM
│
├── data/
│   ├── live/
│   │   └── splits/                  # train/val/test CSVs for live dataset
│   └── processed/
│       └── splits/                  # train/val/test CSVs for drone dataset
│
├── outputs/
│   ├── checkpoints/                 # Trained model weights
│   ├── figures/                     # Confusion matrices
│   └── reports/                     # Classification reports
│
└── requirements.txt
```

---

## Labeling

### Live dataset (per-frame, absolute thresholds)
Each frame is labeled individually based on the number of YOLO-detected vehicles. Per-camera thresholds calibrated by visual inspection:

| Camera | low_max | high_min |
|---|---|---|
| james_ruse_drive_rosehill | 8 | 20 |
| hume_highway_bankstown | 6 | 15 |
| 5_ways_miranda | 7 | 15 |
| parramatta_road_camperdown | 7 | 17 |
| king_georges_road_hurstville | 3 | 10 |
| city_road_newtown | 5 | 15 |
| anzac_parade_moore_park | 5 | 15 |
| memorial_drive_towradgi | 5 | 13 |
| shellharbour_road_warilla | 7 | 15 |

### Drone dataset (composite score)
```
score = 0.4 × norm_vehicle_count + 0.4 × norm_speed_inv + 0.2 × stop_proxy
```
Score thresholds: Low ≤ 0.33, High ≥ 0.67. Derived from SQLite trajectory annotations.

---

## Signal Timing Recommendation (Rule-Based Prototype)

| Predicted Class | Recommended Action | Green Phase Δ |
|---|---|---|
| Low | Maintain current cycle | 0 s |
| Medium | Extend green phase | +10 s |
| High | Extend green phase | +20 s |

---

## Requirements

- Python 3.9+
- PyTorch 2.x (MPS supported for Apple Silicon)
- See `requirements.txt` for full dependency list

---

## Datasets

- **Drone:** [Waterloo Multi-Agent Traffic Dataset](https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/datasets) — 14 intersection pairs, top-down drone footage with SQLite trajectory annotations
- **Live:** NSW TfNSW Open Data API — 10 fixed traffic cameras across Sydney and Wollongong, collected April 2026
