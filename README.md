# Traffic Congestion Classification from Intersection Footage

Visual traffic congestion classification from drone intersection footage, with a rule-based signal timing recommendation prototype. Three congestion levels are predicted — **Low**, **Medium**, and **High** — using CNN-based image classifiers trained on the Waterloo Multi-Agent Traffic Dataset.

---

## Results

| Model | Test Acc | Macro F1 | Low F1 | Med F1 | High F1 | Params |
|---|---|---|---|---|---|---|
| Baseline CNN | 0.7241 | 0.7210 | 0.618 | 0.707 | 0.837 | 619K |
| MobileNetV2 | 0.7874 | 0.7659 | 0.650 | 0.800 | 0.847 | 2.2M |
| ResNet-50 | 0.7615 | 0.7380 | 0.598 | 0.774 | 0.842 | 23.5M |
| EfficientNet-B0 | 0.7759 | 0.7503 | 0.580 | 0.793 | **0.878** | 4.0M |
| **Ensemble + TTA** | **0.8218** | **0.7992** | **0.667** | **0.839** | **0.893** | 4 models |

Trained on 14 intersection pairs — 2,296 samples across 777 five-second windows. Window-level stratified 70/15/15 split. Annotation overlays removed from all frames via HSV inpainting prior to training. Ensemble+TTA averages predictions from all 4 models across 5 augmented variants (original, h-flip, v-flip, ±5° rotation) at inference time — no retraining required.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/OwenChen1103/traffic-congestion-project.git
cd traffic-congestion-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the demo GUI

```bash
python src/gui/app.py
```

Opens at `http://localhost:7860`. Upload any intersection frame to get:
- Predicted congestion class + per-class confidence
- Rule-based signal timing recommendation
- GradCAM heatmap showing which regions drove the prediction

To use a different model:
```bash
python src/gui/app.py --model baseline_cnn
python src/gui/app.py --model mobilenet_v2
python src/gui/app.py --model efficientnet_b0
```

### 3. Evaluate models on the test set

```bash
python src/evaluation/evaluate.py --model resnet50
python src/evaluation/evaluate.py --model baseline_cnn
python src/evaluation/evaluate.py --model mobilenet_v2
python src/evaluation/evaluate.py --model efficientnet_b0
```

Outputs saved to `outputs/reports/` and `outputs/figures/`.

---

## Repository Structure

```
traffic-congestion-project/
│
├── src/
│   ├── config/
│   │   ├── config.yaml          # Central configuration (paths, hyperparameters)
│   │   └── settings.py          # Config loader and path helpers
│   │
│   ├── preprocessing/
│   │   ├── discover_pairs.py    # Scans data/raw/ for valid dataset pairs
│   │   ├── process_pairs.py     # Orchestrator — runs full pipeline for all pairs
│   │   ├── extract_frames.py    # Step 3: extract JPEG frames per window per pair
│   │   └── build_splits.py      # Step 4: window-stratified train/val/test split
│   │
│   ├── labeling/
│   │   └── generate_labels.py   # Step 2: congestion labels from SQLite annotations
│   │
│   ├── datasets/
│   │   └── congestion_dataset.py  # PyTorch Dataset class
│   │
│   ├── models/
│   │   ├── baseline_cnn.py      # Custom 3-block CNN baseline
│   │   └── transfer_models.py   # MobileNetV2, ResNet-50, EfficientNet-B0
│   │
│   ├── training/
│   │   └── train.py             # Training loop with cosine LR, class weighting
│   │
│   ├── evaluation/
│   │   └── evaluate.py          # Test set evaluation, confusion matrix, report
│   │
│   └── gui/
│       └── app.py               # Gradio demo with GradCAM
│
├── data/
│   ├── labels/                  # Generated label CSVs (committed)
│   │   ├── per_pair/            # Per-pair window labels and sample metadata
│   │   ├── window_labels_v1_all.csv
│   │   ├── samples_metadata_v1_all.csv
│   │   └── samples_split_v1_all.csv
│   │
│   └── processed/
│       ├── frames/              # Extracted JPEG frames (224×224, committed)
│       └── splits/              # train.csv / val.csv / test.csv
│
├── outputs/
│   ├── checkpoints/             # Trained model weights (committed)
│   ├── figures/                 # Confusion matrices
│   └── reports/                 # Classification reports
│
└── requirements.txt
```

---

## Reproducing the Pipeline

If you want to re-run the full pipeline from raw data (requires the original `.avi` and `.db` files):

```
data/raw/
  {pair_id}/
    intsc_data_{pair_id}.db
    {pair_id}.avi
```

Then run:

```bash
# Step 1: Inspect dataset structure (optional)
python src/preprocessing/inspect_dataset.py --pair 771

# Steps 2–4: Full pipeline for all pairs
python src/preprocessing/process_pairs.py

# Training
python src/training/train.py --model baseline_cnn
python src/training/train.py --model mobilenet_v2
python src/training/train.py --model resnet50
python src/training/train.py --model efficientnet_b0
```

---

## Labeling Methodology

Congestion labels are derived from trajectory metadata in the SQLite database, not from manual annotation. A composite score is computed per 5-second window:

```
score = 0.4 × norm_vehicle_count + 0.4 × norm_speed_inv + 0.2 × stop_proxy
```

Where `stop_proxy` is the fraction of trajectory rows with speed < 0.5 m/s. Normalisation is percentile-based (5th–95th) within each pair. Score thresholds: Low ≤ 0.33, High ≥ 0.67.

---

## Signal Timing Recommendation (Rule-Based Prototype)

| Predicted Class | Recommended Action | Green Phase Δ |
|---|---|---|
| Low | Maintain current cycle | 0 s |
| Medium | Extend green phase | +10 s |
| High | Extend green phase | +20 s |

This is a rule-based prototype only — not a closed-loop or optimisation-based traffic control system.

---

## Requirements

- Python 3.9+
- PyTorch 2.x (MPS supported for Apple Silicon)
- See `requirements.txt` for full dependency list

---

## Dataset

[Waterloo Multi-Agent Traffic Dataset](https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/datasets) — 14 intersection recording pairs, drone footage at ~29.97 FPS with SQLite trajectory annotations.
