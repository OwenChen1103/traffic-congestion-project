import pandas as pd
import shutil
from pathlib import Path

ROOT = Path(".")
df = pd.read_csv("data/live/splits/train.csv")

for label in ["low", "medium", "high"]:
    out_dir = Path("data/live/preview") / label
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = df[df["label"] == label].sample(5, random_state=42)
    for i, row in enumerate(sample.itertuples()):
        src = ROOT / row.image_path
        shutil.copy(src, out_dir / f"{i}_{src.name}")

print("Done")
