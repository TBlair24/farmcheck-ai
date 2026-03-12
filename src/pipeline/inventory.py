import os
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/plantvillage/PlantVillage")

def build_inventory(root_dir: Path) -> pd.DataFrame:
    records = []
    for class_dir in root_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
            for img in images:
                records.append({
                    "filepath": str(img),
                    "class_label": class_dir.name,
                    "filename": img.name
                })
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = build_inventory(RAW_DIR)
    df.to_csv("data/processed/inventory.csv", index=False)
    print(f"✅ Found {len(df)} images across {df['class_label'].nunique()} classes")
    print(df['class_label'].value_counts().head(10))