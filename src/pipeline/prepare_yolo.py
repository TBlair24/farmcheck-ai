import shutil
import pandas as pd
from pathlib import Path

MANIFEST  = Path("data/processed/dataset_manifest.csv")
YOLO_DIR  = Path("data/processed/yolo_dataset")

def prepare_yolo():
    df = pd.read_csv(MANIFEST)

    # Get unique indicators and create class index
    classes   = sorted(df["indicator"].unique())
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    print(f"📦 Classes: {class_map}\n")

    # Create YOLO folder structure
    for split in ["train", "val", "test"]:
        (YOLO_DIR / split / "images").mkdir(parents=True, exist_ok=True)

    # Copy images into YOLO structure
    for _, row in df.iterrows():
        src = (
            Path("data/processed/dataset")
            / row["split"]
            / row["indicator"]
            / row["filename"]
        )
        dst = YOLO_DIR / row["split"] / "images" / row["filename"]

        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  ⚠️  Missing: {src}")

    # Write dataset.yaml
    yaml_content = f"""
path: {YOLO_DIR.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: {len(classes)}
names: {classes}

# RTV Compliance Indicators
# 1 = compliant (healthy), 0 = non-compliant (diseased)
"""
    (YOLO_DIR / "dataset.yaml").write_text(yaml_content.strip())
    print(f"✅ dataset.yaml written")

    # Save class map
    pd.DataFrame(
        list(class_map.items()), columns=["indicator", "class_id"]
    ).to_csv(YOLO_DIR / "class_map.csv", index=False)

    # Summary
    print(f"\n📊 YOLO Dataset Summary:")
    print(df.groupby(["split", "indicator"])["filename"].count().to_string())
    print(f"\n✅ YOLO dataset ready at: {YOLO_DIR}")

if __name__ == "__main__":
    prepare_yolo()