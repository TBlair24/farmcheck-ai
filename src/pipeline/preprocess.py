import json
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image
from label_map import get_compliance_label, get_binary_label
from sklearn.model_selection import train_test_split

ANNOTATIONS   = Path("data/annotated/annotations_final.json")
SAMPLE_DIR    = Path("data/annotated/sample")
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("data/processed/dataset")

IMG_SIZE = (224, 224)  # Standard input size for classification models

SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}


def load_annotations(path: Path) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)

    records = []
    for task in raw:
        meta       = task.get("meta", {})
        class_name = meta.get("class", "unknown")
        annotation = task["annotations"][0]["result"]

        if annotation:
            label = annotation[0]["value"]["choices"][0]
        else:
            label = "unclassified"

        image_filename = task["data"]["image"].split("d=")[-1]
        # label‑studio sometimes returns a relative path (e.g. "../../data/annotated/..."),
        # which would be prepended again below and lead to duplicated/misspelled paths.
        # keep only the basename so that we can join it with SAMPLE_DIR safely.
        image_filename = Path(image_filename).name

        records.append({
            "filename":  image_filename,
            "class":     class_name,
            "indicator": label,
            "compliant": get_compliance_label(class_name).get("compliant"),
            "binary":    get_binary_label(class_name),
            "domain":    get_compliance_label(class_name).get("domain", "unknown"),
        })

    return pd.DataFrame(records)


def resize_and_save(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        img.save(dst, "JPEG", quality=90)
    except Exception as e:
        print(f"  ⚠️  Skipping {src.name}: {e}")


def split_and_export(df: pd.DataFrame):
    train_val, test = train_test_split(
        df, test_size=SPLITS["test"], stratify=df["indicator"], random_state=42
    )
    train, val = train_test_split(
        train_val,
        test_size=SPLITS["val"] / (SPLITS["train"] + SPLITS["val"]),
        stratify=train_val["indicator"],
        random_state=42
    )

    split_map = {"train": train, "val": val, "test": test}
    summary   = []

    for split_name, split_df in split_map.items():
        print(f"\n📂 {split_name.upper()} — {len(split_df)} images")
        for _, row in split_df.iterrows():
            src = SAMPLE_DIR / row["filename"]
            dst = OUTPUT_DIR / split_name / row["indicator"] / row["filename"]
            resize_and_save(src, dst)

        split_df["split"] = split_name
        summary.append(split_df)

    final = pd.concat(summary)
    final.to_csv(PROCESSED_DIR / "dataset_manifest.csv", index=False)
    return final


def generate_docs(df: pd.DataFrame):
    """Generate a simple annotation doc summarising the dataset"""
    doc_path = PROCESSED_DIR / "annotation_report.md"
    lines = [
        "# FarmCheck AI — Dataset & Annotation Report\n",
        "## Overview",
        f"- **Total images:** {len(df)}",
        f"- **Classes:** {df['class'].nunique()}",
        f"- **Indicators:** {df['indicator'].nunique()}",
        f"- **Compliant (healthy):** {df['compliant'].sum()}",
        f"- **Non-compliant (diseased):** {(~df['compliant'].astype(bool)).sum()}",
        f"- **Image size (preprocessed):** {IMG_SIZE[0]}×{IMG_SIZE[1]}px\n",
        "## Split Distribution",
        df.groupby("split")["filename"].count().to_markdown(),
        "\n## Indicator Distribution",
        df.groupby("indicator")["filename"].count().to_markdown(),
        "\n## Class → Indicator Mapping",
        df[["class", "indicator", "compliant", "binary"]].drop_duplicates()
          .sort_values("class").to_markdown(index=False),
        "\n## Notes",
        "- Stratified sampling used to ensure class balance across splits.",
        "- All images resized to 224×224 and converted to RGB.",
        "- Labels auto-generated from folder names and verified in Label Studio.",
        "- Binary label: 1 = compliant (healthy), 0 = non-compliant (diseased).",
    ]
    # write with utf-8 encoding to avoid Windows cp1252 errors
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 Annotation report saved → {doc_path}")


if __name__ == "__main__":
    print("🔄 Loading annotations...")
    df = load_annotations(ANNOTATIONS)
    print(f"✅ Loaded {len(df)} records")

    print("\n🔄 Splitting and preprocessing images...")
    final_df = split_and_export(df)

    print("\n🔄 Generating annotation docs...")
    generate_docs(final_df)

    print("\n✅ Preprocessing complete!")
    print(final_df.groupby(["split", "indicator"])["filename"].count().to_string())