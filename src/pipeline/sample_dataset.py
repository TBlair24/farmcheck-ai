import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

INVENTORY = Path(__file__).parent / "../../data/processed/inventory.csv"
SAMPLE_DIR = Path(__file__).parent / "../../data/annotated/sample"
SAMPLE_PER_CLASS = 100  # 100 images × 15 classes = 1,500 total

def create_sample():
    df = pd.read_csv(INVENTORY)

    # inventory now uses ``class_label`` but older files may still have
    # ``class``.  normalize both cases to avoid future confusion.
    if "class_label" not in df.columns:
        if "class" in df.columns:
            df = df.rename(columns={"class": "class_label"})
        else:
            raise KeyError(
                "inventory.csv is missing both 'class_label' and 'class' columns: "
                f"found {df.columns.tolist()}"
            )

    # perform group-wise sampling; ``apply`` returns a subset with the
    # *original* indices.  ``class_label`` gets dropped from the intermediate
    # result, so we use the returned index to re-select from the original
    # dataframe and thereby retain the column.
    sampled_indices = (
        df.groupby("class_label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), SAMPLE_PER_CLASS), random_state=42))
        .index
    )
    sampled = df.loc[sampled_indices].reset_index(drop=True)

    # Copy images into flat sample folder.  iterate over the two columns we
    # actually care about to avoid problems if other columns end up in the
    # row index.
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for filepath, class_label in sampled[["filepath", "class_label"]].itertuples(index=False):
        src = Path(filepath)
        dst = SAMPLE_DIR / f"{class_label}__{src.name}"
        shutil.copy2(src, dst)

    # Save sample manifest
    manifest_path = Path("data/processed/sample_manifest.csv")

    # yolov8 (and many other training scripts) will look for a column named
    # ``class``.  a separate ``class_label`` column is kept internally so we
    # don't collide with the Python keyword, but writing a duplicate column
    # here makes the output immediately usable.
    out = sampled.copy()
    out["class"] = out["class_label"]
    out.to_csv(manifest_path, index=False)

    print(f"✅ Sampled {len(sampled)} images across {sampled['class_label'].nunique()} classes")
    print(f"📁 Saved to: {SAMPLE_DIR}")
    print(f"🗂 Manifest: {manifest_path} (includes both class_label and class columns)")
    return sampled

if __name__ == "__main__":
    create_sample()