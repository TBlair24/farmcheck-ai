import json
import pandas as pd
from pathlib import Path
from label_map import get_compliance_label

MANIFEST = Path("data/processed/sample_manifest.csv")
SAMPLE_DIR = Path("data/annotated/sample")
OUTPUT = Path("data/processed/prelabels.json")

def build_prelabel_json(manifest_path: Path) -> list:
    df = pd.read_csv(manifest_path)
    tasks = []

    for _, row in df.iterrows():
        class_name = row["class"]
        filename   = Path(row["filepath"]).name
        image_file = f"{class_name}__{filename}"
        info       = get_compliance_label(class_name)

        task = {
            "data": {
                "image": f"/data/local-files/?d=annotated/sample/{image_file}"
            },
            "annotations": [
                {
                    "result": [
                        {
                            "type": "choices",
                            "value": {
                                "choices": [info["indicator"]]
                            },
                            "from_name": "choice",
                            "to_name":   "image",
                        }
                    ],
                    "ground_truth": True
                }
            ],
            "meta": {
                "class":     class_name,
                "domain":    info["domain"],
                "compliant": info["compliant"]
            }
        }
        tasks.append(task)

    return tasks

if __name__ == "__main__":
    tasks = build_prelabel_json(MANIFEST)
    OUTPUT.write_text(json.dumps(tasks, indent=2))
    print(f"✅ Generated {len(tasks)} pre-labeled tasks → {OUTPUT}")

    # Quick summary
    df = pd.read_csv(MANIFEST)
    compliant_count     = sum(1 for t in tasks if t["meta"]["compliant"])
    non_compliant_count = sum(1 for t in tasks if not t["meta"]["compliant"])
    print(f"   ✅ Compliant:     {compliant_count}")
    print(f"   ❌ Non-compliant: {non_compliant_count}")