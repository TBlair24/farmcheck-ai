from pathlib import Path

RAW_DIR = Path("data/raw/plantvillage/PlantVillage")

# Indicator rules based on keywords in class name
def infer_indicator(class_name: str) -> str:
    name = class_name.lower()
    if "healthy" in name:
        return "crop_healthy"
    elif "bacterial" in name:
        return "bacterial_infection"
    elif "blight" in name:
        return "fungal_blight"
    elif "mold" in name or "mold" in name:
        return "fungal_blight"
    elif "septoria" in name or "target_spot" in name or "leaf_spot" in name:
        return "leaf_disease"
    elif "virus" in name or "mosaic" in name or "curl" in name:
        return "viral_infection"
    elif "spider" in name or "mite" in name:
        return "pest_infestation"
    else:
        return "unclassified"

def build_label_map(raw_dir: Path) -> dict:
    """Builds label map dynamically from actual folder names"""
    label_map = {}
    for folder in sorted(raw_dir.iterdir()):
        if folder.is_dir():
            indicator = infer_indicator(folder.name)
            compliant = "healthy" in folder.name.lower()
            label_map[folder.name] = {
                "domain":    "agriculture",
                "indicator": indicator,
                "compliant": compliant
            }
    return label_map

# Build it once on import
LABEL_MAP = build_label_map(RAW_DIR)

def get_compliance_label(class_name: str) -> dict:
    return LABEL_MAP.get(class_name, {
        "domain": "unknown",
        "indicator": "unclassified",
        "compliant": None
    })

def get_binary_label(class_name: str) -> int:
    info = get_compliance_label(class_name)
    return 1 if info["compliant"] else 0

if __name__ == "__main__":
    print(f"📦 Auto-detected {len(LABEL_MAP)} classes from folders:\n")
    for cls, info in LABEL_MAP.items():
        status = "✅" if info["compliant"] else "❌"
        print(f"  {status} {cls:60} → {info['indicator']}")