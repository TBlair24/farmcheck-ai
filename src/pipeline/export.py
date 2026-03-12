import shutil
import wandb
from pathlib import Path
from ultralytics import YOLO


def find_best_model() -> Path:
    candidates = []
    for d in [Path("runs"), Path("models/weights")]:
        if d.exists():
            candidates.extend(d.rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError("❌ Could not find best.pt")
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"✅ Found model → {best}")
    return best


EXPORT_DIR = Path("models/exported")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = find_best_model()

wandb.init(
    project="farmcheck-ai",
    name="model-export-v1",
    config={"source_model": str(MODEL_PATH)}
)

model = YOLO(MODEL_PATH)

# ── Export to ONNX ────────────────────────────────────────────────────────
print("\n🔄 Exporting to ONNX...")
onnx_path = model.export(
    format   = "onnx",
    imgsz    = 224,
    dynamic  = False,
    simplify = True,
)
onnx_dest = EXPORT_DIR / "farmcheck_v1.onnx"
shutil.copy2(onnx_path, onnx_dest)

# ── Size comparison ───────────────────────────────────────────────────────
def mb(path: Path) -> float:
    return round(path.stat().st_size / 1024 / 1024, 2)

print(f"\n📊 Model Size Comparison:")
print(f"  {'Original (.pt)':<20}: {mb(MODEL_PATH)} MB")
print(f"  {'ONNX':<20}: {mb(onnx_dest)} MB")

wandb.log({
    "export/original_mb": mb(MODEL_PATH),
    "export/onnx_mb":     mb(onnx_dest),
})

artifact = wandb.Artifact("farmcheck-models", type="model")
artifact.add_file(str(onnx_dest))
wandb.log_artifact(artifact)

wandb.finish()
print(f"\n✅ Export complete → {onnx_dest}")