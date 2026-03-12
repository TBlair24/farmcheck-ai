import wandb
import yaml
from pathlib import Path
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────────────
DATASET_YAML = Path("data/processed/dataset")
MODEL_DIR    = Path("models/weights")
PROJECT_NAME = "farmcheck-ai"
RUN_NAME     = "yolov8-agriculture-v1"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Class weights (handles imbalance) ───────────────────────────────────
# Order must match class index in dataset.yaml
# bacterial_infection, crop_healthy, fungal_blight,
# leaf_disease, pest_infestation, viral_infection
CLASS_WEIGHTS = [1.0, 0.5, 0.8, 1.0, 2.0, 1.0]

# ── W&B Init ─────────────────────────────────────────────────────────────
wandb.init(
    project=PROJECT_NAME,
    name=RUN_NAME,
    config={
        "model":       "yolov8n-cls",   # nano classification model
        "epochs":      30,
        "img_size":    224,
        "batch_size":  32,
        "dataset":     str(DATASET_YAML),
        "class_weights": CLASS_WEIGHTS,
        "optimizer":   "Adam",
        "notes":       "Baseline run - PlantVillage agriculture compliance"
    }
)

cfg = wandb.config

# ── Load Model ────────────────────────────────────────────────────────────
print(f"\n🚀 Loading {cfg.model}...")
model = YOLO(cfg.model)

# ── Train ─────────────────────────────────────────────────────────────────
print(f"\n🏋️  Starting training — {cfg.epochs} epochs")
results = model.train(
    data    = str(DATASET_YAML.resolve()),
    epochs  = cfg.epochs,
    imgsz   = cfg.img_size,
    batch   = cfg.batch_size,
    name    = RUN_NAME,
    project = str(MODEL_DIR),
    optimizer = cfg.optimizer,
    patience  = 10,         # early stopping
    augment   = True,       # handles class imbalance
    degrees   = 15,         # rotation augmentation
    flipud    = 0.3,        # vertical flip
    fliplr    = 0.5,        # horizontal flip
    hsv_h     = 0.015,      # colour jitter
    hsv_s     = 0.4,
    hsv_v     = 0.4,
)

# ── Validate ──────────────────────────────────────────────────────────────
print("\n📊 Running validation...")
metrics = model.val()

# ── Log to W&B ────────────────────────────────────────────────────────────
wandb.log({
    "val/top1_accuracy": metrics.top1,
    "val/top5_accuracy": metrics.top5,
})

# ── Save Best Model ───────────────────────────────────────────────────────
best_model_path = Path(MODEL_DIR) / RUN_NAME / "weights" / "best.pt"
if best_model_path.exists():
    wandb.save(str(best_model_path))
    print(f"\n✅ Best model saved → {best_model_path}")

wandb.finish()
print("\n🎉 Training complete! Check your results at wandb.ai")