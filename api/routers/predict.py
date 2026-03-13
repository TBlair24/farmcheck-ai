import time
import io
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from ultralytics import YOLO
from api.models.schemas import PredictionResponse, PredictionResult

router = APIRouter()

# ── Load model once at startup ────────────────────────────────────────────
def load_model() -> YOLO:
    candidates = []
    for d in [Path("runs"), Path("models/weights")]:
        if d.exists():
            candidates.extend(d.rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError("❌ No model found — run training first.")
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"✅ Model loaded → {best}")
    return YOLO(best, task="classify")

MODEL = load_model()

# ── Compliance metadata ───────────────────────────────────────────────────
COMPLIANCE_MAP = {
    "crop_healthy":        {"compliant": True,  "domain": "agriculture"},
    "bacterial_infection": {"compliant": False, "domain": "agriculture"},
    "fungal_blight":       {"compliant": False, "domain": "agriculture"},
    "leaf_disease":        {"compliant": False, "domain": "agriculture"},
    "pest_infestation":    {"compliant": False, "domain": "agriculture"},
    "viral_infection":     {"compliant": False, "domain": "agriculture"},
}


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # ── Validate file type ────────────────────────────────────────────────
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG/PNG accepted."
        )

    # ── Read and convert image ────────────────────────────────────────────
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")

    # ── Run inference ─────────────────────────────────────────────────────
    start   = time.perf_counter()
    results = MODEL.predict(source=image, verbose=False)
    end     = time.perf_counter()

    inference_ms = round((end - start) * 1000, 2)
    result       = results[0]

    # ── Parse outputs ─────────────────────────────────────────────────────
    top_idx    = result.probs.top1
    indicator  = result.names[top_idx]
    confidence = float(result.probs.top1conf)

    # All class scores as a dict
    all_scores = {
        result.names[i]: round(float(result.probs.data[i]), 4)
        for i in range(len(result.names))
    }

    # Compliance metadata
    meta       = COMPLIANCE_MAP.get(indicator, {"compliant": False, "domain": "unknown"})
    compliant  = meta["compliant"]

    return PredictionResponse(
        status       = "success",
        filename     = file.filename,
        inference_ms = inference_ms,
        prediction   = PredictionResult(
            indicator    = indicator,
            confidence   = round(confidence, 4),
            compliant    = compliant,
            binary_score = 1 if compliant else 0,
            domain       = meta["domain"],
        ),
        all_scores = all_scores,
    )