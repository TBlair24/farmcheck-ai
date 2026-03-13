from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import predict
from api.models.schemas import HealthResponse

app = FastAPI(
    title       = "FarmCheck AI",
    description = "Crop compliance classification API for RTV field evaluation",
    version     = "1.0.0",
)

# ── CORS — allows WorkMate or any field app to call this API ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])


@app.get("/health", response_model=HealthResponse)
def health():
    from api.routers.predict import MODEL
    return HealthResponse(
        status        = "healthy",
        model_loaded  = MODEL is not None,
        model_version = "yolov8n-cls-v1",
        classes       = list(MODEL.names.values()) if MODEL else [],
    )


@app.get("/")
def root():
    return {
        "service": "FarmCheck AI",
        "docs":    "/docs",
        "health":  "/health",
    }