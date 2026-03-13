import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from api.main import app

client = TestClient(app)


def make_test_image() -> bytes:
    """Creates a simple green test image"""
    img    = Image.new("RGB", (224, 224), color=(34, 139, 34))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert len(data["classes"]) == 6


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "FarmCheck AI" in response.json()["service"]


def test_predict_valid_image():
    img_bytes = make_test_image()
    response  = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data
    assert data["prediction"]["indicator"] in [
        "crop_healthy", "bacterial_infection", "fungal_blight",
        "leaf_disease", "pest_infestation", "viral_infection"
    ]
    assert data["inference_ms"] > 0
    assert len(data["all_scores"]) == 6


def test_predict_invalid_file_type():
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400