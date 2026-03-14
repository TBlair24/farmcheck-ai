import json
import uuid
import duckdb
import random
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH     = Path("pipeline/farmcheck.duckdb")
BRONZE_DIR  = Path("pipeline/bronze_events")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH))


def create_bronze_table():
    con = get_connection()
    con.execute("""
        CREATE TABLE IF NOT EXISTS bronze_predictions (
            event_id        VARCHAR PRIMARY KEY,
            received_at     TIMESTAMP,
            filename        VARCHAR,
            indicator       VARCHAR,
            confidence      DOUBLE,
            compliant       BOOLEAN,
            binary_score    INTEGER,
            domain          VARCHAR,
            model_version   VARCHAR,
            inference_ms    DOUBLE,
            raw_payload     JSON
        )
    """)
    con.close()
    print("✅ Bronze table ready")


def ingest_prediction(prediction: dict) -> str:
    """Ingest a single API prediction response into bronze layer"""
    event_id = str(uuid.uuid4())
    con      = get_connection()

    con.execute("""
        INSERT INTO bronze_predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        event_id,
        datetime.utcnow(),
        prediction["filename"],
        prediction["prediction"]["indicator"],
        prediction["prediction"]["confidence"],
        prediction["prediction"]["compliant"],
        prediction["prediction"]["binary_score"],
        prediction["prediction"]["domain"],
        prediction["model_version"],
        prediction["inference_ms"],
        json.dumps(prediction)
    ])

    con.close()
    return event_id


def simulate_field_data(n_records: int = 500):
    """
    Simulates field predictions from RTV program households.
    Mirrors how WorkMate would feed predictions into the warehouse.
    """
    indicators = [
        "crop_healthy", "bacterial_infection", "fungal_blight",
        "leaf_disease", "pest_infestation", "viral_infection"
    ]
    compliance = {
        "crop_healthy":        True,
        "bacterial_infection": False,
        "fungal_blight":       False,
        "leaf_disease":        False,
        "pest_infestation":    False,
        "viral_infection":     False,
    }

    # Simulate 50 households across 5 villages
    households = [f"HH_{i:03d}" for i in range(1, 51)]
    villages   = ["Rwengwe", "Kashongi", "Bubare", "Nyakashuri", "Kicwamba"]

    records = []
    base_date = datetime.utcnow() - timedelta(days=30)

    for _ in range(n_records):
        indicator  = random.choices(
            indicators,
            weights=[0.3, 0.15, 0.25, 0.1, 0.1, 0.1]  # realistic field distribution
        )[0]
        household  = random.choice(households)
        village    = random.choice(villages)
        confidence = random.uniform(0.75, 0.99)
        days_offset = random.randint(0, 30)
        timestamp  = base_date + timedelta(
            days=days_offset,
            hours=random.randint(7, 17)
        )

        payload = {
            "filename":      f"{household}_{indicator}_{uuid.uuid4().hex[:8]}.jpg",
            "prediction":    {
                "indicator":    indicator,
                "confidence":   round(confidence, 4),
                "compliant":    compliance[indicator],
                "binary_score": 1 if compliance[indicator] else 0,
                "domain":       "agriculture",
            },
            "model_version": "yolov8n-cls-v1",
            "inference_ms":  round(random.uniform(10, 50), 2),
            # Extra field context for warehouse
            "household_id":  household,
            "village":       village,
        }
        records.append(payload)

    return records


if __name__ == "__main__":
    create_bronze_table()

    print("\n🔄 Simulating 500 field predictions...")
    records = simulate_field_data(500)

    ingested = 0
    for record in records:
        ingest_prediction(record)
        ingested += 1

    # Quick summary
    con = get_connection()
    total = con.execute("SELECT COUNT(*) FROM bronze_predictions").fetchone()[0]
    print(f"✅ Bronze layer: {total} total records")

    summary = con.execute("""
        SELECT indicator, COUNT(*) as count,
               ROUND(AVG(confidence), 3) as avg_confidence
        FROM bronze_predictions
        GROUP BY indicator
        ORDER BY count DESC
    """).df()
    print("\n📊 Bronze Summary:")
    print(summary.to_string(index=False))
    con.close()