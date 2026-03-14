import duckdb
from pathlib import Path

DB_PATH = Path("pipeline/farmcheck.duckdb")


def get_connection():
    return duckdb.connect(str(DB_PATH))


def create_silver_table():
    con = get_connection()
    con.execute("""
        CREATE TABLE IF NOT EXISTS silver_predictions (
            event_id            VARCHAR PRIMARY KEY,
            received_at         TIMESTAMP,
            household_id        VARCHAR,
            village             VARCHAR,
            indicator           VARCHAR,
            confidence          DOUBLE,
            compliant           BOOLEAN,
            binary_score        INTEGER,
            domain              VARCHAR,
            model_version       VARCHAR,
            inference_ms        DOUBLE,
            -- Enriched fields
            confidence_tier     VARCHAR,   -- high / medium / low
            week_number         INTEGER,
            month               INTEGER,
            year                INTEGER,
            is_valid            BOOLEAN    -- confidence > 0.75
        )
    """)
    con.close()
    print("✅ Silver table ready")


def transform_bronze_to_silver():
    con = get_connection()

    # Extract household_id and village from raw JSON payload
    con.execute("""
        INSERT OR REPLACE INTO silver_predictions
        SELECT
            event_id,
            received_at,
            -- Extract from JSON payload
            json_extract_string(raw_payload, '$.household_id') AS household_id,
            json_extract_string(raw_payload, '$.village')      AS village,
            indicator,
            confidence,
            compliant,
            binary_score,
            domain,
            model_version,
            inference_ms,
            -- Confidence tier
            CASE
                WHEN confidence >= 0.90 THEN 'high'
                WHEN confidence >= 0.75 THEN 'medium'
                ELSE 'low'
            END AS confidence_tier,
            -- Time dimensions
            WEEK(received_at)   AS week_number,
            MONTH(received_at)  AS month,
            YEAR(received_at)   AS year,
            -- Validity flag
            confidence >= 0.75  AS is_valid
        FROM bronze_predictions
        WHERE event_id NOT IN (SELECT event_id FROM silver_predictions)
    """)

    count = con.execute("SELECT COUNT(*) FROM silver_predictions").fetchone()[0]
    print(f"✅ Silver layer: {count} records transformed")

    summary = con.execute("""
        SELECT
            confidence_tier,
            COUNT(*)                            AS count,
            ROUND(AVG(binary_score) * 100, 1)  AS compliance_pct
        FROM silver_predictions
        GROUP BY confidence_tier
        ORDER BY confidence_tier
    """).df()
    print("\n📊 Silver Summary:")
    print(summary.to_string(index=False))
    con.close()


if __name__ == "__main__":
    create_silver_table()
    transform_bronze_to_silver()