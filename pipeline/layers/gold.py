import duckdb
from pathlib import Path

DB_PATH = Path("pipeline/farmcheck.duckdb")


def get_connection():
    return duckdb.connect(str(DB_PATH))


def create_gold_tables():
    con = get_connection()

    con.execute("DROP TABLE IF EXISTS gold_household_adoption")
    con.execute("DROP TABLE IF EXISTS gold_village_summary")

    con.execute("""
        CREATE TABLE IF NOT EXISTS gold_household_adoption (
            household_id        VARCHAR PRIMARY KEY,
            village             VARCHAR,
            total_assessments   INTEGER,
            compliant_count     INTEGER,
            adoption_score      DOUBLE,
            dominant_indicator  VARCHAR,
            last_assessed_at    TIMESTAMP,
            trend               VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS gold_village_summary (
            village             VARCHAR PRIMARY KEY,
            total_households    INTEGER,
            avg_adoption_score  DOUBLE,
            fully_compliant     INTEGER,
            at_risk             INTEGER,
            top_issue           VARCHAR,
            week_number         INTEGER,
            year                INTEGER
        )
    """)

    con.close()
    print("✅ Gold tables ready")


def build_household_adoption():
    con = get_connection()

    con.execute("DELETE FROM gold_household_adoption")
    con.execute("""
        INSERT INTO gold_household_adoption
        SELECT
            household_id,
            MODE(village)                           AS village,
            COUNT(*)                                AS total_assessments,
            SUM(binary_score)                       AS compliant_count,
            ROUND(AVG(binary_score) * 100, 1)       AS adoption_score,
            MODE(CASE WHEN compliant = false
                 THEN indicator END)                AS dominant_indicator,
            MAX(received_at)                        AS last_assessed_at,
            CASE
                WHEN AVG(CASE WHEN row_num > total_rows / 2
                         THEN binary_score ELSE NULL END)
                   > AVG(CASE WHEN row_num <= total_rows / 2
                         THEN binary_score ELSE NULL END)
                THEN 'improving'
                WHEN AVG(CASE WHEN row_num > total_rows / 2
                         THEN binary_score ELSE NULL END)
                   < AVG(CASE WHEN row_num <= total_rows / 2
                         THEN binary_score ELSE NULL END)
                THEN 'declining'
                ELSE 'stable'
            END AS trend
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY household_id ORDER BY received_at
                ) AS row_num,
                COUNT(*) OVER (
                    PARTITION BY household_id
                ) AS total_rows
            FROM silver_predictions
            WHERE is_valid = true
        ) ranked
        GROUP BY household_id
    """)

    count = con.execute(
        "SELECT COUNT(*) FROM gold_household_adoption"
    ).fetchone()[0]
    print(f"✅ Gold household adoption: {count} households scored")

    sample = con.execute("""
        SELECT household_id, village, total_assessments,
               adoption_score, dominant_indicator, trend
        FROM gold_household_adoption
        ORDER BY adoption_score DESC
        LIMIT 10
    """).df()
    print("\n📊 Top 10 Households by Adoption Score:")
    print(sample.to_string(index=False))
    con.close()


def build_village_summary():
    con = get_connection()

    con.execute("DELETE FROM gold_village_summary")
    con.execute("""
        INSERT INTO gold_village_summary
        SELECT
            village,
            COUNT(DISTINCT household_id)            AS total_households,
            ROUND(AVG(adoption_score), 1)           AS avg_adoption_score,
            SUM(CASE WHEN adoption_score = 100
                THEN 1 ELSE 0 END)                  AS fully_compliant,
            SUM(CASE WHEN adoption_score < 50
                THEN 1 ELSE 0 END)                  AS at_risk,
            MODE(dominant_indicator)                AS top_issue,
            WEEK(MAX(last_assessed_at))             AS week_number,
            YEAR(MAX(last_assessed_at))             AS year
        FROM gold_household_adoption
        GROUP BY village
    """)

    results = con.execute("""
        SELECT village, total_households, avg_adoption_score,
               fully_compliant, at_risk, top_issue
        FROM gold_village_summary
        ORDER BY avg_adoption_score DESC
    """).df()

    print("\n📊 Gold Village Summary:")
    print(results.to_string(index=False))
    con.close()


if __name__ == "__main__":
    create_gold_tables()
    build_household_adoption()
    build_village_summary()