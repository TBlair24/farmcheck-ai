import sys
from pathlib import Path

# Add pipeline directory to path
sys.path.insert(0, str(Path(__file__).parent))

from layers.bronze import create_bronze_table, simulate_field_data, ingest_prediction
from layers.silver import create_silver_table, transform_bronze_to_silver
from layers.gold   import create_gold_tables, build_household_adoption, build_village_summary

print("=" * 55)
print("🚀 FarmCheck AI — Medallion Pipeline")
print("=" * 55)

print("\n🥉 BRONZE — Initialising & ingesting...")
create_bronze_table()
records = simulate_field_data(500)
for r in records:
    ingest_prediction(r)

print("\n🥈 SILVER — Transforming & enriching...")
create_silver_table()
transform_bronze_to_silver()

print("\n🥇 GOLD — Aggregating adoption scores...")
create_gold_tables()
build_household_adoption()
build_village_summary()

print("\n✅ Pipeline complete!")