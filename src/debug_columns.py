import pandas as pd
import os

RAW_PATH = 'data/raw/'

print("🔍 INSPECTING CSV HEADERS...\n")

for filename in os.listdir(RAW_PATH):
    if filename.endswith(".csv"):
        print(f"📂 File: {filename}")
        try:
            df = pd.read_csv(os.path.join(RAW_PATH, filename), nrows=2) # Baca 2 baris je
            print(f"   📋 Columns: {list(df.columns)}")
        except Exception as e:
            print(f"   ❌ Error reading: {e}")
        print("-" * 30)