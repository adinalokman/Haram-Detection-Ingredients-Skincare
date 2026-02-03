import pandas as pd
import joblib
import os
import numpy as np


# CONFIGURATION
INPUT_DATA = 'data/processed/master_ingredients_v3.csv'
MODEL_PATH = 'models/haram_detector_model.pkl'
VECT_PATH = 'models/tfidf_vectorizer.pkl'
OUTPUT_DATA = 'data/processed/final_ingredients_labeled.csv'


# 1. LOAD RESOURCES
print("🤖 Loading resources...")
try:
    df = pd.read_csv(INPUT_DATA)
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    print("   ✅ Data, Model, and Vectorizer loaded.")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit()

# 2. SEPARATE DATA
# Filter data that is currently Unknown (NaN)
unknown_data = df[df['Label'].isna()].copy()
known_data = df.dropna(subset=['Label']).copy()

print(f"   📊 Stats:")
print(f"   - Known Labels: {len(known_data)}")
print(f"   - Unknown (To Predict): {len(unknown_data)}")

if len(unknown_data) == 0:
    print("   🎉 No unknown data! Everything is labeled.")
    exit()


# 3. AI PREDICTION 🧠
print("\n🔮 AI is predicting unknown ingredients...")

# Convert text to numbers using the loaded vectorizer
X_unknown = vectorizer.transform(unknown_data['Name'].astype(str))

# Predict Label (0 or 1)
predictions = model.predict(X_unknown)

# Predict Probability (Confidence level)
# model.predict_proba gives probability for [0, 1]
# We take the maximum probability as "Confidence Score"
probs = model.predict_proba(X_unknown)
confidence = np.max(probs, axis=1)

# Assign results back to dataframe
unknown_data['Label'] = predictions
unknown_data['Prediction_Source'] = 'AI_Model' # Mark that AI guessed this
unknown_data['Confidence'] = confidence

# Mark known data source as 'Manual_Rule'
known_data['Prediction_Source'] = 'Manual_Rule'
known_data['Confidence'] = 1.0 # 100% confident for rules


# 4. MERGE & SAVE
print("\n🔄 Merging and Saving...")
final_df = pd.concat([known_data, unknown_data], ignore_index=True)

# Save to CSV
final_df.to_csv(OUTPUT_DATA, index=False)

print(f"🎉 SUCCESS! Final Database Saved: {OUTPUT_DATA}")
print(f"📊 New Stats:\n{final_df['Label'].value_counts()}")
print(f"   - AI Detected Haram: {len(unknown_data[unknown_data['Label']==1.0])}")
print(f"   - AI Detected Halal: {len(unknown_data[unknown_data['Label']==0.0])}")