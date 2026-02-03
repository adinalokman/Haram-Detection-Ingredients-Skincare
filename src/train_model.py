import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# CONFIGURATION

INPUT_FILE = 'data/processed/master_ingredients_v3.csv'
MODEL_DIR = 'models/'
REPORT_DIR = 'reports/'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# 1. LOAD & PREPARE DATA
print("🤖 Loading dataset...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"❌ Error: File {INPUT_FILE} not found. Run preprocessing first.")
    exit()

#train using the data that has this label (Halal/Haram)
# Buang NaN dan ambil label 0.0 (Halal) & 1.0 (Haram) sahaja
train_data = df.dropna(subset=['Label'])
train_data = train_data[train_data['Label'].isin([0.0, 1.0])]

print(f"   Training samples available: {len(train_data)}")
print(f"   - Halal (0.0): {len(train_data[train_data['Label']==0.0])}")
print(f"   - Haram (1.0): {len(train_data[train_data['Label']==1.0])}")

# X = Nama Bahan, y = Label
X = train_data['Name'].astype(str)
y = train_data['Label'].astype(int)

# Pecahkan data: 80% untuk train, 20% untuk test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. FEATURE EXTRACTION (TF-IDF)
print("\n🔠 Converting text to numbers (TF-IDF)...")
# tukar jadi number
vectorizer = TfidfVectorizer(ngram_range=(1, 2)) 
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# 3. TRAIN MODEL 

print("\n🧠 Training Logistic Regression Model...")
# 'class_weight=balanced' 
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_vec, y_train)

print("   ✅ Model trained successfully!")


# 4. EVALUATE PERFORMANCE

print("\n📊 Evaluating Model...")
y_pred = model.predict(X_test_vec)

# Print Report Accuracy
report = classification_report(y_test, y_pred, target_names=['Halal', 'Haram'])
print(report)

# Save Report text
with open(os.path.join(REPORT_DIR, 'model_performance.txt'), 'w') as f:
    f.write(report)

# Generate Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred Halal', 'Pred Haram'], 
            yticklabels=['Actual Halal', 'Actual Haram'])
plt.title('Confusion Matrix')
plt.savefig(os.path.join(REPORT_DIR, 'confusion_matrix.png'))
plt.close()


# 5. SAVE MODEL
print("\n💾 Saving Model & Vectorizer...")
joblib.dump(model, os.path.join(MODEL_DIR, 'haram_detector_model.pkl'))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))

print(f"   🎉 Model saved to {MODEL_DIR}")