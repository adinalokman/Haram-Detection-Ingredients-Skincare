import pandas as pd
import os
import re


# CONFIGURATION
RAW_PATH = 'data/raw/'
PROCESSED_PATH = 'data/processed/'
os.makedirs(PROCESSED_PATH, exist_ok=True)

def clean_text(text):
    """
    Clean ingredient text: remove parentheses, lowercase, strip spaces.
    Only keep text that looks like an ingredient (< 60 chars).
    """
    if not isinstance(text, str): return ""
    
    # 1. Cleaning standard
    text = re.sub(r'\s*\(.*?\)', '', text) # Buang (Aqua) etc
    text = text.strip().lower()
    
    # 2. NOISE FILTER 🗑️
    if len(text) > 50 or len(text) < 3:
        return None  
        
    return text

# 1. LOAD REFERENCE LABELS
print("📚 Loading old labeled dataset (Reference)...")
try:
    df_old = pd.read_csv(os.path.join(RAW_PATH, 'ingredientsList_balanced_150.csv'), 
                         usecols=['name', 'Halal_Status'])
    df_old.columns = ['Name', 'Label'] 
    
    # Map text labels to numbers (0=Halal, 1=Haram, 2=Syubhah)
    label_map = {'haram': 1, 'halal': 0, 'shubhah': 2} 
    df_old['Label'] = df_old['Label'].replace(label_map)
    
    # Clean & Drop None
    df_old['Name'] = df_old['Name'].apply(clean_text)
    df_old = df_old.dropna(subset=['Name'])
    
    print(f"✅ Loaded {len(df_old)} reference labels.")
except Exception as e:
    print(f"❌ Error loading reference data: {e}")
    df_old = pd.DataFrame(columns=['Name', 'Label'])


# 2. HARVEST INGREDIENTS
ingredients_bucket = []

def extract_from_csv(filename, col_name, is_comma_separated=True):
    path = os.path.join(RAW_PATH, filename)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if col_name in df.columns:
                print(f"   extracting from {filename}...")
                for item in df[col_name].dropna():
                 
                    if is_comma_separated:
                        parts = str(item).split(',')
                        for p in parts:
                            cleaned = clean_text(p)
                            if cleaned: ingredients_bucket.append(cleaned)
                    else:
                        cleaned = clean_text(item)
                        if cleaned: ingredients_bucket.append(cleaned)
            else:
                print(f"⚠️ Column '{col_name}' not found in {filename}")
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")

# RUN EXTRACTION
print("\n📚 Processing all datasets...")
extract_from_csv('cosmetics.csv', 'Ingredients')
extract_from_csv('ingredientsList.csv', 'name', is_comma_separated=False)

# Auto-detect 2024 file
feb_files = [f for f in os.listdir(RAW_PATH) if '2024' in f and f.endswith('.csv')]
if feb_files:
    extract_from_csv(feb_files[0], 'text', is_comma_separated=True)


# 3. MERGE & CLEAN
print("\n🔄 Cleaning & Removing Duplicates...")
unique_ingredients = list(set(ingredients_bucket))
df_master = pd.DataFrame(unique_ingredients, columns=['Name'])

print(f"   Original Count: {len(ingredients_bucket)}")
print(f"   Unique Clean Count: {len(df_master)}")


df_final = df_master.merge(df_old, on='Name', how='left')


# 4. SMART AUTO-LABELING (Expanded) 🧠

print("🔍 Running Expanded Smart Labeling...")

# A. HARAM (1.0) - Haiwan/Alkohol
haram_keywords = [
    'pig', 'porcine', 'lard', 'boar', 'swine', 'hog', 'tallow', 'carmine', 
    'cochineal', 'wine', 'beer', 'alcohol denat', 'sd alcohol'
]

# B. SYUBHAH (2.0) - Ragu-ragu (Haiwan atau Tumbuhan?)
syubhah_keywords = [
    'collagen', 'gelatin', 'glycerin', 'glycerol', 'stearic acid', 
    'keratin', 'biotin', 'hyaluronic acid', 'panthenol', 'allantoin'
]

# C. HALAL (0.0) - Tumbuhan/Mineral/Air (BARU TAMBAH!) 🌿
halal_keywords = [
    'water', 'aqua', 'extract', 'oil', 'leaf', 'root', 'flower', 'fruit', 
    'seed', 'juice', 'butter', 'aloe', 'shea', 'cocoa', 'bark', 'stem', 
    'peel', 'ferment', 'filtrate', 'vitamin', 'niacinamide', 'tocopherol',
    'silica', 'mica', 'titanium dioxide', 'iron oxide', 'kaolin', 'clay'
]

def apply_expanded_rules(row):
    name = str(row['Name'])
    current_label = row['Label']
    
    # 1. kekalkan label if ada
    if pd.notna(current_label): return current_label
    
    # 2. Check HARAM
    if any(k in name for k in haram_keywords): return 1.0
    
    # 3. Check HALAL (Selamat)
    if any(k in name for k in halal_keywords): return 0.0
    
    # 4. Check SYUBHAH
    if any(k in name for k in syubhah_keywords): return 2.0
    
    return current_label # Masih NaN

df_final['Label'] = df_final.apply(apply_expanded_rules, axis=1)


# 5. SAVE

outfile = os.path.join(PROCESSED_PATH, 'master_ingredients_v3.csv')
df_final.to_csv(outfile, index=False)

print(f"\n🎉 SUCCESS! Saved to: {outfile}")
print(f"📊 Final Stats:\n{df_final['Label'].value_counts(dropna=False)}")