import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os


# CONFIGURATION
# Load the latest processed data
INPUT_FILE = 'data/processed/master_ingredients_v3.csv'
OUTPUT_DIR = 'reports/figures/'

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Data
print("📊 Loading data for visualization...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"❌ Error: File not found at {INPUT_FILE}. Please run data_preprocessing.py first.")
    exit()

# Drop NaN for visualization purposes
df_clean = df.dropna(subset=['Label'])
df_clean['Label'] = df_clean['Label'].astype(int)

# Map numbers back to text for better charts
label_map = {0: 'Halal', 1: 'Haram', 2: 'Syubhah'}
df_clean['Label_Text'] = df_clean['Label'].map(label_map)

# 1. BAR CHART: Label Distribution
print("   Generating Bar Chart...")
plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x='Label_Text', palette='viridis', order=['Halal', 'Haram', 'Syubhah'])
plt.title('Distribution of Ingredients by Halal Status')
plt.xlabel('Category')
plt.ylabel('Count')
plt.savefig(os.path.join(OUTPUT_DIR, 'label_distribution.png'))
plt.close()
print("   ✅ Bar chart saved.")


# 2. WORD CLOUD GENERATOR
def generate_wordcloud(label_code, title, filename):
    # Filter text by label
    subset = df_clean[df_clean['Label'] == label_code]
    
    if subset.empty:
        print(f"   ⚠️ No data found for {title}, skipping WordCloud.")
        return

    text_data = " ".join(subset['Name'].astype(str))
    
    # Generate Cloud
    wc = WordCloud(width=800, height=400, background_color='white', colormap='Dark2').generate(text_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Most Common Words in {title} Ingredients', fontsize=15)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"   ✅ {title} WordCloud saved.")

# Generate Clouds
print("   Generating WordClouds...")
generate_wordcloud(0, 'Halal', 'wordcloud_halal.png')
generate_wordcloud(1, 'Haram', 'wordcloud_haram.png')

print(f"\n🎉 All charts saved in: {OUTPUT_DIR}")