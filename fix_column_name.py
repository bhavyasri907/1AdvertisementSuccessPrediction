# fix_column_name.py
import pandas as pd
import os

print("🔧 Fixing dataset column name...")

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Check if dataset exists
if not os.path.exists("data/train.csv"):
    print("❌ Error: data/train.csv not found!")
    print("Please place your dataset in data/train.csv")
    exit(1)

# Load dataset
df = pd.read_csv("data/train.csv")
print(f"✅ Loaded dataset with {len(df)} rows")

# Fix column name
if 'realtionship_status' in df.columns:
    df = df.rename(columns={'realtionship_status': 'relationship_status'})
    print("✅ Fixed column name: 'realtionship_status' → 'relationship_status'")
else:
    print("ℹ️ Column 'realtionship_status' not found, checking alternatives...")
    
# Save fixed dataset
df.to_csv("data/train_fixed.csv", index=False)
print("✅ Saved fixed dataset to data/train_fixed.csv")

# Also create a backup
df.to_csv("data/train_backup.csv", index=False)
print("✅ Created backup at data/train_backup.csv")