# train_model.py
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                           precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting model training...")
print("=" * 50)

# Create directories if they don't exist
os.makedirs("model", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Check if fixed dataset exists, otherwise use original
if os.path.exists("data/train_fixed.csv"):
    df = pd.read_csv("data/train_fixed.csv")
    print("✅ Using fixed dataset")
else:
    df = pd.read_csv("data/train.csv")
    print("⚠️ Using original dataset (run fix_column_name.py first)")
    
    # Fix column name on the fly
    if 'realtionship_status' in df.columns:
        df = df.rename(columns={'realtionship_status': 'relationship_status'})

print(f"📊 Dataset Info:")
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"   Missing values: {df.isnull().sum().sum()}")

# Drop unwanted columns
df = df.drop(["UserID"], axis=1, errors='ignore')

# Separate features and targets
X = df.drop(["netgain", "ratings", "money_back_guarantee"], axis=1, errors='ignore')
y_ratings = df["ratings"]
y_success = df["netgain"]
y_money = df["money_back_guarantee"]

print(f"\n📈 Feature sets:")
print(f"   Features: {X.columns.tolist()}")
print(f"   Target (ratings): {y_ratings.name}")
print(f"   Target (success): {y_success.name}")
print(f"   Target (money): {y_money.name}")

# Train-test split
X_train, X_test, y_train_ratings, y_test_ratings = train_test_split(
    X, y_ratings, test_size=0.2, random_state=42
)
_, _, y_train_success, y_test_success = train_test_split(
    X, y_success, test_size=0.2, random_state=42, stratify=y_success
)
_, _, y_train_money, y_test_money = train_test_split(
    X, y_money, test_size=0.2, random_state=42, stratify=y_money
)

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\n📊 Column types:")
print(f"   Categorical: {categorical_cols}")
print(f"   Numerical: {numerical_cols}")

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ],
    remainder="passthrough"
)

# Create models
rating_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

success_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ))
])

money_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ))
])

print("\n🤖 Training models...")

# Train rating model
print("   Training rating model...")
rating_model.fit(X_train, y_train_ratings)
y_pred_ratings = rating_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test_ratings, y_pred_ratings))
r2 = r2_score(y_test_ratings, y_pred_ratings)

print(f"\n📊 Rating Model Performance:")
print(f"   RMSE: {rmse:.4f}")
print(f"   R² Score: {r2:.4f}")

# Train success model
print("   Training success model...")
success_model.fit(X_train, y_train_success)
y_pred_success = success_model.predict(X_test)
print(f"\n📊 Success Model Performance:")
print(f"   Accuracy: {accuracy_score(y_test_success, y_pred_success):.4f}")

# Train money model
print("   Training money model...")
money_model.fit(X_train, y_train_money)
y_pred_money = money_model.predict(X_test)
print(f"\n📊 Money-back Model Performance:")
print(f"   Accuracy: {accuracy_score(y_test_money, y_pred_money):.4f}")

# Feature importance (for rating model)
try:
    feature_names = categorical_cols + numerical_cols
    importance = rating_model.named_steps['regressor'].feature_importances_
    
    plt.figure(figsize=(10, 6))
    feature_importance_df = pd.DataFrame({
        'feature': feature_names[:len(importance)],
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importances (Rating Model)')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png')
    print("\n✅ Feature importance plot saved to reports/feature_importance.png")
except Exception as e:
    print(f"\n⚠️ Could not generate feature importance plot: {e}")

# Save models
models = {
    'rating_model': rating_model,
    'success_model': success_model,
    'money_model': money_model,
    'feature_names': X.columns.tolist(),
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols
}

with open("model/model.pkl", "wb") as f:
    pickle.dump(models, f)

print("\n✅ Models trained and saved successfully!")
print("📁 Model saved to: model/model.pkl")
print("=" * 50)