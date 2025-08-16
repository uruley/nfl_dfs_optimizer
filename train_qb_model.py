import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Load feature data
print("📥 Loading features...")
df = pd.read_csv("data/processed/features.csv")

# Filter to QBs only
df = df[df["position"] == "QB"].copy()

# Drop rows with missing fantasy scores
df = df.dropna(subset=["fantasy_points"])

# Set up feature columns - include only numeric features
non_features = [
    "player_id", "player_name", "player_display_name", 
    "fantasy_points", "position", "team", "opponent", 
    "week", "season"
]

# Select only numeric columns not in the ignore list
feature_cols = [
    col for col in df.select_dtypes(include=[np.number]).columns 
    if col not in non_features
]

X = df[feature_cols]
y = df["fantasy_points"]

# Set up time series split
tscv = TimeSeriesSplit(n_splits=5)
r2s = []
rmses = []
maes = []

print("📊 Training model with time series cross-validation...")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2s.append(r2_score(y_test, y_pred))
    # Updated RMSE calculation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmses.append(rmse)
    maes.append(mean_absolute_error(y_test, y_pred))

    print(f"✅ Fold {fold + 1} — R²: {r2s[-1]:.3f}, RMSE: {rmses[-1]:.2f}, MAE: {maes[-1]:.2f}")

# Final model on all data
print("📦 Training final model on full dataset...")
final_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
final_model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/qb_model.pkl")

print("✅ Model saved to: models/qb_model.pkl")
# Print final metrics