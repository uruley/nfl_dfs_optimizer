import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def get_training_features(model):
    """Return the features the model was trained on."""
    return model.feature_names_in_

# Load the full features dataset
df = pd.read_csv("data/processed/features.csv")

# Filter only 2023 season
df_2023 = df[df["season"] == 2023].copy()

# Load model
model = joblib.load("models/qb_model.pkl")

# Automatically align features with training data
X_test = df_2023[get_training_features(model)]
y_test = df_2023["fantasy_points_ppr"]

# Predict
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
correlation = np.corrcoef(y_test, y_pred)[0, 1]

print("\n📊 Backtest Results on 2023:")
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ Correlation: {correlation:.2f}")

# Save predictions
df_2023["predicted_points"] = y_pred
df_2023[["player_name", "week", "fantasy_points_ppr", "predicted_points"]].to_csv(
    "data/processed/qb_predictions_2023.csv", index=False
)
print("✅ Predictions saved to: data/processed/qb_predictions_2023.csv")
