# features.py

import pandas as pd
import numpy as np
from DFSDataManager import DFSDataManager

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=["player_id", "season", "week"])
    for window in [3, 5, 8]:
        df[f'fp_avg_{window}'] = (
            df.groupby("player_id")["fantasy_points"].shift(1).rolling(window).mean()
        )
    return df

def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    # Opponent defense strength placeholder
    df["opp_def_rank"] = np.random.randint(1, 33, size=len(df))  # TEMPORARY until real data
    return df

def add_environment_features(df: pd.DataFrame) -> pd.DataFrame:
    # Safe fallback if "team" column missing
    if "team" not in df.columns or "home_team" not in df.columns:
        df["is_home"] = False
    else:
        df["is_home"] = df["team"] == df["home_team"]

    # Handle missing stadium column
    if "stadium" not in df.columns:
        df["indoors"] = False
    else:
        df["indoors"] = df["stadium"].fillna("").str.lower().str.contains("dome|indoor")

    return df

def build_features():
    dm = DFSDataManager()
    df = dm.load("weekly_2020_2023")

    # Drop preseason + clean up
    df = df[df["week"] <= 18]

    # Add engineered features
    df = add_rolling_features(df)
    df = add_matchup_features(df)
    df = add_environment_features(df)

    # Drop rows without past rolling data
    df = df.dropna(subset=["fp_avg_3", "fp_avg_5", "fp_avg_8"])

    # Output final DataFrame
    output_path = "data/processed/features.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Features saved to: {output_path}")
    print(f"📊 Shape: {df.shape}")

if __name__ == "__main__":
    build_features()
# To run this script, ensure you have the necessary data files in 'data/raw' and the DFSDataManager class available.