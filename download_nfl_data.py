# download_nfl_data.py

import nfl_data_py as nfl
import pandas as pd
import os

YEARS = list(range(2020, 2024))  # 2020 to 2023

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_csv(df: pd.DataFrame, name: str):
    path = f"data/raw/{name}.csv"
    df.to_csv(path, index=False)
    print(f"✅ Saved: {path}")

if __name__ == "__main__":
    ensure_folder("data/raw")

    print("📦 Downloading weekly data...")
    weekly = nfl.import_weekly_data(YEARS)
    save_csv(weekly, "weekly_2020_2023")

    print("📦 Downloading snap counts...")
    snaps = nfl.import_snap_counts(YEARS)
    save_csv(snaps, "snap_counts_2020_2023")

    print("📦 Downloading schedules...")
    sched = nfl.import_schedules(YEARS)
    save_csv(sched, "schedules_2020_2023")

    print("📦 Downloading play-by-play data (big)...")
    pbp = nfl.import_pbp_data(YEARS, downcast=True)
    save_csv(pbp, "pbp_2020_2023")

    # Optional: rosters (fetched another way if needed)
    try:
        print("📦 Attempting to load rosters (manual fallback)...")
        roster_url = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/rosters.csv"
        rosters = pd.read_csv(roster_url)
        save_csv(rosters, "rosters_2020_2023")
    except Exception as e:
        print(f"❌ Failed to fetch rosters: {e}")
