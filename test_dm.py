# test_dm.py

from DFSDataManager import DFSDataManager

dm = DFSDataManager()

# Try loading a file we know exists
weekly_df = dm.load("weekly_2020_2023")

# Show shape and a few rows
print("✅ Loaded weekly data:")
print(weekly_df.shape)
print(weekly_df.head())
