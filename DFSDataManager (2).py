# DFSDataManager.py

import os
import pandas as pd

class DFSDataManager:
    def __init__(self, raw_path="data/raw", processed_path="data/processed"):
        self.raw_path = raw_path
        self.processed_path = processed_path

    def _file_path(self, name, processed=False):
        folder = self.processed_path if processed else self.raw_path
        return os.path.join(folder, f"{name}.csv")

    def load(self, name, processed=False):
        path = self._file_path(name, processed)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        print(f"📂 Loading {'processed' if processed else 'raw'}: {name}")
        return pd.read_csv(path)

    def list_raw_files(self):
        return [f for f in os.listdir(self.raw_path) if f.endswith(".csv")]

    def save_processed(self, df, name):
        path = self._file_path(name, processed=True)
        df.to_csv(path, index=False)
        print(f"✅ Saved processed file: {path}")
