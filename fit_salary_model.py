# Fits salary ~ a + b * FPTS per position using all CSVs in data/salaries_raw
# Writes model coefficients to data/salary_model.json

import json, glob, os
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"C:\Users\ruley\dfs_optimizer\data\salaries_raw")
OUT_JSON = Path(r"C:\Users\ruley\dfs_optimizer\data\salary_model.json")

files = sorted(glob.glob(str(RAW_DIR / "dk_salaries_*.csv")))
if not files:
    raise SystemExit(f"No files found in {RAW_DIR}")

frames = []
for fp in files:
    df = pd.read_csv(fp)
    # expected cols from your ingest script:
    # season,week,rank,name,team,opp,pos,opp_rank,opp_pos_rank,salary,fpts
    keep = [c for c in ["season","week","name","team","opp","pos","salary","fpts"] if c in df.columns]
    df = df[keep].dropna(subset=["salary","fpts","pos"])
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(all_df)} rows from {len(files)} files")

coefs = {}
for pos, grp in all_df.groupby("pos"):
    if len(grp) < 8:   # need enough points to be stable
        print(f"Skipping {pos}: only {len(grp)} rows")
        continue
    X = np.column_stack([np.ones(len(grp)), grp["fpts"].values])  # [1, FPTS]
    y = grp["salary"].values
    # OLS: beta = (X'X)^-1 X'y
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    a, b = float(beta[0]), float(beta[1])
    coefs[pos] = {"a": a, "b": b, "n": int(len(grp))}
    print(f"{pos}: salary ≈ {a:.1f} + {b:.1f}*FPTS (n={len(grp)})")

# Fallbacks for sparse positions (adjust later as you gather more weeks)
DEFAULTS = {
    "QB": {"a": 2000.0, "b": 240.0},
    "RB": {"a": 2000.0, "b": 310.0},
    "WR": {"a": 1500.0, "b": 335.0},
    "TE": {"a":  800.0, "b": 380.0},
    "DST": {"a": 1000.0, "b": 220.0},
    "K":   {"a": 1000.0, "b": 180.0},  # DK NFL usually has no K; keep as placeholder
}
for p, v in DEFAULTS.items():
    coefs.setdefault(p, v)

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(coefs, indent=2))
print(f"Saved coefficients -> {OUT_JSON}")
