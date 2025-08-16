import pandas as pd, numpy as np, glob
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
RAW_DIR = DATA / "salaries_raw"
SYN_FILE = DATA / "projections_with_salaries.csv"
OUT_ALL = DATA / "salaries_merged.csv"

# load synthetic/proj salaries
syn = pd.read_csv(SYN_FILE)
syn.columns = [c.lower() for c in syn.columns]

# load all real DK salaries
dfs = []
for fp in glob.glob(str(RAW_DIR / "dk_salaries_*.csv")):
    df = pd.read_csv(fp)
    df.columns = [c.lower() for c in df.columns]
    keep = [c for c in ["season","week","name","team","opp","pos","salary","fpts","rank"] if c in df.columns]
    dfs.append(df[keep])
real_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["season","week","name","team","opp","pos","salary"])

# normalize keys (trim spaces)
for df in (syn, real_all):
    for col in ("name","team","opp","pos"):
        if col in df.columns: df[col] = df[col].astype(str).str.strip()

# merge on exact keys
merged = syn.merge(real_all, on=["season","week","name","team","opp","pos"], how="left", suffixes=("","_real"))

# prefer real
merged["salary_final"] = merged["salary"].fillna(merged.get("proj_salary"))
merged["salary_source"] = np.where(merged["salary"].notna(), "real",
                              np.where(merged.get("proj_salary").notna(), "proj", "missing"))

# drop rows with no salary at all
merged = merged[merged["salary_final"].notna()].copy()

# order columns
cols_order = [c for c in ["season","week","name","team","opp","pos","proj_fpts","salary","proj_salary","salary_final","salary_source","fpts","rank"] if c in merged.columns]
merged = merged[cols_order + [c for c in merged.columns if c not in cols_order]]

OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_ALL, index=False)
print(f"✅ merged rows: {len(merged)} -> {OUT_ALL}")
