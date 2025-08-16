import json
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]      # ...\dfs_optimizer
DATA = BASE / "data"
MODEL_JSON = DATA / "salary_model.json"
PROJ_CSV   = DATA / "projections.csv"
OUT_CSV    = DATA / "projections_with_salaries.csv"

# Load projections
if not PROJ_CSV.exists():
    raise FileNotFoundError(f"Missing projections file: {PROJ_CSV}")
df = pd.read_csv(PROJ_CSV)
df.columns = [c.strip().lower() for c in df.columns]

# Accept 'player' or 'name'; 'fpts' or 'proj_fpts'
if "player" in df.columns and "name" not in df.columns:
    df.rename(columns={"player":"name"}, inplace=True)
if "fpts" in df.columns and "proj_fpts" not in df.columns:
    df.rename(columns={"fpts":"proj_fpts"}, inplace=True)

need = {"name","pos","proj_fpts"}
missing = need - set(df.columns)
if missing:
    raise SystemExit(f"projections.csv is missing {missing}. Have: {list(df.columns)}")

for col, default in [("season",0),("week",0),("team",""),("opp","")]:
    if col not in df.columns: df[col] = default
df["pos"] = df["pos"].astype(str).str.upper().str.strip()
df["proj_fpts"] = pd.to_numeric(df["proj_fpts"], errors="coerce")

# Load model (supports keys {"a","b"} or {"intercept","slope"})
if not MODEL_JSON.exists():
    raise FileNotFoundError(f"Missing model: {MODEL_JSON} (run fit_salary_model.py first)")
model = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
def get_ab(pos: str):
    m = model.get(pos) or model.get(pos.upper())
    if not m: m = model.get("WR", {"a":3000.0,"b":215.0})
    a = m.get("a", m.get("intercept", 0.0))
    b = m.get("b", m.get("slope", 0.0))
    return float(a), float(b)

FLOOR = {"QB":4500,"RB":3000,"WR":3000,"TE":2800,"DST":2000,"K":3000}
CAP = 10200
def dk_round(x): return int(round(x/100.0)*100)

def estimate(row):
    pos = row["pos"]
    f = row["proj_fpts"]
    if pd.isna(f): return None
    a,b = get_ab(pos)
    raw = a + b*float(f)
    return dk_round(max(FLOOR.get(pos,3000), min(CAP, raw)))

df["proj_salary"] = df.apply(estimate, axis=1)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"✅ Saved {len(df)} rows → {OUT_CSV}")
