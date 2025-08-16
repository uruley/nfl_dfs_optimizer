# ingest_salary_text.py
# Usage examples:
#   python Scripts\ingest_salary_text.py --season 2024 --week 11 --in data\incoming.txt
#   python Scripts\ingest_salary_text.py --season 2024 --week 11 --in data\incoming.txt --append
import argparse, re
from pathlib import Path
import pandas as pd

POS_SET = {"QB","RB","WR","TE","DST","D/ST","K"}

def _clean(s: str) -> str:
    # normalize spaces, strip NBSP and weird whitespace
    return s.replace("\u00a0", " ").replace("\u200b","").strip()

def _to_int(x):
    try: return int(str(x).replace("$","").replace(",","").strip())
    except: return None

def _to_float(x):
    try: return float(str(x).replace(",","").strip())
    except: return None

def parse_text(text: str, season: int, week: int):
    raw_lines = text.replace("\r","").split("\n")
    lines = []
    for l in raw_lines:
        l = _clean(l)
        if not l:
            continue
        u = l.upper()
        # drop navigation / headers that reappear
        if u.startswith("NAME") and "POS" in u:  # header
            continue
        if "NFL DAILY FANTASY SALARIES" in u:    # page title
            continue
        if u in {"CSV","XLS","PAGE 1","PAGE 2","PAGE 3","NEXT","PREV"}:
            continue
        if "PAGE" in u and "NEXT" in u:
            continue
        lines.append(l)

    out = []
    i = 0
    ranks_seen = 0
    failed_blocks = 0
    while i < len(lines):
        l = lines[i]
        if re.fullmatch(r"\d+", l):
            # rank
            ranks_seen += 1
            rank = int(l); i += 1
            if i+2 >= len(lines):
                break
            name = lines[i]; i += 1
            team = lines[i]; i += 1
            opp  = lines[i]; i += 1

            # find a line within next ~6 lines that looks like: POS <int> <int> <money> <float>
            pos_line = None
            lookahead = 0
            while i < len(lines) and lookahead < 6:
                cand = lines[i]
                parts = re.split(r"[ \t]+", cand)
                if parts and parts[0].upper() in POS_SET and len(parts) >= 5:
                    pos_line = parts
                    i += 1
                    break
                i += 1
                lookahead += 1

            if not pos_line:
                failed_blocks += 1
                continue

            pos = pos_line[0].upper()
            opp_rank     = _to_int(pos_line[1]) if len(pos_line) > 1 else None
            opp_pos_rank = _to_int(pos_line[2]) if len(pos_line) > 2 else None
            salary       = _to_int(pos_line[3]) if len(pos_line) > 3 else None
            fpts         = _to_float(pos_line[4]) if len(pos_line) > 4 else None

            # normalize D/ST to DST
            if pos == "D/ST": pos = "DST"

            out.append({
                "season": season,
                "week": week,
                "rank": rank,
                "name": name,
                "team": team,
                "opp": opp,
                "pos": pos,
                "opp_rank": opp_rank,
                "opp_pos_rank": opp_pos_rank,
                "salary": salary,
                "fpts": fpts,
            })
        else:
            i += 1

    return out, ranks_seen, failed_blocks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--in", dest="infile", type=str, default=r"data\incoming.txt",
                   help="path to raw text you pasted (default: data\\incoming.txt)")
    p.add_argument("--append", action="store_true",
                   help="merge with existing CSV if present (dedupe by name+team+pos+salary)")
    args = p.parse_args()

    raw_path = Path(args.infile)
    if not raw_path.exists():
        raise SystemExit(f"Raw text file not found: {raw_path}")

    text = raw_path.read_text(encoding="utf-8", errors="ignore")
    rows, ranks_seen, failed = parse_text(text, args.season, args.week)

    out_dir = Path("data") / "salaries_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dk_salaries_{args.season}_w{args.week:02d}.csv"

    new_df = pd.DataFrame(rows)
    if args.append and out_path.exists():
        old_df = pd.read_csv(out_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["name","team","pos","salary"], keep="last")
        combined = combined.sort_values(by=["rank","name"]).reset_index(drop=True)
        combined.to_csv(out_path, index=False)
        print(f"APPEND mode: parsed={len(new_df)} ranks_seen={ranks_seen} failed_blocks={failed} -> {out_path} (rows now {len(combined)})")
    else:
        new_df.to_csv(out_path, index=False)
        print(f"SAVED: parsed={len(new_df)} ranks_seen={ranks_seen} failed_blocks={failed} -> {out_path}")

if __name__ == "__main__":
    main()
