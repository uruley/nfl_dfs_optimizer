# Create the position-specific salary generator
@"
import pandas as pd
from pathlib import Path

def generate_position_specific_salaries():
    print("💰 Generating salaries with position-specific DK formulas...")
    
    proj_file = Path("data/projections.csv")
    if not proj_file.exists():
        print("❌ No projections.csv found")
        return None
    
    df = pd.read_csv(proj_file)
    print(f"✅ Loaded {len(df)} player projections")
    
    formulas = {
        "QB": {"intercept": 1472, "slope": 275},
        "RB": {"intercept": 3491, "slope": 208},
        "WR": {"intercept": 2023, "slope": 298}, 
        "TE": {"intercept": 1706, "slope": 276},
        "DST": {"intercept": 1308, "slope": 238},
        "K": {"intercept": 1308, "slope": 238}
    }
    
    def calculate_salary(row):
        pos = str(row['pos']).upper()
        fpts = row['proj_fpts']
        
        if pos in formulas:
            formula = formulas[pos]
            salary = formula["intercept"] + formula["slope"] * fpts
            return round(salary / 100) * 100
        else:
            return round((2000 + 250 * fpts) / 100) * 100
    
    df['proj_salary'] = df.apply(calculate_salary, axis=1)
    
    output_file = Path("data/projections_with_salaries.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df)} rows → {output_file}")
    
    return df

if __name__ == "__main__":
    generate_position_specific_salaries()
"@ | Out-File -FilePath "Scripts\generate_salaries_position_specific.py" -Encoding UTF8

# Now run it
python Scripts\generate_salaries_position_specific.py