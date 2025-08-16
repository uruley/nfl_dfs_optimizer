import pandas as pd

df = pd.read_csv("data/projections_with_salaries.csv")
df["error"] = df["proj_salary"] - df["salary"]

print(df[["name", "pos", "salary", "proj_salary", "error"]])
print("Mean Absolute Error:", df["error"].abs().mean())
