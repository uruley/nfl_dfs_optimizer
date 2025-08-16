import pandas as pd

df = pd.read_csv('data/salaries_with_position_specific.csv')

# Simple lineup: pick cheapest players that give us 9 players under $50K
qb = df[df['pos'] == 'QB'].nsmallest(1, 'salary_final')
rb = df[df['pos'] == 'RB'].nsmallest(2, 'salary_final') 
wr = df[df['pos'] == 'WR'].nsmallest(3, 'salary_final')
te = df[df['pos'] == 'TE'].nsmallest(1, 'salary_final')
dst = df[df['pos'] == 'DST'].nsmallest(1, 'salary_final')

# Get one more RB as FLEX
used_players = list(qb.index) + list(rb.index) + list(wr.index) + list(te.index) + list(dst.index)
flex = df[(df['pos'] == 'RB') & (~df.index.isin(used_players))].nsmallest(1, 'salary_final')

lineup = pd.concat([qb, rb, wr, te, dst, flex])

print("SIMPLE LINEUP TEST:")
print(f"Players: {len(lineup)}")
print(f"Total Salary: ${lineup['salary_final'].sum():,.0f}")
print(f"Projected: {lineup['proj_fpts'].sum():.1f}")
print(f"Under budget: {lineup['salary_final'].sum() <= 50000}")

print("\nLineup:")
for _, p in lineup.iterrows():
    print(f"{p['name']} ({p['pos']}): ${p['salary_final']:,.0f}")