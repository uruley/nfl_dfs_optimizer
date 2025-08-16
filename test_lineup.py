import pandas as pd

df = pd.read_csv('data/salaries_with_position_specific.csv')
df['value'] = df['proj_fpts'] / (df['salary_final'] / 1000)

qb = df[df['pos'] == 'QB'].nlargest(1, 'value')
rb = df[df['pos'] == 'RB'].nlargest(2, 'value') 
wr = df[df['pos'] == 'WR'].nlargest(3, 'value')
te = df[df['pos'] == 'TE'].nlargest(1, 'value')
dst = df[df['pos'] == 'DST'].nlargest(1, 'value')

lineup = pd.concat([qb, rb, wr, te, dst])
total_salary = int(lineup['salary_final'].sum())
total_proj = lineup['proj_fpts'].sum()

print('LINEUP TEST RESULTS:')
print('Players:', len(lineup))
print('Total Salary:', total_salary)
print('Projected Points:', round(total_proj, 1))
print('Remaining Cap:', 50000 - total_salary)
print()
print('Lineup:')
for _, player in lineup.iterrows():
    sal = int(player['salary_final'])
    pts = round(player['proj_fpts'], 1)
    print(f"{player['name']} ({player['pos']}): ${sal} salary, {pts} pts")