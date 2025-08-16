import pandas as pd
df = pd.read_csv('data/salaries_merged.csv')
df['value'] = df['proj_fpts'] / (df['salary_final'] / 1000)

lineup = []
for pos, count in [('QB',1), ('RB',2), ('WR',3), ('TE',1), ('DST',1)]:
    selected = df[df['pos']==pos].nlargest(count, 'value')
    lineup.extend(selected.index.tolist())

# Add FLEX
flex = df[df['pos'].isin(['RB','WR','TE']) & ~df.index.isin(lineup)].nlargest(1, 'value')
lineup.extend(flex.index.tolist())

result = df.loc[lineup]
print('🏆 OPTIMAL LINEUP:')
print(result[['name','pos','proj_fpts','salary_final']].to_string(index=False))
print(f'\nProjected: {result.proj_fpts.sum():.1f} pts')
print(f'Salary: ')

if 'actual_fpts' in result.columns:
    actual = result.actual_fpts.sum()
    print(f'Actual: {actual:.1f} pts')
    print('Result:', '✅ CASH' if actual >= 135 else '⚠️ BORDERLINE' if actual >= 120 else '❌ NO CASH')
