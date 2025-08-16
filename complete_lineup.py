import pandas as pd

df = pd.read_csv('data/salaries_merged.csv')
df['value'] = df['proj_fpts'] / (df['salary_final'] / 1000)

lineup = []
for pos, count in [('QB',1), ('RB',2), ('WR',3), ('TE',1), ('DST',1)]:
    selected = df[df['pos']==pos].nlargest(count, 'value')
    lineup.extend(selected.index.tolist())

flex = df[df['pos'].isin(['RB','WR','TE']) & ~df.index.isin(lineup)].nlargest(1, 'value')
lineup.extend(flex.index.tolist())

result = df.loc[lineup]
total_salary = result['salary_final'].sum()
total_proj = result['proj_fpts'].sum()

print(f'Total Salary: ')
print(f'Remaining: ')
print(f'Projected Score: {total_proj:.1f}')

if 'actual_fpts' in result.columns:
    actual_total = result['actual_fpts'].sum()
    print(f'Actual Score: {actual_total:.1f}')
    if actual_total >= 135:
        print('Contest Result: ✅ CASH')
    elif actual_total >= 120:
        print('Contest Result: ⚠️ BORDERLINE')
    else:
        print('Contest Result: ❌ NO CASH')
else:
    print('No actual results available for contest simulation')
