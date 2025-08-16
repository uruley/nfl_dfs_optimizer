import pandas as pd

df = pd.read_csv('data/salaries_with_position_specific.csv')
df['value'] = df['proj_fpts'] / (df['salary_final'] / 1000)

print('🏈 DFS LINEUP OPTIMIZER WITH BUDGET CONSTRAINT')
print('=' * 50)

# Build lineup within budget
lineup = []
remaining_budget = 50000
positions_needed = [('QB', 1), ('RB', 2), ('WR', 3), ('TE', 1), ('DST', 1)]

for pos, count in positions_needed:
    pos_players = df[df['pos'] == pos].copy()
    
    # Filter by budget constraint
    affordable = pos_players[pos_players['salary_final'] <= remaining_budget/2]  # Conservative budget
    
    if len(affordable) >= count:
        selected = affordable.nlargest(count, 'value')
        lineup.extend(selected.index.tolist())
        remaining_budget -= selected['salary_final'].sum()
        print(f"{pos}: {len(selected)} players selected, ${int(selected['salary_final'].sum())} spent")
    else:
        print(f"❌ Not enough affordable {pos} players")

# Add FLEX if budget allows
remaining_players = df[df['pos'].isin(['RB','WR','TE']) & ~df.index.isin(lineup)]
affordable_flex = remaining_players[remaining_players['salary_final'] <= remaining_budget]

if len(affordable_flex) > 0:
    flex = affordable_flex.nlargest(1, 'value')
    lineup.extend(flex.index.tolist())
    remaining_budget -= flex['salary_final'].sum()
    print(f"FLEX: 1 player selected, ${int(flex['salary_final'].sum())} spent")

# Final lineup
if len(lineup) == 9:
    lineup_df = df.loc[lineup]
    total_salary = int(lineup_df['salary_final'].sum())
    total_proj = lineup_df['proj_fpts'].sum()
    
    print(f"\n🏆 FINAL LINEUP (9 players):")
    print(f"Total Salary: ${total_salary:,}")
    print(f"Remaining: ${50000 - total_salary:,}")
    print(f"Projected: {total_proj:.1f} points")
    
    if total_salary <= 50000:
        print("✅ UNDER BUDGET!")
    else:
        print("❌ Still over budget")
        
    print(f"\nLineup Details:")
    for _, player in lineup_df.iterrows():
        print(f"{player['name']} ({player['pos']}): ${int(player['salary_final']):,} - {player['proj_fpts']:.1f} pts")
else:
    print(f"❌ Only built {len(lineup)}/9 players")