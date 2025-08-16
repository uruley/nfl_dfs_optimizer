import pandas as pd

df = pd.read_csv('data/salaries_with_position_specific.csv')
df['value'] = df['proj_fpts'] / (df['salary_final'] / 1000)

print('🎯 SMART DFS LINEUP OPTIMIZER')
print('=' * 35)

# Sort all players by value
df_sorted = df.sort_values('value', ascending=False)

# Build lineup with proper position requirements
lineup = []
salary_used = 0
positions_filled = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'DST': 0}
position_limits = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1}

# First pass - fill required positions
for _, player in df_sorted.iterrows():
    pos = player['pos']
    salary = player['salary_final']
    
    # Check if we need this position and can afford it
    if (pos in position_limits and 
        positions_filled[pos] < position_limits[pos] and
        salary_used + salary <= 50000):
        
        lineup.append(player.name)
        salary_used += salary
        positions_filled[pos] += 1
        
        print(f"Added {player['name']} ({pos}): ${int(salary):,} - {player['proj_fpts']:.1f} pts")
        
        # Check if we have all required positions
        if all(positions_filled[p] >= position_limits[p] for p in position_limits):
            break

# Add FLEX if we have budget and need one more player
total_players = sum(positions_filled.values())
if total_players == 8:  # Need one more for FLEX
    remaining_budget = 50000 - salary_used
    
    # Find best FLEX player we can afford
    flex_candidates = df[
        (df['pos'].isin(['RB', 'WR', 'TE'])) & 
        (~df['name'].isin(lineup)) &
        (df['salary_final'] <= remaining_budget)
    ].sort_values('value', ascending=False)
    
    if len(flex_candidates) > 0:
        flex_player = flex_candidates.iloc[0]
        lineup.append(flex_player['name'])
        salary_used += flex_player['salary_final']
        print(f"Added FLEX {flex_player['name']} ({flex_player['pos']}): ${int(flex_player['salary_final']):,} - {flex_player['proj_fpts']:.1f} pts")

# Final results
lineup_df = df[df['name'].isin(lineup)]
total_proj = lineup_df['proj_fpts'].sum()

print(f"\n🏆 FINAL LINEUP:")
print(f"Players: {len(lineup)}/9")
print(f"Total Salary: ${int(salary_used):,}")
print(f"Remaining: ${50000 - int(salary_used):,}")
print(f"Projected: {total_proj:.1f} points")

if len(lineup) == 9 and salary_used <= 50000:
    print("✅ COMPLETE VALID LINEUP!")
    
    if total_proj >= 140:
        print("🔥 High-scoring lineup - tournament ready!")
    elif total_proj >= 130:
        print("⚡ Solid lineup - cash game viable!")
else:
    print("❌ Lineup incomplete or over budget")