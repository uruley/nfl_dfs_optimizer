# Scripts/debug_lineup_selection.py

import pandas as pd
import numpy as np

def debug_lineup_selection():
    """
    Debug why lineup scored only 85.3 points with good projections
    """
    
    print("🔍 Debugging Lineup Selection Issues")
    print("=" * 50)
    
    # Load the backtest detailed data if it exists
    try:
        backtest_df = pd.read_csv("data/backtest_detailed.csv")
        print(f"✅ Loaded {len(backtest_df)} matched players")
    except:
        print("❌ No backtest_detailed.csv found. Run backtest first.")
        return
    
    # Check data quality
    print(f"\n📊 Data Overview:")
    print(f"Players with proj_fpts: {backtest_df['proj_fpts'].notna().sum()}")
    print(f"Players with actual_fpts: {backtest_df['actual_fpts'].notna().sum()}")
    print(f"Players with salary_final: {backtest_df['salary_final'].notna().sum()}")
    
    # Show position breakdown
    print(f"\n📋 Position Availability:")
    pos_counts = backtest_df['pos'].value_counts()
    print(pos_counts)
    
    # Show top players by position
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_data = backtest_df[backtest_df['pos'] == pos].copy()
        if len(pos_data) > 0:
            pos_data['value'] = pos_data['proj_fpts'] / (pos_data['salary_final'] / 1000)
            top_players = pos_data.nlargest(3, 'value')[['name', 'proj_fpts', 'actual_fpts', 'salary_final', 'value']]
            print(f"\n🏆 Top 3 {pos} by Value:")
            print(top_players.to_string(index=False))
    
    # Manually simulate optimal lineup
    print(f"\n🎯 Manual Optimal Lineup Simulation:")
    
    positions_needed = {
        'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1
    }
    
    lineup_players = []
    total_salary = 0
    total_projected = 0
    total_actual = 0
    
    # Calculate value for all players
    backtest_df['value'] = backtest_df['proj_fpts'] / (backtest_df['salary_final'] / 1000)
    
    # Select best players by position
    for pos, count in positions_needed.items():
        pos_data = backtest_df[backtest_df['pos'] == pos].copy()
        
        if len(pos_data) >= count:
            # Sort by value (points per $1000)
            pos_data = pos_data.sort_values('value', ascending=False)
            selected = pos_data.head(count)
            
            print(f"\n{pos} selections:")
            print(selected[['name', 'proj_fpts', 'actual_fpts', 'salary_final', 'value']].to_string(index=False))
            
            lineup_players.extend(selected.index.tolist())
            total_salary += selected['salary_final'].sum()
            total_projected += selected['proj_fpts'].sum()
            total_actual += selected['actual_fpts'].sum()
        else:
            print(f"❌ Not enough {pos} players available: {len(pos_data)}")
    
    # Add FLEX (best remaining RB/WR/TE)
    flex_candidates = backtest_df[
        backtest_df['pos'].isin(['RB', 'WR', 'TE']) & 
        ~backtest_df.index.isin(lineup_players)
    ].copy()
    
    if len(flex_candidates) > 0:
        flex_candidates = flex_candidates.sort_values('value', ascending=False)
        flex_player = flex_candidates.head(1)
        
        print(f"\nFLEX selection:")
        print(flex_player[['name', 'pos', 'proj_fpts', 'actual_fpts', 'salary_final', 'value']].to_string(index=False))
        
        total_salary += flex_player['salary_final'].sum()
        total_projected += flex_player['proj_fpts'].sum()
        total_actual += flex_player['actual_fpts'].sum()
    
    print(f"\n🏆 OPTIMAL LINEUP SUMMARY:")
    print(f"Total Salary: ${total_salary:,.0f}")
    print(f"Projected Score: {total_projected:.1f}")
    print(f"Actual Score: {total_actual:.1f}")
    print(f"Projection Error: {total_projected - total_actual:+.1f}")
    
    # Check if this is reasonable
    if total_actual > 130:
        print("✅ Lineup score looks competitive for DFS")
    elif total_actual > 100:
        print("⚠️ Lineup score is mediocre but playable")
    else:
        print("❌ Lineup score is too low - check data quality")
    
    # Show salary efficiency
    if total_salary > 0:
        points_per_1k = total_actual / (total_salary / 1000)
        print(f"Points per $1000 salary: {points_per_1k:.2f}")
        if points_per_1k > 2.8:
            print("✅ Good salary efficiency")
        else:
            print("⚠️ Poor salary efficiency")

if __name__ == "__main__":
    debug_lineup_selection()