# Scripts/fix_backtest_issues.py

import pandas as pd
import numpy as np

def add_missing_dst_players():
    """
    Add DST players to your projection data for complete lineups
    """
    
    print("🛠️ Fixing Missing DST Players")
    print("=" * 40)
    
    # Load your merged salary data
    df = pd.read_csv("data/salaries_merged.csv")
    print(f"Original players: {len(df)}")
    print(f"Positions: {df['pos'].value_counts().to_dict()}")
    
    # Check if DST players exist
    dst_players = df[df['pos'] == 'DST']
    if len(dst_players) > 0:
        print(f"✅ Found {len(dst_players)} DST players")
        return
    
    # Add synthetic DST players for backtesting
    print("⚠️ No DST players found. Adding synthetic ones...")
    
    dst_teams = [
        'BAL', 'SF', 'DAL', 'BUF', 'MIA', 'PIT', 'CLE', 'NYJ',
        'NE', 'DEN', 'KC', 'LV', 'LAC', 'HOU', 'IND', 'TEN',
        'JAX', 'PHI', 'NYG', 'WAS', 'CHI', 'DET', 'GB', 'MIN',
        'NO', 'TB', 'ATL', 'CAR', 'LAR', 'ARI', 'SEA'
    ]
    
    dst_data = []
    for i, team in enumerate(dst_teams[:12]):  # Add 12 DST options
        dst_data.append({
            'season': 2024,
            'week': 15,
            'name': f'{team} DST',
            'team': team,
            'opp': 'OPP',
            'pos': 'DST',
            'proj_fpts': np.random.normal(8.0, 3.0),  # Average ~8 points, std 3
            'salary': int(np.random.normal(2400, 300)),  # Average $2400, std $300
            'proj_salary': int(np.random.normal(2400, 300)),
            'salary_final': int(np.random.normal(2400, 300)),
            'name_clean': f'{team.lower()} dst'
        })
    
    # Add to dataframe
    dst_df = pd.DataFrame(dst_data)
    combined_df = pd.concat([df, dst_df], ignore_index=True)
    
    # Save updated data
    combined_df.to_csv("data/salaries_merged_with_dst.csv", index=False)
    print(f"✅ Added {len(dst_df)} DST players")
    print(f"New total: {len(combined_df)} players")
    
    return combined_df

def check_projection_calibration():
    """
    Analyze projection vs actual performance to improve calibration
    """
    
    print("\n📊 Analyzing Projection Calibration")
    print("=" * 40)
    
    try:
        backtest_df = pd.read_csv("data/backtest_detailed.csv")
    except:
        print("❌ No backtest_detailed.csv found")
        return
    
    # Calculate bias by position
    bias_by_pos = backtest_df.groupby('pos').agg({
        'proj_fpts': 'mean',
        'actual_fpts': 'mean',
        'fpts_error': 'mean'
    }).round(2)
    
    bias_by_pos['bias_pct'] = (bias_by_pos['fpts_error'] / bias_by_pos['actual_fpts'] * 100).round(1)
    
    print("Position Bias Analysis:")
    print(bias_by_pos)
    
    # Show biggest misses
    print(f"\n🎯 Biggest Projection Misses:")
    biggest_misses = backtest_df.nlargest(10, 'abs_error')[
        ['name', 'pos', 'proj_fpts', 'actual_fpts', 'fpts_error']
    ]
    print(biggest_misses.to_string(index=False))
    
    # Suggest calibration adjustments
    overall_bias = backtest_df['fpts_error'].mean()
    print(f"\n🔧 Calibration Suggestions:")
    print(f"Overall bias: {overall_bias:+.2f} points")
    
    if abs(overall_bias) > 1.0:
        adjustment = 1 - (overall_bias / backtest_df['proj_fpts'].mean())
        print(f"Suggested multiplier: {adjustment:.3f}")
        print(f"Example: proj_fpts * {adjustment:.3f}")

def create_improved_backtest():
    """
    Run backtest with DST players and better lineup selection
    """
    
    print("\n🏈 Running Improved Backtest")
    print("=" * 40)
    
    # Load data with DST
    try:
        df = pd.read_csv("data/salaries_merged_with_dst.csv")
    except:
        print("❌ Run add_missing_dst_players() first")
        return
    
    # Simple lineup optimization
    positions_needed = {
        'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1
    }
    
    # Calculate value and sort
    df['value'] = df['proj_fpts'] / (df['salary_final'] / 1000)
    
    lineup_players = []
    total_salary = 0
    total_projected = 0
    
    # Select best by position
    for pos, count in positions_needed.items():
        if pos == 'FLEX':
            # FLEX from remaining RB/WR/TE
            available = df[
                df['pos'].isin(['RB', 'WR', 'TE']) & 
                ~df.index.isin(lineup_players)
            ].sort_values('value', ascending=False)
        else:
            available = df[df['pos'] == pos].sort_values('value', ascending=False)
        
        if len(available) >= count:
            selected = available.head(count)
            lineup_players.extend(selected.index.tolist())
            total_salary += selected['salary_final'].sum()
            total_projected += selected['proj_fpts'].sum()
    
    # Show improved lineup
    if len(lineup_players) == 9:
        lineup = df.loc[lineup_players]
        actual_score = lineup['actual_fpts'].sum() if 'actual_fpts' in lineup.columns else 0
        
        print(f"🏆 IMPROVED LINEUP:")
        print(lineup[['name', 'pos', 'proj_fpts', 'salary_final']].to_string(index=False))
        print(f"\nTotal Salary: ${total_salary:,.0f}")
        print(f"Projected Score: {total_projected:.1f}")
        if actual_score > 0:
            print(f"Actual Score: {actual_score:.1f}")
            
            # Contest performance estimate
            if actual_score >= 135:
                print("✅ Would likely CASH in contests")
            elif actual_score >= 120:
                print("⚠️ Borderline cash performance")
            else:
                print("❌ Would not cash in contests")
    else:
        print(f"❌ Could only select {len(lineup_players)} players")

if __name__ == "__main__":
    # Fix the data issues
    add_missing_dst_players()
    check_projection_calibration()
    create_improved_backtest()