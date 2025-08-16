import pandas as pd
import os
import sys

# Add Scripts directory to path
sys.path.append('Scripts')

# Import the optimizer (using the correct filename)
from lineup_optimizer import SimpleDraftKingsOptimizer

def main():
    print("🏈 GENERATING LINEUPS FROM DRAFTKINGS DATA")
    print("="*50)
    
    # Find latest slate file
    slate_dir = 'data/live_slates'
    
    if not os.path.exists(slate_dir):
        print(f"❌ Directory {slate_dir} not found!")
        print("Run: python Scripts\\manual_salary_loader.py first")
        return
    
    slate_files = [f for f in os.listdir(slate_dir) if f.endswith('.csv')]
    
    if not slate_files:
        print("❌ No slate files found!")
        print("Run: python Scripts\\manual_salary_loader.py first")
        return
    
    latest_slate = max(slate_files)
    slate_path = f'{slate_dir}/{latest_slate}'
    
    print(f"📁 Loading: {latest_slate}")
    
    # Load data
    player_pool = pd.read_csv(slate_path)
    print(f"✅ Loaded {len(player_pool)} players")
    print(f"📊 Positions: {player_pool['position'].value_counts().to_dict()}")
    
    # Show salary ranges
    print(f"\n💰 Salary Ranges:")
    for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
        pos_players = player_pool[player_pool['position'] == pos]
        if not pos_players.empty:
            min_sal = pos_players['salary'].min()
            max_sal = pos_players['salary'].max()
            print(f"   {pos}: ${min_sal:,} - ${max_sal:,}")
    
    # Generate lineup
    print(f"\n🎯 Generating optimal lineup...")
    optimizer = SimpleDraftKingsOptimizer()
    lineup = optimizer.optimize_lineup(player_pool)
    
    if not lineup.empty:
        print(f"\n📋 OPTIMAL LINEUP FROM DRAFTKINGS DATA:")
        display = lineup[['player_name', 'position', 'salary', 'projection']].sort_values('position')
        print(display.to_string(index=False))
        
        total_salary = lineup['salary'].sum()
        total_points = lineup['projection'].sum()
        
        print(f"\n💰 Total Salary: ${total_salary:,}")
        print(f"🎯 Projected Points: {total_points:.1f}")
        print(f"💸 Remaining: ${50000 - total_salary:,}")
        
        print(f"\n🏆 SUCCESS! This lineup is ready for DraftKings upload!")
    else:
        print("❌ Lineup generation failed")

if __name__ == "__main__":
    main()