import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Scripts.lineup_optimizer import SimpleDraftKingsOptimizer

def generate_week1_lineups():
    """Generate Week 1 lineups using your player pool"""
    
    print("🚀 GENERATING WEEK 1 LINEUPS WITH YOUR DATA!")
    print("="*60)
    
    # Load the player pool you just created
    player_pool_file = 'data/processed/week1_player_pool_with_qb_model.csv'
    
    if not os.path.exists(player_pool_file):
        print(f"❌ {player_pool_file} not found!")
        return
    
    player_pool = pd.read_csv(player_pool_file)
    print(f"✅ Loaded {len(player_pool)} players")
    
    # Initialize optimizer
    optimizer = SimpleDraftKingsOptimizer()
    
    # Generate single lineup first
    print(f"\n🎯 Generating optimal lineup...")
    lineup = optimizer.optimize_lineup(player_pool)
    
    if lineup.empty:
        print(f"❌ Lineup generation failed")
        return
    
    print(f"\n📋 YOUR WEEK 1 LINEUP:")
    lineup_display = lineup[['player_name', 'position', 'salary', 'projection']].sort_values('position')
    print(lineup_display.to_string(index=False))
    
    # Create exports folder if it doesn't exist
    os.makedirs('data/exports', exist_ok=True)
    
    # Save lineup in DraftKings format
    dk_lineup = pd.DataFrame({
        'Entry ID': ['Entry_1'],
        'Contest Name': ['NFL Week 1'],
        'QB': [lineup[lineup['position'] == 'QB']['player_name'].iloc[0]],
        'RB1': [lineup[lineup['position'] == 'RB']['player_name'].iloc[0]],
        'RB2': [lineup[lineup['position'] == 'RB']['player_name'].iloc[1] if len(lineup[lineup['position'] == 'RB']) > 1 else lineup[lineup['position'] == 'RB']['player_name'].iloc[0]],
        'WR1': [lineup[lineup['position'] == 'WR']['player_name'].iloc[0]],
        'WR2': [lineup[lineup['position'] == 'WR']['player_name'].iloc[1]],
        'WR3': [lineup[lineup['position'] == 'WR']['player_name'].iloc[2] if len(lineup[lineup['position'] == 'WR']) > 2 else lineup[lineup['position'] == 'WR']['player_name'].iloc[0]],
        'TE': [lineup[lineup['position'] == 'TE']['player_name'].iloc[0]],
        'FLEX': [lineup[lineup['position'].isin(['RB', 'WR', 'TE'])]['player_name'].iloc[-1]],  # Last flex eligible player
        'DST': [lineup[lineup['position'] == 'DST']['player_name'].iloc[0]],
        'Total Salary': [lineup['salary'].sum()],
        'Projected Points': [lineup['projection'].sum()]
    })
    
    export_file = 'data/exports/week1_single_lineup.csv'
    dk_lineup.to_csv(export_file, index=False)
    
    print(f"\n🏆 SUCCESS!")
    print(f"📁 Lineup exported to: {export_file}")
    print(f"💰 Total Salary: ${lineup['salary'].sum():,}")
    print(f"🎯 Projected Points: {lineup['projection'].sum():.1f}")
    print(f"📤 Ready to upload to DraftKings!")

# Sample salary data for testing
salary_data = {
    'salary': [
        # RBs
        7200, 6600, 6000, 5400, 5800, 4800, 4400, 4000,
        # WRs
        6200, 5800, 5600, 5200, 4800, 5400, 5000, 4600, 4400, 5600,
        # TEs
        5400, 4600, 4000, 3600, 4200,
        # DSTs
        2400, 2200, 2000, 1800
    ]
}

if __name__ == "__main__":
    generate_week1_lineups()