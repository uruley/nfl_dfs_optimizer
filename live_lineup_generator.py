import pandas as pd
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from Scripts
try:
    from Scripts.lineup_optimizer import SimpleDraftKingsOptimizer
except ImportError:
    # Fallback: import directly
    sys.path.append('Scripts')
    from lineup_optimizer import SimpleDraftKingsOptimizer

def generate_lineups_from_dk_slate():
    """
    Generate lineups using the DraftKings slate data
    """
    
    print("🏈 GENERATING LINEUPS FROM DRAFTKINGS SLATE DATA")
    print("="*60)
    
    # Find the most recent DraftKings slate file
    slate_dir = 'data/live_slates'
    
    if not os.path.exists(slate_dir):
        print(f"❌ Directory {slate_dir} not found!")
        print("Run: python Scripts\\manual_salary_loader.py first")
        return
    
    slate_files = [f for f in os.listdir(slate_dir) if f.startswith('dk_slate_') and f.endswith('.csv')]
    
    if not slate_files:
        print("❌ No DraftKings slate files found!")
        print("Run: python Scripts\\manual_salary_loader.py first")
        return
    
    # Get the most recent file
    latest_slate = max(slate_files, key=lambda f: os.path.getmtime(os.path.join(slate_dir, f)))
    slate_file = os.path.join(slate_dir, latest_slate)
    
    print(f"📁 Loading slate: {latest_slate}")
    
    # Load the slate data
    player_pool = pd.read_csv(slate_file)
    print(f"✅ Loaded {len(player_pool)} players")
    print(f"📊 Positions: {player_pool['position'].value_counts().to_dict()}")
    
    # Check if we have the required columns
    required_cols = ['player_name', 'position', 'salary', 'projection']
    missing_cols = [col for col in required_cols if col not in player_pool.columns]
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        print(f"Available columns: {list(player_pool.columns)}")
        return
    
    # Show salary ranges
    print(f"\n💰 Salary Ranges:")
    for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
        pos_players = player_pool[player_pool['position'] == pos]
        if not pos_players.empty:
            min_sal = pos_players['salary'].min()
            max_sal = pos_players['salary'].max()
            print(f"   {pos}: ${min_sal:,} - ${max_sal:,}")
    
    # Initialize optimizer
    optimizer = SimpleDraftKingsOptimizer()
    
    # Generate single lineup
    print(f"\n🎯 Generating optimal lineup...")
    lineup = optimizer.optimize_lineup(player_pool)
    
    if lineup.empty:
        print(f"❌ Lineup generation failed")
        return
    
    # Display the lineup
    print(f"\n📋 OPTIMAL LINEUP FROM DRAFTKINGS DATA:")
    lineup_display = lineup[['player_name', 'position', 'salary', 'projection']].sort_values('position')
    print(lineup_display.to_string(index=False))
    
    total_salary = lineup['salary'].sum()
    total_projection = lineup['projection'].sum()
    
    print(f"\n💰 Total Salary: ${total_salary:,}")
    print(f"🎯 Projected Points: {total_projection:.1f}")
    print(f"💸 Salary Remaining: ${50000 - total_salary:,}")
    
    # Export to DraftKings format
    dk_lineup = format_for_draftkings(lineup)
    
    # Save lineup
    os.makedirs('data/exports', exist_ok=True)
    export_file = 'data/exports/dk_slate_lineup.csv'
    dk_lineup.to_csv(export_file, index=False)
    
    print(f"\n🏆 SUCCESS!")
    print(f"📁 Lineup exported: {export_file}")
    print(f"📤 Ready to upload to DraftKings!")
    
    # Show the DraftKings format
    print(f"\n📋 DRAFTKINGS CSV FORMAT:")
    print(dk_lineup.to_string(index=False))
    
    return export_file

def format_for_draftkings(lineup):
    """
    Convert lineup to DraftKings upload format
    """
    
    # Get players by position
    qb = lineup[lineup['position'] == 'QB'].iloc[0]
    rbs = lineup[lineup['position'] == 'RB']
    wrs = lineup[lineup['position'] == 'WR'] 
    te = lineup[lineup['position'] == 'TE'].iloc[0]
    dst = lineup[lineup['position'] == 'DST'].iloc[0]
    
    # Handle FLEX (remaining RB/WR/TE after required positions)
    used_players = set([qb['player_name']] + 
                      rbs['player_name'].tolist()[:2] +  # First 2 RBs
                      wrs['player_name'].tolist()[:3] +  # First 3 WRs
                      [te['player_name'], dst['player_name']])
    
    flex_candidates = lineup[~lineup['player_name'].isin(used_players)]
    flex = flex_candidates.iloc[0] if not flex_candidates.empty else rbs.iloc[-1]
    
    dk_format = pd.DataFrame({
        'Entry ID': ['Entry_1'],
        'Contest Name': ['NFL Slate Contest'],
        'QB': [qb['player_name']],
        'RB1': [rbs.iloc[0]['player_name']],
        'RB2': [rbs.iloc[1]['player_name'] if len(rbs) > 1 else rbs.iloc[0]['player_name']],
        'WR1': [wrs.iloc[0]['player_name']],
        'WR2': [wrs.iloc[1]['player_name'] if len(wrs) > 1 else wrs.iloc[0]['player_name']],
        'WR3': [wrs.iloc[2]['player_name'] if len(wrs) > 2 else wrs.iloc[0]['player_name']],
        'TE': [te['player_name']],
        'FLEX': [flex['player_name']],
        'DST': [dst['player_name']],
        'Total Salary': [lineup['salary'].sum()],
        'Projected Points': [lineup['projection'].sum()]
    })
    
    return dk_format

if __name__ == "__main__":
    generate_lineups_from_dk_slate()