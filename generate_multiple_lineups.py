import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Scripts.lineup_optimizer import SimpleDraftKingsOptimizer

def generate_multiple_lineups(num_lineups=10):
    """Generate multiple diverse lineups for tournaments"""
    
    print(f"🚀 GENERATING {num_lineups} DIVERSE LINEUPS FOR TOURNAMENTS!")
    print("="*70)
    
    # Load player pool
    player_pool = pd.read_csv('data/processed/week1_player_pool_with_qb_model.csv')
    print(f"✅ Loaded {len(player_pool)} players")
    
    optimizer = SimpleDraftKingsOptimizer()
    lineups = []
    
    for i in range(num_lineups):
        print(f"\n🎯 Generating lineup {i+1}/{num_lineups}...")
        
        # Add some randomness to create diversity
        player_pool_copy = player_pool.copy()
        
        # Slightly randomize projections to create different optimal lineups
        noise = np.random.normal(0, 0.5, len(player_pool_copy))  # Small random noise
        player_pool_copy['projection'] = player_pool_copy['projection'] + noise
        
        lineup = optimizer.optimize_lineup(player_pool_copy)
        
        if not lineup.empty:
            lineups.append(lineup)
            qb_name = lineup[lineup['position'] == 'QB']['player_name'].iloc[0]
            total_proj = lineup['projection'].sum()
            print(f"   ✅ Lineup {i+1}: {qb_name}, {total_proj:.1f} pts")
        else:
            print(f"   ❌ Lineup {i+1} failed")
    
    if not lineups:
        print("❌ No lineups generated!")
        return
    
    print(f"\n📊 GENERATED {len(lineups)} LINEUPS!")
    
    # Export all lineups to DraftKings format
    export_data = []
    
    for i, lineup in enumerate(lineups):
        try:
            qb = lineup[lineup['position'] == 'QB'].iloc[0]
            rbs = lineup[lineup['position'] == 'RB']
            wrs = lineup[lineup['position'] == 'WR'] 
            te = lineup[lineup['position'] == 'TE'].iloc[0]
            dst = lineup[lineup['position'] == 'DST'].iloc[0]
            
            # Handle FLEX (get remaining RB/WR/TE not already used)
            used_names = set([qb['player_name']] + 
                            rbs['player_name'].tolist()[:2] +  # First 2 RBs
                            wrs['player_name'].tolist()[:3] +  # First 3 WRs
                            [te['player_name'], dst['player_name']])
            
            flex_candidates = lineup[~lineup['player_name'].isin(used_names)]
            flex = flex_candidates.iloc[0] if not flex_candidates.empty else rbs.iloc[-1]
            
            row = {
                'Entry ID': f"Entry_{i+1}",
                'Contest Name': f"NFL Week 1 GPP",
                'QB': qb['player_name'],
                'RB1': rbs.iloc[0]['player_name'],
                'RB2': rbs.iloc[1]['player_name'] if len(rbs) > 1 else rbs.iloc[0]['player_name'],
                'WR1': wrs.iloc[0]['player_name'],
                'WR2': wrs.iloc[1]['player_name'] if len(wrs) > 1 else wrs.iloc[0]['player_name'],
                'WR3': wrs.iloc[2]['player_name'] if len(wrs) > 2 else wrs.iloc[0]['player_name'],
                'TE': te['player_name'],
                'FLEX': flex['player_name'],
                'DST': dst['player_name'],
                'Total Salary': lineup['salary'].sum(),
                'Projected Points': lineup['projection'].sum()
            }
            
            export_data.append(row)
            
        except Exception as e:
            print(f"❌ Error exporting lineup {i+1}: {e}")
    
    if export_data:
        # Save multi-entry file
        export_df = pd.DataFrame(export_data)
        export_file = f'data/exports/week1_tournament_lineups_{len(export_data)}_entries.csv'
        export_df.to_csv(export_file, index=False)
        
        print(f"\n🏆 TOURNAMENT LINEUPS READY!")
        print(f"📁 File: {export_file}")
        print(f"💰 Salary range: ${export_df['Total Salary'].min():,} - ${export_df['Total Salary'].max():,}")
        print(f"🎯 Projection range: {export_df['Projected Points'].min():.1f} - {export_df['Projected Points'].max():.1f}")
        print(f"📤 Ready to upload {len(export_data)} lineups to DraftKings!")
        
        # Show lineup diversity
        qb_usage = export_df['QB'].value_counts()
        print(f"\n📈 QB Usage:")
        print(qb_usage.head().to_string())
        
        return export_file
    else:
        print("❌ No lineups exported!")

if __name__ == "__main__":
    generate_multiple_lineups(15)  # Generate 15 lineups