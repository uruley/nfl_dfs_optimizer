import pandas as pd
import sys
sys.path.append('Scripts')
from showdown_optimizer import ShowdownOptimizer

def generate_showdown_lineups():
    """
    Generate showdown lineups using the processed data
    """
    
    print("🏈 GENERATING SHOWDOWN LINEUPS")
    print("="*50)
    
    # Load processed showdown data
    showdown_data = pd.read_csv('data/live_slates/showdown_processed.csv')
    print(f"✅ Loaded {len(showdown_data)} players")
    
    # Rename columns for optimizer compatibility
    showdown_data['salary'] = showdown_data['flex_salary']  # Use FLEX salary as base
    
    # Generate lineup
    optimizer = ShowdownOptimizer(salary_cap=50000)
    lineup = optimizer.optimize_showdown(showdown_data)
    
    if not lineup.empty:
        print(f"\n📋 SHOWDOWN LINEUP:")
        display_cols = ['player_name', 'role', 'position', 'team', 'actual_salary', 'fantasy_points']
        print(lineup[display_cols].to_string(index=False))
        
        total_salary = lineup['actual_salary'].sum()
        total_points = lineup['fantasy_points'].sum()
        
        print(f"\n💰 Total Salary: ${total_salary:,.0f}")
        print(f"🎯 Total Points: {total_points:.1f}")
        print(f"💸 Remaining: ${50000 - total_salary:,.0f}")
        
        # Format for DraftKings upload
        dk_format = optimizer.format_showdown_for_dk(lineup)
        
        # Save lineup
        import os
        os.makedirs('data/exports', exist_ok=True)
        export_file = 'data/exports/showdown_lineup.csv'
        dk_format.to_csv(export_file, index=False)
        
        print(f"\n🏆 SUCCESS!")
        print(f"📁 Exported: {export_file}")
        print(f"📤 Ready for DraftKings upload!")
        
    else:
        print("❌ Showdown optimization failed")

if __name__ == "__main__":
    generate_showdown_lineups()