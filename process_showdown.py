import pandas as pd
import os

def process_showdown_csv():
    """
    Process the DraftKings Showdown CSV into showdown-ready format
    """
    
    print("🏈 PROCESSING SHOWDOWN CSV")
    print("="*50)
    
    # Load the raw CSV
    raw_data = pd.read_csv('data/raw/DKSalaries.csv')
    print(f"✅ Loaded {len(raw_data)} entries")
    
    # Show what we have
    cpt_count = len(raw_data[raw_data['Roster Position'] == 'CPT'])
    flex_count = len(raw_data[raw_data['Roster Position'] == 'FLEX'])
    
    print(f"📊 CPT entries: {cpt_count}")
    print(f"📊 FLEX entries: {flex_count}")
    print(f"📊 Game: {raw_data['Game Info'].iloc[0]}")
    
    # Create showdown-ready format
    showdown_players = []
    
    # Get unique players (by name)
    unique_players = raw_data['Name'].unique()
    
    for player_name in unique_players:
        player_entries = raw_data[raw_data['Name'] == player_name]
        
        # Get CPT and FLEX versions
        cpt_entry = player_entries[player_entries['Roster Position'] == 'CPT']
        flex_entry = player_entries[player_entries['Roster Position'] == 'FLEX']
        
        if not cpt_entry.empty and not flex_entry.empty:
            # Use the FLEX entry as base (lower salary for projections)
            base_entry = flex_entry.iloc[0]
            cpt_salary = cpt_entry.iloc[0]['Salary']
            
            showdown_players.append({
                'player_name': base_entry['Name'],
                'position': base_entry['Position'],
                'team': base_entry['TeamAbbrev'],
                'flex_salary': base_entry['Salary'],
                'cpt_salary': cpt_salary,
                'projection': base_entry['AvgPointsPerGame'],
                'game_info': base_entry['Game Info']
            })
    
    showdown_df = pd.DataFrame(showdown_players)
    
    print(f"\n📊 Processed {len(showdown_df)} unique players")
    print(f"📊 Positions: {showdown_df['position'].value_counts().to_dict()}")
    
    # Show salary ranges
    print(f"\n💰 Salary Ranges:")
    print(f"   FLEX: ${showdown_df['flex_salary'].min():,} - ${showdown_df['flex_salary'].max():,}")
    print(f"   CPT: ${showdown_df['cpt_salary'].min():,} - ${showdown_df['cpt_salary'].max():,}")
    
    # Save processed data
    os.makedirs('data/live_slates', exist_ok=True)
    output_file = 'data/live_slates/showdown_processed.csv'
    showdown_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved processed showdown data: {output_file}")
    
    # Show top players
    print(f"\n🏆 Top 5 Players by Projection:")
    top_players = showdown_df.nlargest(5, 'projection')[['player_name', 'position', 'cpt_salary', 'flex_salary', 'projection']]
    print(top_players.to_string(index=False))
    
    return output_file

if __name__ == "__main__":
    process_showdown_csv()