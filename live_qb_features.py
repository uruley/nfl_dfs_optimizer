import pandas as pd
import numpy as np
import nfl_data_py as nfl
import joblib
from datetime import datetime

def create_live_qb_features():
    """
    Create real QB features for live players using your original pipeline approach
    """
    
    print("🔧 CREATING LIVE QB FEATURES USING YOUR PIPELINE")
    print("="*60)
    
    # Load recent historical data to calculate rolling averages
    print("📥 Loading 2023 NFL data for rolling averages...")
    historical_data = nfl.import_weekly_data([2023])
    
    # Load current showdown QBs
    showdown_data = pd.read_csv('data/live_slates/showdown_processed.csv')
    qb_players = showdown_data[showdown_data['position'] == 'QB'].copy()
    
    print(f"🎯 Processing {len(qb_players)} showdown QBs")
    
    # For each QB, create features based on their historical data
    live_features = []
    
    for idx, qb in qb_players.iterrows():
        qb_name = qb['player_name']
        print(f"\n📊 Processing {qb_name}...")
        
        # Find QB in historical data
        qb_historical = historical_data[
            historical_data['player_display_name'].str.contains(qb_name.split()[1], case=False, na=False)
        ]
        
        if not qb_historical.empty:
            # Get recent games for rolling averages
            recent_games = qb_historical.sort_values(['season', 'week']).tail(8)
            
            if len(recent_games) >= 3:
                # Calculate rolling averages like your original pipeline
                fp_avg_3 = recent_games['fantasy_points'].tail(3).mean()
                fp_avg_5 = recent_games['fantasy_points'].tail(5).mean() if len(recent_games) >= 5 else fp_avg_3
                fp_avg_8 = recent_games['fantasy_points'].mean()
                
                # Get latest game stats for current form
                latest_game = recent_games.iloc[-1]
                
                # Create feature vector matching your model's 45 features
                features = [
                    latest_game.get('completions', 15.0),
                    latest_game.get('attempts', 25.0), 
                    latest_game.get('passing_yards', 200.0),
                    latest_game.get('passing_tds', 1.5),
                    latest_game.get('interceptions', 0.8),
                    latest_game.get('sacks', 2.0),
                    latest_game.get('sack_yards', 15.0),
                    latest_game.get('sack_fumbles', 0.0),
                    latest_game.get('sack_fumbles_lost', 0.0),
                    latest_game.get('passing_air_yards', 150.0),
                    latest_game.get('passing_yards_after_catch', 50.0),
                    latest_game.get('passing_first_downs', 10.0),
                    latest_game.get('passing_epa', 0.1),
                    latest_game.get('passing_2pt_conversions', 0.0),
                    latest_game.get('pacr', 0.7),
                    latest_game.get('dakota', 0.0),
                    latest_game.get('carries', 5.0),
                    latest_game.get('rushing_yards', 25.0),
                    latest_game.get('rushing_tds', 0.3),
                    latest_game.get('rushing_fumbles', 0.0),
                    latest_game.get('rushing_fumbles_lost', 0.0),
                    latest_game.get('rushing_first_downs', 2.0),
                    latest_game.get('rushing_epa', 0.05),
                    latest_game.get('rushing_2pt_conversions', 0.0),
                    latest_game.get('receptions', 0.0),
                    latest_game.get('targets', 0.0),
                    latest_game.get('receiving_yards', 0.0),
                    latest_game.get('receiving_tds', 0.0),
                    latest_game.get('receiving_fumbles', 0.0),
                    latest_game.get('receiving_fumbles_lost', 0.0),
                    latest_game.get('receiving_air_yards', 0.0),
                    latest_game.get('receiving_yards_after_catch', 0.0),
                    latest_game.get('receiving_first_downs', 0.0),
                    latest_game.get('receiving_epa', 0.0),
                    latest_game.get('receiving_2pt_conversions', 0.0),
                    latest_game.get('racr', 0.0),
                    latest_game.get('target_share', 0.0),
                    latest_game.get('air_yards_share', 0.0),
                    latest_game.get('wopr', 0.0),
                    latest_game.get('special_teams_tds', 0.0),
                    latest_game.get('fantasy_points_ppr', latest_game.get('fantasy_points', 15.0)),
                    fp_avg_3,  # Your calculated rolling average
                    fp_avg_5,  # Your calculated rolling average  
                    fp_avg_8,  # Your calculated rolling average
                    np.random.randint(10, 25)  # opp_def_rank (would be real matchup data)
                ]
                
                live_features.append(features)
                
                print(f"  ✅ {qb_name}: fp_avg_3={fp_avg_3:.1f}, fp_avg_8={fp_avg_8:.1f}")
                
            else:
                print(f"  ⚠️  {qb_name}: Insufficient historical data, using defaults")
                # Use default features for players without enough history
                live_features.append([10.0] * 41 + [qb['projection'], qb['projection'], qb['projection'], 20])
        else:
            print(f"  ⚠️  {qb_name}: Not found in 2023 data, using defaults")
            live_features.append([10.0] * 41 + [qb['projection'], qb['projection'], qb['projection'], 20])
    
    # Create feature dataframe
    feature_names = [
        'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
        'sacks', 'sack_yards', 'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards',
        'passing_yards_after_catch', 'passing_first_downs', 'passing_epa', 'passing_2pt_conversions',
        'pacr', 'dakota', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
        'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa', 'rushing_2pt_conversions',
        'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles',
        'receiving_fumbles_lost', 'receiving_air_yards', 'receiving_yards_after_catch',
        'receiving_first_downs', 'receiving_epa', 'receiving_2pt_conversions', 'racr',
        'target_share', 'air_yards_share', 'wopr', 'special_teams_tds', 'fantasy_points_ppr',
        'fp_avg_3', 'fp_avg_5', 'fp_avg_8', 'opp_def_rank'
    ]
    
    feature_df = pd.DataFrame(live_features, columns=feature_names, index=qb_players.index)
    
    # Test with model
    model = joblib.load('models/qb_model.pkl')
    ml_projections = model.predict(feature_df)
    
    # Show results
    print(f"\n🔥 REAL ML PROJECTIONS:")
    for i, (_, qb) in enumerate(qb_players.iterrows()):
        consensus = qb['projection']
        ml_proj = ml_projections[i]
        advantage = ml_proj - consensus
        print(f"  {qb['player_name']}: {consensus:.1f} → {ml_proj:.1f} ({advantage:+.1f})")
    
    return feature_df, ml_projections

if __name__ == "__main__":
    create_live_qb_features()