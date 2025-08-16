import pandas as pd
import numpy as np
import nfl_data_py as nfl
import joblib
from datetime import datetime, timedelta
import os

class ProperLivePredictions:
    """
    Generate live predictions using REAL historical data for each player
    No more simulated features!
    """
    
    def __init__(self):
        self.models = {}
        self.feature_columns = {}
        
    def load_models(self):
        """Load properly trained models"""
        
        positions = ['QB']  # Expand as you add more models
        
        for position in positions:
            try:
                model_path = f'models/{position.lower()}_model_proper.pkl'
                model_data = joblib.load(model_path)
                
                self.models[position] = model_data['model']
                self.feature_columns[position] = model_data['feature_columns']
                
                print(f"✅ Loaded {position} model with {len(model_data['feature_columns'])} features")
                
            except Exception as e:
                print(f"❌ Error loading {position} model: {e}")
    
    def get_real_player_features(self, player_name, position, current_date=None):
        """
        Get REAL historical features for a specific player
        Uses actual NFL data, not simulations!
        """
        
        if current_date is None:
            current_date = datetime.now()
        
        print(f"🔍 Getting real features for {player_name} ({position})...")
        
        try:
            # Load recent historical data to get player's past performance
            recent_seasons = [2022, 2023, 2024]  # Adjust based on current year
            historical_data = nfl.import_weekly_data(recent_seasons)
            
            # Find this player's games
            player_games = historical_data[
                (historical_data['player_display_name'].str.contains(player_name.split()[-1], case=False, na=False)) &
                (historical_data['position'] == position)
            ].copy()
            
            if player_games.empty:
                print(f"  ⚠️ No historical data found for {player_name}")
                return self.get_default_features(position)
            
            # Sort by time and get recent games
            player_games = player_games.sort_values(['season', 'week'])
            recent_games = player_games.tail(8)  # Last 8 games
            
            print(f"  ✅ Found {len(player_games)} total games, using {len(recent_games)} recent games")
            
            # Create features from real historical data
            features = self.extract_real_features(recent_games, position)
            
            return features
            
        except Exception as e:
            print(f"  ❌ Error getting real features: {e}")
            return self.get_default_features(position)
    
    def extract_real_features(self, games, position):
        """Extract features from real game data"""
        
        if len(games) < 3:
            return self.get_default_features(position)
        
        try:
            # Core fantasy points features
            features = {
                'fp_avg_3': games['fantasy_points'].tail(3).mean(),
                'fp_avg_5': games['fantasy_points'].tail(5).mean() if len(games) >= 5 else games['fantasy_points'].mean(),
                'fp_avg_8': games['fantasy_points'].mean(),
                'fp_trend': self.calculate_trend(games['fantasy_points']),
                'fp_consistency': games['fantasy_points'].std(),
                'games_played': len(games),
                'recent_games_played': min(len(games), 3),
                'season': games['season'].iloc[-1],
                'week': games['week'].iloc[-1] + 1,  # Next week
                'is_late_season': 1 if games['week'].iloc[-1] > 12 else 0,
            }
            
            # Position-specific real features
            if position == 'QB':
                for stat in ['passing_yards', 'passing_tds', 'interceptions', 'rushing_yards', 'rushing_tds']:
                    if stat in games.columns:
                        features[f'{stat}_avg_3'] = games[stat].tail(3).mean()
                        features[f'{stat}_avg_5'] = games[stat].tail(5).mean() if len(games) >= 5 else games[stat].mean()
                    else:
                        features[f'{stat}_avg_3'] = 0.0
                        features[f'{stat}_avg_5'] = 0.0
            
            elif position == 'RB':
                for stat in ['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']:
                    if stat in games.columns:
                        features[f'{stat}_avg_3'] = games[stat].tail(3).mean()
                        features[f'{stat}_avg_5'] = games[stat].tail(5).mean() if len(games) >= 5 else games[stat].mean()
                    else:
                        features[f'{stat}_avg_3'] = 0.0
                        features[f'{stat}_avg_5'] = 0.0
            
            elif position in ['WR', 'TE']:
                for stat in ['receptions', 'receiving_yards', 'receiving_tds', 'targets']:
                    if stat in games.columns:
                        features[f'{stat}_avg_3'] = games[stat].tail(3).mean()
                        features[f'{stat}_avg_5'] = games[stat].tail(5).mean() if len(games) >= 5 else games[stat].mean()
                    else:
                        features[f'{stat}_avg_3'] = 0.0
                        features[f'{stat}_avg_5'] = 0.0
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return self.get_default_features(position)
    
    def calculate_trend(self, values):
        """Calculate performance trend"""
        if len(values) < 6:
            return 0
        
        recent = values.tail(3).mean()
        previous = values.tail(6).head(3).mean()
        
        return recent - previous
    
    def get_default_features(self, position):
        """Default features when no historical data available"""
        
        defaults = {
            'fp_avg_3': 15.0,
            'fp_avg_5': 15.0,
            'fp_avg_8': 15.0,
            'fp_trend': 0.0,
            'fp_consistency': 5.0,
            'games_played': 8,
            'recent_games_played': 3,
            'season': 2024,
            'week': 1,
            'is_late_season': 0,
        }
        
        # Add position-specific defaults
        if position == 'QB':
            defaults.update({
                'passing_yards_avg_3': 250.0,
                'passing_yards_avg_5': 250.0,
                'passing_tds_avg_3': 1.5,
                'passing_tds_avg_5': 1.5,
                'interceptions_avg_3': 0.8,
                'interceptions_avg_5': 0.8,
                'rushing_yards_avg_3': 20.0,
                'rushing_yards_avg_5': 20.0,
                'rushing_tds_avg_3': 0.3,
                'rushing_tds_avg_5': 0.3,
            })
        
        return defaults
    
    def generate_live_predictions(self, players_df):
        """Generate predictions for current players using real historical data"""
        
        print(f"\n🤖 GENERATING REAL ML PREDICTIONS")
        print("="*50)
        
        predictions = {}
        
        for position in self.models.keys():
            position_players = players_df[players_df['position'] == position].copy()
            
            if position_players.empty:
                continue
            
            print(f"\n📊 Processing {len(position_players)} {position}s...")
            
            # Get real features for each player
            feature_rows = []
            
            for idx, player in position_players.iterrows():
                player_name = player['player_name']
                
                # Get REAL historical features
                real_features = self.get_real_player_features(player_name, position)
                
                # Ensure all model features are present
                feature_row = {}
                for feature_col in self.feature_columns[position]:
                    feature_row[feature_col] = real_features.get(feature_col, 0.0)
                
                feature_rows.append(feature_row)
            
            # Convert to DataFrame for prediction
            features_df = pd.DataFrame(feature_rows, index=position_players.index)
            
            # Generate ML predictions
            ml_predictions = self.models[position].predict(features_df)
            
            # Store results
            predictions[position] = {
                'indices': position_players.index,
                'predictions': ml_predictions,
                'consensus': position_players['projection'].values
            }
            
            # Show comparison
            print(f"\n🔥 {position} ML vs Consensus:")
            for i, idx in enumerate(position_players.index):
                player_name = position_players.loc[idx, 'player_name']
                consensus = position_players.loc[idx, 'projection']
                ml_pred = ml_predictions[i]
                advantage = ml_pred - consensus
                
                print(f"  {player_name}: {consensus:.1f} → {ml_pred:.1f} ({advantage:+.1f})")
        
        return predictions
    
    def update_player_projections(self, players_df, predictions):
        """Update player dataframe with ML projections"""
        
        updated_df = players_df.copy()
        
        for position, pred_data in predictions.items():
            # Store consensus projections
            for idx in pred_data['indices']:
                updated_df.loc[idx, 'consensus_projection'] = updated_df.loc[idx, 'projection']
            
            # Update with ML projections
            for i, idx in enumerate(pred_data['indices']):
                updated_df.loc[idx, 'projection'] = pred_data['predictions'][i]
                updated_df.loc[idx, 'ml_projection'] = pred_data['predictions'][i]
        
        return updated_df

def test_proper_predictions():
    """Test the proper prediction system with showdown data"""
    
    print("🧪 TESTING PROPER ML PREDICTIONS")
    print("="*50)
    
    # Load slate data from most recent file in live_slates directory
    try:
        # List all DK slate files
        slate_dir = 'data/live_slates'
        slate_files = [f for f in os.listdir(slate_dir) if f.startswith('dk_slate')]
        
        if not slate_files:
            print("❌ No slate files found in data/live_slates/")
            return
            
        # Get most recent slate file
        latest_slate = sorted(slate_files)[-1]
        slate_path = os.path.join(slate_dir, latest_slate)
        
        print(f"📊 Loading slate: {latest_slate}")
        slate_data = pd.read_csv(slate_path)
        
    except Exception as e:
        print(f"❌ Error loading slate data: {e}")
        return
    
    # Initialize predictor
    predictor = ProperLivePredictions()
    predictor.load_models()
    
    if not predictor.models:
        print("❌ No models loaded! Run proper_ml_training.py first")
        return
    
    # Generate real ML predictions
    predictions = predictor.generate_live_predictions(slate_data)
    
    # Update player projections
    updated_players = predictor.update_player_projections(slate_data, predictions)
    
    # Show results
    print(f"\n📊 UPDATED PROJECTIONS:")
    qb_players = updated_players[updated_players['position'] == 'QB']
    
    if not qb_players.empty:
        display_cols = ['player_name', 'consensus_projection', 'ml_projection']
        print(qb_players[display_cols].to_string(index=False))
    
    return updated_players

if __name__ == "__main__":
    test_proper_predictions()