import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProperMLTraining:
    """
    Proper ML training that uses ONLY historical data to predict future performance
    No cheating with future data!
    """
    
    def __init__(self):
        self.models = {}
        self.training_history = []
        self.feature_columns = []
        
    def load_historical_data(self, seasons=[2020, 2021, 2022, 2023]):
        """Load and prepare historical NFL data"""
        
        print(f"📥 Loading {len(seasons)} seasons of NFL data...")
        
        # Load weekly data
        weekly_data = nfl.import_weekly_data(seasons)
        
        # Focus on skill positions with enough data
        skill_positions = ['QB', 'RB', 'WR', 'TE']
        weekly_data = weekly_data[weekly_data['position'].isin(skill_positions)].copy()
        
        # Add week identifier for proper time ordering
        weekly_data['week_id'] = weekly_data['season'] * 100 + weekly_data['week']
        
        # Sort by player and time
        weekly_data = weekly_data.sort_values(['player_id', 'week_id'])
        
        print(f"✅ Loaded {len(weekly_data):,} player-week records")
        print(f"📊 Positions: {weekly_data['position'].value_counts().to_dict()}")
        
        return weekly_data
    
    def create_historical_features(self, data, position='QB'):
        """
        Create features using ONLY past performance data
        This is the key - NO FUTURE LEAKAGE!
        """
        
        print(f"\n🔧 Creating historical features for {position}...")
        
        # Filter to position
        pos_data = data[data['position'] == position].copy()
        
        # Sort by player and time (critical for proper feature creation)
        pos_data = pos_data.sort_values(['player_id', 'week_id'])
        
        feature_rows = []
        
        # For each player-week, create features from PAST games only
        for player_id in pos_data['player_id'].unique():
            player_games = pos_data[pos_data['player_id'] == player_id].copy()
            
            # Need at least 4 games of history to create meaningful features
            if len(player_games) < 4:
                continue
                
            # For each game (starting from game 4), create features from previous games
            for i in range(3, len(player_games)):  # Start from 4th game
                current_game = player_games.iloc[i]
                past_games = player_games.iloc[:i]  # ONLY past games!
                
                # Create features from past performance
                features = self.extract_features_from_history(past_games, current_game)
                
                if features is not None:
                    # Target is the current game's fantasy points
                    features['target_fantasy_points'] = current_game['fantasy_points']
                    features['player_id'] = current_game['player_id']
                    features['player_name'] = current_game['player_display_name']
                    features['week_id'] = current_game['week_id']
                    features['season'] = current_game['season']
                    features['week'] = current_game['week']
                    
                    feature_rows.append(features)
        
        if feature_rows:
            features_df = pd.DataFrame(feature_rows)
            print(f"✅ Created {len(features_df):,} training examples for {position}")
            return features_df
        else:
            print(f"❌ No features created for {position}")
            return pd.DataFrame()
    
    def extract_features_from_history(self, past_games, current_game):
        """Extract features from a player's historical games"""
        
        if len(past_games) < 3:
            return None
            
        try:
            # Rolling averages from past games (key ML features)
            features = {
                # Fantasy points rolling averages
                'fp_avg_3': past_games['fantasy_points'].tail(3).mean(),
                'fp_avg_5': past_games['fantasy_points'].tail(5).mean() if len(past_games) >= 5 else past_games['fantasy_points'].mean(),
                'fp_avg_8': past_games['fantasy_points'].tail(8).mean() if len(past_games) >= 8 else past_games['fantasy_points'].mean(),
                
                # Recent trend (last 3 vs previous 3)
                'fp_trend': self.calculate_trend(past_games['fantasy_points']),
                
                # Consistency (standard deviation)
                'fp_consistency': past_games['fantasy_points'].std(),
                
                # Position-specific features
                **self.get_position_specific_features(past_games),
                
                # Game context features
                'games_played': len(past_games),
                'recent_games_played': min(len(past_games), 3),
                
                # Season context
                'season': current_game['season'],
                'week': current_game['week'],
                'is_late_season': 1 if current_game['week'] > 12 else 0,
            }
            
            # Add opponent strength (if available)
            if 'opponent_team' in current_game:
                features['opponent_team'] = current_game['opponent_team']
            
            return features
            
        except Exception as e:
            print(f"⚠️ Error creating features: {e}")
            return None
    
    def get_position_specific_features(self, past_games):
        """Get position-specific rolling averages"""
        
        position = past_games['position'].iloc[0]
        features = {}
        
        if position == 'QB':
            # QB-specific metrics
            for stat in ['passing_yards', 'passing_tds', 'interceptions', 'rushing_yards', 'rushing_tds']:
                if stat in past_games.columns:
                    features[f'{stat}_avg_3'] = past_games[stat].tail(3).mean()
                    features[f'{stat}_avg_5'] = past_games[stat].tail(5).mean() if len(past_games) >= 5 else past_games[stat].mean()
        
        elif position == 'RB':
            # RB-specific metrics
            for stat in ['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']:
                if stat in past_games.columns:
                    features[f'{stat}_avg_3'] = past_games[stat].tail(3).mean()
                    features[f'{stat}_avg_5'] = past_games[stat].tail(5).mean() if len(past_games) >= 5 else past_games[stat].mean()
        
        elif position in ['WR', 'TE']:
            # WR/TE-specific metrics
            for stat in ['receptions', 'receiving_yards', 'receiving_tds', 'targets']:
                if stat in past_games.columns:
                    features[f'{stat}_avg_3'] = past_games[stat].tail(3).mean()
                    features[f'{stat}_avg_5'] = past_games[stat].tail(5).mean() if len(past_games) >= 5 else past_games[stat].mean()
        
        return features
    
    def calculate_trend(self, values):
        """Calculate trend (positive = improving, negative = declining)"""
        if len(values) < 6:
            return 0
        
        recent = values.tail(3).mean()
        previous = values.tail(6).head(3).mean()
        
        return recent - previous
    
    def walk_forward_validation(self, features_df, position='QB'):
        """
        Proper ML validation - train on past, test on future
        This simulates real-world deployment
        """
        
        print(f"\n🧪 Walk-forward validation for {position}...")
        
        # Sort by time
        features_df = features_df.sort_values('week_id')
        
        # Define validation periods
        validation_results = []
        
        # Start validation from 2022 (need 2020-2021 for initial training)
        validation_seasons = [2022, 2023]
        
        for season in validation_seasons:
            print(f"\n📊 Validating {season} season...")
            
            # Train on all data BEFORE this season
            train_data = features_df[features_df['season'] < season]
            test_data = features_df[features_df['season'] == season]
            
            if len(train_data) < 100 or len(test_data) < 10:
                print(f"⚠️ Insufficient data for {season}")
                continue
            
            # Prepare features and targets
            feature_cols = [col for col in train_data.columns 
                          if col not in ['target_fantasy_points', 'player_id', 'player_name', 'week_id', 'season', 'week', 'opponent_team']]
            
            X_train = train_data[feature_cols].fillna(0)
            y_train = train_data['target_fantasy_points']
            X_test = test_data[feature_cols].fillna(0)
            y_test = test_data['target_fantasy_points']
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            correlation = np.corrcoef(y_test, predictions)[0, 1]
            
            validation_results.append({
                'season': season,
                'position': position,
                'mse': mse,
                'mae': mae,
                'correlation': correlation,
                'n_train': len(train_data),
                'n_test': len(test_data),
                'model': model,
                'feature_cols': feature_cols
            })
            
            print(f"  📈 Results: MAE={mae:.2f}, Correlation={correlation:.3f}")
        
        return validation_results
    
    def train_final_model(self, features_df, position='QB'):
        """Train final model on all available data"""
        
        print(f"\n🎯 Training final {position} model on all data...")
        
        # Prepare features
        feature_cols = [col for col in features_df.columns 
                       if col not in ['target_fantasy_points', 'player_id', 'player_name', 'week_id', 'season', 'week', 'opponent_team']]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['target_fantasy_points']
        
        # Train final model
        final_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        
        final_model.fit(X, y)
        
        # Save model and feature columns
        model_data = {
            'model': final_model,
            'feature_columns': feature_cols,
            'position': position,
            'training_samples': len(features_df),
            'training_date': datetime.now().isoformat()
        }
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        model_path = f'models/{position.lower()}_model_proper.pkl'
        joblib.dump(model_data, model_path)
        
        print(f"✅ {position} model saved to {model_path}")
        print(f"📊 Features: {len(feature_cols)}")
        print(f"📊 Training samples: {len(features_df):,}")
        
        return final_model, feature_cols
    
    def generate_feature_importance_report(self, model, feature_cols, position):
        """Show which features matter most"""
        
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 10 Features for {position}:")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df

def main():
    """Run the complete proper ML training pipeline"""
    
    print("🚀 PROPER ML TRAINING PIPELINE")
    print("="*60)
    
    trainer = ProperMLTraining()
    
    # Load historical data
    historical_data = trainer.load_historical_data([2020, 2021, 2022, 2023])
    
    # Train models for each position
    positions = ['QB']  # Start with QB, expand to ['QB', 'RB', 'WR', 'TE']
    
    for position in positions:
        print(f"\n{'='*20} {position} MODEL {'='*20}")
        
        # Create historical features (NO FUTURE LEAKAGE!)
        features_df = trainer.create_historical_features(historical_data, position)
        
        if features_df.empty:
            print(f"❌ No data for {position}")
            continue
        
        # Validate with walk-forward approach
        validation_results = trainer.walk_forward_validation(features_df, position)
        
        # Show validation summary
        if validation_results:
            print(f"\n📊 {position} Validation Summary:")
            for result in validation_results:
                print(f"  {result['season']}: MAE={result['mae']:.2f}, Corr={result['correlation']:.3f}")
        
        # Train final model on all data
        final_model, feature_cols = trainer.train_final_model(features_df, position)
        
        # Show feature importance
        trainer.generate_feature_importance_report(final_model, feature_cols, position)
    
    print(f"\n🏆 TRAINING COMPLETE!")
    print(f"✅ Models trained using proper historical data")
    print(f"✅ No future data leakage")
    print(f"✅ Walk-forward validation completed")
    print(f"📁 Models saved in models/ directory")

if __name__ == "__main__":
    main()