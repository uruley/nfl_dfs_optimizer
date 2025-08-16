"""
Complete Working RB ML Model Training Pipeline  
Includes 2024 data and all safety fixes
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class RBMLTrainer:
    def __init__(self, years=None):
        """Initialize RB ML trainer with historical data"""
        self.years = years or list(range(2020, 2025))  # Include 2024!
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def load_data(self):
        """Load and prepare RB-specific training data"""
        print("Loading NFL data...")
        
        # Load multiple data sources
        pbp = nfl.import_pbp_data(self.years)
        weekly = nfl.import_weekly_data(self.years)
        seasonal = nfl.import_seasonal_data(self.years)
        rosters = nfl.import_seasonal_rosters(self.years)
        
        # Filter for RBs only
        rb_weekly = weekly[weekly['position'] == 'RB'].copy()
        
        # Merge with rosters for additional context
        rb_data = rb_weekly.merge(
            rosters[['player_id', 'depth_chart_position', 'years_exp']], 
            on='player_id', 
            how='left'
        )
        
        print(f"Loaded {len(rb_data)} RB game records from {min(self.years)}-{max(self.years)}")
        return rb_data, pbp
    
    def engineer_rb_features(self, rb_data, pbp):
        """Create RB-specific features with all safety checks"""
        print("Engineering RB-specific features...")
        
        # Sort by player and week for rolling calculations
        rb_data = rb_data.sort_values(['player_id', 'season', 'week'])
        
        # Check what columns are actually available
        available_columns = rb_data.columns.tolist()
        print(f"Available columns: {len(available_columns)} total")
        
        features = []
        
        for player_id in rb_data['player_id'].unique():
            player_data = rb_data[rb_data['player_id'] == player_id].copy()
            
            # Basic stats - rolling averages (SAFE CHECKS)
            for window in [3, 5, 8]:
                if 'carries' in player_data.columns:
                    player_data[f'carries_avg_{window}'] = player_data['carries'].rolling(window, min_periods=1).mean().shift(1)
                else:
                    player_data[f'carries_avg_{window}'] = 15.0
                    
                if 'rushing_yards' in player_data.columns:
                    player_data[f'rushing_yards_avg_{window}'] = player_data['rushing_yards'].rolling(window, min_periods=1).mean().shift(1)
                else:
                    player_data[f'rushing_yards_avg_{window}'] = 60.0
                    
                if 'targets' in player_data.columns:
                    player_data[f'targets_avg_{window}'] = player_data['targets'].rolling(window, min_periods=1).mean().shift(1)
                else:
                    player_data[f'targets_avg_{window}'] = 5.0
                    
                if 'receptions' in player_data.columns:
                    player_data[f'receptions_avg_{window}'] = player_data['receptions'].rolling(window, min_periods=1).mean().shift(1)
                else:
                    player_data[f'receptions_avg_{window}'] = 4.0
                    
                if 'receiving_yards' in player_data.columns:
                    player_data[f'receiving_yards_avg_{window}'] = player_data['receiving_yards'].rolling(window, min_periods=1).mean().shift(1)
                else:
                    player_data[f'receiving_yards_avg_{window}'] = 30.0
                    
                if 'fantasy_points_ppr' in player_data.columns:
                    player_data[f'fp_avg_{window}'] = player_data['fantasy_points_ppr'].rolling(window, min_periods=1).mean().shift(1)
                else:
                    player_data[f'fp_avg_{window}'] = 14.0
            
            # RB efficiency metrics (SAFE)
            if 'rushing_yards' in player_data.columns and 'carries' in player_data.columns:
                player_data['ypc_avg_8'] = (player_data['rushing_yards'] / player_data['carries'].replace(0, 1)).rolling(8, min_periods=1).mean().shift(1)
            else:
                player_data['ypc_avg_8'] = 4.2
            
            # Target share (SAFE)
            if 'target_share' in player_data.columns:
                player_data['target_share_avg_5'] = player_data['target_share'].rolling(5, min_periods=1).mean().shift(1)
            else:
                player_data['target_share_avg_5'] = 0.15
            
            # Air yards share (SAFE)
            if 'air_yards_share' in player_data.columns:
                player_data['air_yards_share_avg_5'] = player_data['air_yards_share'].rolling(5, min_periods=1).mean().shift(1)
            else:
                player_data['air_yards_share_avg_5'] = 0.10
            
            # Consistency metrics (SAFE)
            if 'fantasy_points_ppr' in player_data.columns:
                player_data['fp_consistency_8'] = player_data['fantasy_points_ppr'].rolling(8, min_periods=3).std().shift(1)
            else:
                player_data['fp_consistency_8'] = 3.5
                
            if 'carries' in player_data.columns:
                player_data['carries_consistency_5'] = player_data['carries'].rolling(5, min_periods=3).std().shift(1)
            else:
                player_data['carries_consistency_5'] = 2.5
            
            # Trend metrics (SAFE)
            if 'fantasy_points_ppr' in player_data.columns:
                player_data['fp_trend_3'] = player_data['fantasy_points_ppr'].rolling(3).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
                ).shift(1)
            else:
                player_data['fp_trend_3'] = 0.0
            
            # Snap percentage trends (SAFE CHECK)
            snap_col = None
            for col in ['snap_pct', 'snap_count_offense', 'offense_snaps', 'snaps']:
                if col in player_data.columns:
                    snap_col = col
                    break
            
            if snap_col:
                player_data['snap_pct_trend_3'] = player_data[snap_col].rolling(3).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
                ).shift(1)
            else:
                player_data['snap_pct_trend_3'] = 0.0
            
            # Red zone usage (SAFE CHECK)
            if 'redzone_carries' in player_data.columns:
                player_data['redzone_carries_avg_5'] = player_data['redzone_carries'].rolling(5, min_periods=1).mean().shift(1)
            elif 'red_zone_carries' in player_data.columns:
                player_data['redzone_carries_avg_5'] = player_data['red_zone_carries'].rolling(5, min_periods=1).mean().shift(1)
            else:
                player_data['redzone_carries_avg_5'] = 2.0
            
            # Game environment factors
            player_data['team_total_implied'] = 22.5  # Average team total
            player_data['spread_impact'] = 0.0  # Neutral spread
            
            # Experience factor (SAFE)
            if 'years_exp' in player_data.columns:
                player_data['years_exp'] = player_data['years_exp'].fillna(3)
            else:
                player_data['years_exp'] = 3.0
            
            features.append(player_data)
        
        final_data = pd.concat(features, ignore_index=True)
        
        # Fill NaN values for all numeric columns
        numeric_columns = final_data.select_dtypes(include=[np.number]).columns
        final_data[numeric_columns] = final_data[numeric_columns].fillna(0)
        
        print(f"Feature engineering complete. Final dataset: {len(final_data)} rows")
        return final_data
    
    def prepare_training_data(self, engineered_data):
        """Prepare features and target for training"""
        
        # Define core RB feature columns
        base_feature_columns = [
            'carries_avg_3', 'carries_avg_5', 'carries_avg_8',
            'rushing_yards_avg_3', 'rushing_yards_avg_5', 'rushing_yards_avg_8',
            'targets_avg_3', 'targets_avg_5', 'targets_avg_8',
            'receptions_avg_3', 'receptions_avg_5', 'receptions_avg_8',
            'receiving_yards_avg_3', 'receiving_yards_avg_5', 'receiving_yards_avg_8',
            'fp_avg_3', 'fp_avg_5', 'fp_avg_8',
            'ypc_avg_8', 'target_share_avg_5', 'air_yards_share_avg_5',
            'fp_consistency_8', 'carries_consistency_5', 'fp_trend_3',
            'snap_pct_trend_3', 'redzone_carries_avg_5',
            'team_total_implied', 'spread_impact', 'years_exp'
        ]
        
        # Only use features that actually exist in the data
        feature_columns = []
        for col in base_feature_columns:
            if col in engineered_data.columns:
                feature_columns.append(col)
            else:
                print(f"Warning: {col} not found in data, skipping...")
        
        print(f"Using {len(feature_columns)} features for training")
        
        X = engineered_data[feature_columns].copy()
        y = engineered_data['fantasy_points_ppr'].copy()
        
        # Remove rows where target is NaN or features are all NaN
        mask = ~(y.isna() | X.isna().all(axis=1))
        X = X[mask]
        y = y[mask]
        
        # Add week and season for walk-forward validation
        metadata = engineered_data[mask][['season', 'week', 'player_name', 'player_id']].copy()
        
        print(f"Final training data: {len(X)} examples with {len(X.columns)} features")
        return X, y, metadata
    
    def walk_forward_validation(self, X, y, metadata):
        """Perform walk-forward validation"""
        print("\nPerforming walk-forward validation...")
        
        results = []
        
        # Split by seasons for proper validation
        for test_season in [2023, 2024]:  # Test on recent seasons
            train_mask = metadata['season'] < test_season
            test_mask = metadata['season'] == test_season
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            if len(X_train) == 0 or len(X_test) == 0:
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            predictions = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            correlation = np.corrcoef(y_test, predictions)[0, 1]
            
            results.append({
                'test_season': test_season,
                'mae': mae,
                'r2': r2,
                'correlation': correlation,
                'n_train': len(X_train),
                'n_test': len(X_test)
            })
            
            print(f"Season {test_season}: MAE={mae:.2f}, R²={r2:.3f}, Corr={correlation:.3f}")
        
        return results
    
    def train_final_model(self, X, y):
        """Train final model on all available data"""
        print("\nTraining final RB model on all data...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train final model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"Model trained on {len(X)} examples")
        print("\nTop 10 Most Important Features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.3f}")
    
    def save_model(self, filepath='models/rb_model_proper.pkl'):
        """Save trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'feature_columns': list(self.feature_importance['feature'])
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")
        
        # Print model size
        import os
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Model size: {size_mb:.1f} MB")
    
    def run_full_training(self):
        """Run complete training pipeline"""
        print("🏈 Starting RB ML Training Pipeline (2020-2024)")
        print("=" * 50)
        
        # Load data
        rb_data, pbp = self.load_data()
        
        # Engineer features
        engineered_data = self.engineer_rb_features(rb_data, pbp)
        
        # Prepare training data
        X, y, metadata = self.prepare_training_data(engineered_data)
        
        print(f"\nTraining dataset: {len(X)} examples with {len(X.columns)} features")
        
        # Validate model
        validation_results = self.walk_forward_validation(X, y, metadata)
        
        # Train final model
        self.train_final_model(X, y)
        
        # Save model
        self.save_model()
        
        print("\n🎯 RB Model Training Complete!")
        return validation_results

if __name__ == "__main__":
    # Initialize and run training
    trainer = RBMLTrainer(years=list(range(2020, 2025)))  # Include 2024
    results = trainer.run_full_training()
    
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    for result in results:
        print(f"Season {result['test_season']}: MAE={result['mae']:.2f}, Correlation={result['correlation']:.3f}")