"""
Wide Receiver ML Model Training Pipeline
Adapted from the successful RB model framework (MAE 2.35)
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class WRMLTrainer:
    def __init__(self, years=None):
        """Initialize WR ML trainer with historical data"""
        self.years = years or list(range(2020, 2024))
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def load_data(self):
        """Load and prepare WR-specific training data"""
        print("Loading NFL data...")
        
        # Load multiple data sources
        pbp = nfl.import_pbp_data(self.years)
        weekly = nfl.import_weekly_data(self.years)
        seasonal = nfl.import_seasonal_data(self.years)
        rosters = nfl.import_seasonal_rosters(self.years)
        
        # Filter for WRs only
        wr_weekly = weekly[weekly['position'] == 'WR'].copy()
        
        # Merge with rosters for additional context
        wr_data = wr_weekly.merge(
            rosters[['player_id', 'depth_chart_position', 'years_exp']], 
            on='player_id', 
            how='left'
        )
        
        print(f"Loaded {len(wr_data)} WR game records from {min(self.years)}-{max(self.years)}")
        return wr_data, pbp
    
    def engineer_wr_features(self, wr_data, pbp):
        """Create WR-specific features"""
        print("Engineering WR-specific features...")
        
        # Sort by player and week for rolling calculations
        wr_data = wr_data.sort_values(['player_id', 'season', 'week'])
        
        # Check what columns are actually available
        available_columns = wr_data.columns.tolist()
        print(f"Available columns: {len(available_columns)} total")
        
        features = []
        
        for player_id in wr_data['player_id'].unique():
            player_data = wr_data[wr_data['player_id'] == player_id].copy()
            
            # Basic receiving stats - rolling averages
            for window in [3, 5, 8]:
                if 'targets' in player_data.columns:
                    player_data[f'targets_avg_{window}'] = player_data['targets'].rolling(window, min_periods=1).mean().shift(1)
                if 'receptions' in player_data.columns:
                    player_data[f'receptions_avg_{window}'] = player_data['receptions'].rolling(window, min_periods=1).mean().shift(1)
                if 'receiving_yards' in player_data.columns:
                    player_data[f'receiving_yards_avg_{window}'] = player_data['receiving_yards'].rolling(window, min_periods=1).mean().shift(1)
                if 'receiving_tds' in player_data.columns:
                    player_data[f'receiving_tds_avg_{window}'] = player_data['receiving_tds'].rolling(window, min_periods=1).mean().shift(1)
                if 'fantasy_points_ppr' in player_data.columns:
                    player_data[f'fp_avg_{window}'] = player_data['fantasy_points_ppr'].rolling(window, min_periods=1).mean().shift(1)
            
            # WR efficiency metrics
            if 'receiving_yards' in player_data.columns and 'targets' in player_data.columns:
                player_data['yards_per_target_avg_8'] = (player_data['receiving_yards'] / player_data['targets'].replace(0, 1)).rolling(8, min_periods=1).mean().shift(1)
            else:
                player_data['yards_per_target_avg_8'] = 8.5  # NFL average
            
            if 'receptions' in player_data.columns and 'targets' in player_data.columns:
                player_data['catch_rate_avg_5'] = (player_data['receptions'] / player_data['targets'].replace(0, 1)).rolling(5, min_periods=1).mean().shift(1)
            else:
                player_data['catch_rate_avg_5'] = 0.65  # NFL average
            
            # Advanced WR metrics (with safe checks)
            if 'target_share' in player_data.columns:
                player_data['target_share_avg_5'] = player_data['target_share'].rolling(5, min_periods=1).mean().shift(1)
            else:
                player_data['target_share_avg_5'] = 0.15
            
            if 'air_yards_share' in player_data.columns:
                player_data['air_yards_share_avg_5'] = player_data['air_yards_share'].rolling(5, min_periods=1).mean().shift(1)
            else:
                player_data['air_yards_share_avg_5'] = 0.20
            
            if 'wopr' in player_data.columns:
                player_data['wopr_avg_5'] = player_data['wopr'].rolling(5, min_periods=1).mean().shift(1)
            else:
                player_data['wopr_avg_5'] = 0.10
            
            # Consistency metrics
            if 'fantasy_points_ppr' in player_data.columns:
                player_data['fp_consistency_8'] = (
                    player_data['fantasy_points_ppr'].rolling(8, min_periods=3).std().shift(1)
                )
            
            if 'targets' in player_data.columns:
                player_data['targets_consistency_5'] = (
                    player_data['targets'].rolling(5, min_periods=3).std().shift(1)
                )
            
            # Trend metrics
            if 'fantasy_points_ppr' in player_data.columns:
                player_data['fp_trend_3'] = player_data['fantasy_points_ppr'].rolling(3).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
                ).shift(1)
            
            # Usage trends
            if 'targets' in player_data.columns:
                player_data['targets_trend_3'] = player_data['targets'].rolling(3).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
                ).shift(1)
            
            # Red zone metrics
            if 'red_zone_targets' in player_data.columns:
                player_data['redzone_targets_avg_5'] = player_data['red_zone_targets'].rolling(5, min_periods=1).mean().shift(1)
            else:
                player_data['redzone_targets_avg_5'] = 1.5
            
            # Depth of target metrics
            if 'air_yards_per_target' in player_data.columns:
                player_data['air_yards_per_target_avg_5'] = player_data['air_yards_per_target'].rolling(5, min_periods=1).mean().shift(1)
            else:
                player_data['air_yards_per_target_avg_5'] = 8.0
            
            # Game environment factors
            player_data['team_total_implied'] = 22.5  # Average team total
            player_data['spread_impact'] = 0.0  # Neutral spread
            
            # Experience factor
            if 'years_exp' in player_data.columns:
                player_data['years_exp'] = player_data['years_exp'].fillna(3)
            else:
                player_data['years_exp'] = 3  # Default experience
            
            # Snap percentage (important for WRs)
            snap_col = None
            for col in ['snap_pct', 'snap_count_offense', 'offense_snaps', 'snaps']:
                if col in player_data.columns:
                    snap_col = col
                    break
            
            if snap_col:
                player_data['snap_pct_avg_5'] = player_data[snap_col].rolling(5, min_periods=1).mean().shift(1)
            else:
                player_data['snap_pct_avg_5'] = 0.65
            
            features.append(player_data)
        
        final_data = pd.concat(features, ignore_index=True)
        
        # Fill NaN values for all numeric columns
        numeric_columns = final_data.select_dtypes(include=[np.number]).columns
        final_data[numeric_columns] = final_data[numeric_columns].fillna(0)
        
        print(f"Feature engineering complete. Final dataset: {len(final_data)} rows")
        return final_data
    
    def prepare_training_data(self, engineered_data):
        """Prepare features and target for training"""
        
        # Define core WR feature columns
        base_feature_columns = [
            'targets_avg_3', 'targets_avg_5', 'targets_avg_8',
            'receptions_avg_3', 'receptions_avg_5', 'receptions_avg_8',
            'receiving_yards_avg_3', 'receiving_yards_avg_5', 'receiving_yards_avg_8',
            'receiving_tds_avg_3', 'receiving_tds_avg_5', 'receiving_tds_avg_8',
            'fp_avg_3', 'fp_avg_5', 'fp_avg_8',
            'yards_per_target_avg_8', 'catch_rate_avg_5', 'target_share_avg_5',
            'air_yards_share_avg_5', 'wopr_avg_5', 'fp_consistency_8',
            'targets_consistency_5', 'fp_trend_3', 'targets_trend_3',
            'redzone_targets_avg_5', 'air_yards_per_target_avg_5',
            'snap_pct_avg_5', 'team_total_implied', 'spread_impact', 'years_exp'
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
        for test_season in [2022, 2023]:
            train_mask = metadata['season'] < test_season
            test_mask = metadata['season'] == test_season
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            if len(X_train) == 0 or len(X_test) == 0:
                continue
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model (adjusted hyperparameters for WRs)
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
        print("\nTraining final WR model on all data...")
        
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
    
    def save_model(self, filepath='models/wr_model_proper.pkl'):
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
        print("🏈 Starting WR ML Training Pipeline")
        print("=" * 50)
        
        # Load data
        wr_data, pbp = self.load_data()
        
        # Engineer features
        engineered_data = self.engineer_wr_features(wr_data, pbp)
        
        # Prepare training data
        X, y, metadata = self.prepare_training_data(engineered_data)
        
        print(f"\nTraining dataset: {len(X)} examples with {len(X.columns)} features")
        
        # Validate model
        validation_results = self.walk_forward_validation(X, y, metadata)
        
        # Train final model
        self.train_final_model(X, y)
        
        # Save model
        self.save_model()
        
        print("\n🎯 WR Model Training Complete!")
        return validation_results

if __name__ == "__main__":
    # Initialize and run training
    trainer = WRMLTrainer(years=list(range(2020, 2024)))
    results = trainer.run_full_training()
    
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    for result in results:
        print(f"Season {result['test_season']}: MAE={result['mae']:.2f}, Correlation={result['correlation']:.3f}")