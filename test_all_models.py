# test_all_models.py
# Quick test script for all four elite models

import joblib
import numpy as np
import pandas as pd

def test_all_models():
    """Test all four models with proper feature preparation"""
    
    print("🏈 Testing Complete 4-Model System")
    print("=" * 50)
    
    # Load all models
    models = {}
    positions = ['qb', 'rb', 'wr', 'te']
    
    for pos in positions:
        try:
            model_data = joblib.load(f'models/{pos}_model_proper.pkl')
            models[pos] = model_data
            print(f"✅ {pos.upper()} model loaded")
            if 'performance' in model_data:
                mae = model_data['performance'].get('mae', 'N/A')
                corr = model_data['performance'].get('correlation', 'N/A')
                print(f"   Performance: MAE={mae}, Correlation={corr}")
            elif 'feature_columns' in model_data:
                print(f"   Features: {len(model_data['feature_columns'])}")
        except Exception as e:
            print(f"❌ {pos.upper()} model failed: {e}")
            
    print("\n🧪 Testing Individual Model Projections:")
    print("-" * 50)
    
    # Test QB Model
    if 'qb' in models:
        qb_features = create_qb_features("Josh Allen", 22.5, 3.2)
        qb_proj = predict_player(models['qb'], qb_features)
        print(f"Josh Allen (QB): {qb_proj:.1f}")
    
    # Test RB Model  
    if 'rb' in models:
        rb_features = create_rb_features("Christian McCaffrey", 18.2, 2.1)
        rb_proj = predict_player(models['rb'], rb_features)
        print(f"Christian McCaffrey (RB): {rb_proj:.1f}")
    
    # Test WR Model
    if 'wr' in models:
        wr_features = create_wr_features("Tyreek Hill", 16.8, 2.8)
        wr_proj = predict_player(models['wr'], wr_features)
        print(f"Tyreek Hill (WR): {wr_proj:.1f}")
    
    # Test TE Model
    if 'te' in models:
        te_features = create_te_features("Travis Kelce", 14.2, 4.1)
        te_proj = predict_player(models['te'], te_features)
        print(f"Travis Kelce (TE): {te_proj:.1f}")
    
    print("\n🎯 Model Feature Counts:")
    for pos, model_data in models.items():
        if 'feature_columns' in model_data:
            print(f"{pos.upper()}: {len(model_data['feature_columns'])} features")
        elif 'features' in model_data:
            print(f"{pos.upper()}: {len(model_data['features'])} features")

def predict_player(model_data, features):
    """Make prediction using model"""
    try:
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Get feature columns
        if 'feature_columns' in model_data:
            feature_cols = model_data['feature_columns']
        elif 'features' in model_data:
            feature_cols = model_data['features']
        else:
            feature_cols = list(features.keys())
        
        # Create feature array in correct order
        feature_array = []
        for col in feature_cols:
            feature_array.append(features.get(col, 0.0))
        
        # Scale and predict
        scaled_features = scaler.transform([feature_array])
        prediction = model.predict(scaled_features)[0]
        
        return prediction
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.0

def create_qb_features(name, fp_avg_3, fp_trend):
    """Create QB features using known good ranges"""
    return {
        'fp_avg_3': fp_avg_3,
        'fp_trend_3': fp_trend,
        'years_exp': 7,
        'fp_avg_5': fp_avg_3 * 0.95,
        'fp_avg_8': fp_avg_3 * 0.90,
        'fp_consistency_8': 4.2,
        'passing_epa_avg_5': 0.15,
        'air_yards_per_attempt_avg_5': 8.5,
        'completions_avg_3': 22.0,
        'carries_avg_8': 6.5,
        'passing_yards_consistency_5': 45.0,
        'completion_pct_avg_5': 0.67,
        'attempts_avg_8': 34.0,
        'rushing_yards_avg_8': 35.0,
        'yards_per_attempt_avg_8': 7.8,
        'team_total_implied': 25.5,
        'spread_impact': -3.0,
        'weather_wind': 8.0,
        'opponent_def_rank': 18,
        'passing_tds_avg_3': 1.8,
        'interceptions_avg_8': 0.8,
        'rushing_tds_avg_8': 0.4,
        'sacks_avg_5': 2.1,
        'fumbles_avg_8': 0.3,
        'qb_rating_avg_5': 98.5,
        'passer_rating_consistency_5': 15.2,
        'red_zone_passing_tds_avg_3': 1.2,
        'passing_yards_avg_3': 275.0,
        'passing_yards_avg_5': 270.0,
        'passing_yards_avg_8': 265.0,
        'passing_tds_avg_5': 1.7,
        'passing_tds_avg_8': 1.6,
        'rushing_yards_avg_3': 38.0,
        'rushing_yards_avg_5': 36.0,
        'rushing_tds_avg_3': 0.4,
        'rushing_tds_avg_5': 0.4,
        'interceptions_avg_3': 0.8,
        'interceptions_avg_5': 0.8,
        'sacks_avg_3': 2.0,
        'sacks_avg_8': 2.1,
        'fumbles_avg_3': 0.3,
        'fumbles_avg_5': 0.3
    }

def create_rb_features(name, fp_avg_3, fp_trend):
    """Create RB features using known good ranges"""
    return {
        'fp_avg_3': fp_avg_3,
        'fp_trend_3': fp_trend,
        'years_exp': 5,
        'carries_consistency_5': 3.2,
        'fp_avg_5': fp_avg_3 * 0.95,
        'fp_avg_8': fp_avg_3 * 0.90,
        'carries_avg_8': 18.5,
        'fp_consistency_8': 4.8,
        'rushing_yards_avg_8': 85.0,
        'air_yards_share_avg_5': 0.08,
        'target_share_avg_5': 0.12,
        'wopr_avg_5': 0.15,
        'rushing_tds_avg_8': 0.7,
        'targets_avg_8': 4.5,
        'receptions_avg_8': 3.2,
        'receiving_yards_avg_8': 28.0,
        'receiving_tds_avg_8': 0.2,
        'yards_per_carry_avg_8': 4.6,
        'catch_rate_avg_5': 0.72,
        'team_total_implied': 24.0,
        'spread_impact': -2.5,
        'snap_pct_avg_5': 0.68,
        'red_zone_carries_avg_5': 2.8,
        'goal_line_carries_avg_3': 0.8,
        'fumbles_avg_8': 0.4,
        'rushing_yards_avg_3': 88.0,
        'rushing_yards_avg_5': 86.0,
        'carries_avg_3': 19.0,
        'carries_avg_5': 18.8
    }

def create_wr_features(name, fp_avg_3, fp_trend):
    """Create WR features using known good ranges"""
    return {
        'fp_avg_3': fp_avg_3,
        'fp_trend_3': fp_trend,
        'years_exp': 6,
        'targets_avg_8': 8.5,
        'fp_avg_5': fp_avg_3 * 0.95,
        'fp_avg_8': fp_avg_3 * 0.90,
        'target_share_avg_5': 0.22,
        'air_yards_share_avg_5': 0.25,
        'wopr_avg_5': 0.18,
        'catch_rate_avg_5': 0.68,
        'yards_per_target_avg_8': 9.2,
        'receiving_yards_avg_8': 78.0,
        'receiving_tds_avg_8': 0.6,
        'receptions_avg_8': 5.8,
        'fp_consistency_8': 5.2,
        'targets_consistency_5': 2.1,
        'targets_trend_3': 0.5,
        'redzone_targets_avg_5': 1.8,
        'air_yards_per_target_avg_5': 11.5,
        'snap_pct_avg_5': 0.78,
        'team_total_implied': 25.0,
        'spread_impact': -1.5,
        'targets_avg_3': 8.8,
        'targets_avg_5': 8.6,
        'receptions_avg_3': 6.0,
        'receptions_avg_5': 5.9,
        'receiving_yards_avg_3': 80.0,
        'receiving_yards_avg_5': 79.0,
        'receiving_tds_avg_3': 0.6,
        'receiving_tds_avg_5': 0.6
    }

def create_te_features(name, fp_avg_3, fp_trend):
    """Create TE features using ranges from your trained model"""
    return {
        'fp_avg_3': fp_avg_3,
        'fp_avg_5': fp_avg_3 * 0.95,
        'fp_avg_8': fp_avg_3 * 0.90,
        'fp_trend_3': fp_trend,
        'fp_trend_5': fp_trend * 0.8,
        'targets_avg_3': 6.2,
        'targets_avg_5': 6.0,
        'receptions_avg_3': 4.1,
        'receiving_yards_avg_3': 48.0,
        'years_exp': 8,
        'fp_std_3': 4.5,
        'fp_std_5': 4.2,
        'receiving_tds_avg_3': 0.4,
        'air_yards_avg_3': 35.0,
        'yards_per_target_avg_3': 7.8,
        'catch_rate_avg_3': 0.66,
        'opp_te_dk_allow_avg_5': 8.5
    }

if __name__ == "__main__":
    test_all_models()