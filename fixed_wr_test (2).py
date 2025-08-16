import joblib
import numpy as np

def test_wr_projections_fixed():
    # Load WR model
    wr_model = joblib.load('models/wr_model_proper.pkl')
    print("Testing WR projections with proper features:")
    
    # Realistic WR feature profiles
    wr_profiles = {
        'Tyreek Hill': {
            'fp_avg_3': 16.8, 'fp_trend_3': 1.2, 'years_exp': 8,
            'fp_avg_5': 16.2, 'targets_avg_8': 8.5, 'fp_avg_8': 15.8,
            'target_share_avg_5': 0.22, 'fp_consistency_8': 4.8, 'wopr_avg_5': 0.18,
            'catch_rate_avg_5': 0.75, 'receiving_yards_avg_8': 85, 'air_yards_share_avg_5': 0.25,
            'receptions_avg_8': 6.5, 'receiving_yards_avg_5': 88, 'yards_per_target_avg_8': 10.5
        },
        'Stefon Diggs': {
            'fp_avg_3': 15.9, 'fp_trend_3': 0.5, 'years_exp': 9,
            'fp_avg_5': 15.5, 'targets_avg_8': 9.2, 'fp_avg_8': 15.2,
            'target_share_avg_5': 0.25, 'fp_consistency_8': 3.9, 'wopr_avg_5': 0.20,
            'catch_rate_avg_5': 0.78, 'receiving_yards_avg_8': 92, 'air_yards_share_avg_5': 0.22,
            'receptions_avg_8': 7.2, 'receiving_yards_avg_5': 95, 'yards_per_target_avg_8': 10.0
        },
        'A.J. Brown': {
            'fp_avg_3': 14.2, 'fp_trend_3': -0.3, 'years_exp': 5,
            'fp_avg_5': 14.8, 'targets_avg_8': 7.5, 'fp_avg_8': 14.5,
            'target_share_avg_5': 0.18, 'fp_consistency_8': 5.2, 'wopr_avg_5': 0.16,
            'catch_rate_avg_5': 0.72, 'receiving_yards_avg_8': 78, 'air_yards_share_avg_5': 0.20,
            'receptions_avg_8': 5.4, 'receiving_yards_avg_5': 82, 'yards_per_target_avg_8': 10.8
        }
    }
    
    for wr_name, profile in wr_profiles.items():
        # Create feature vector with all 30 features
        features = []
        for feature_name in wr_model['feature_columns']:
            if feature_name in profile:
                features.append(profile[feature_name])
            else:
                # Default values for missing features
                features.append(0.0)
        
        # Make prediction
        features_scaled = wr_model['scaler'].transform([features])
        prediction = wr_model['model'].predict(features_scaled)[0]
        
        print(f"  {wr_name}: {prediction:.1f}")

if __name__ == "__main__":
    test_wr_projections_fixed()