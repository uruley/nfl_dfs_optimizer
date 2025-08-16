"""
Quick fix for QB feature preparation
"""
import pandas as pd

# Test QB projections with proper features
sample_qbs = [
    {'name': 'Josh Allen', 'position': 'QB', 'salary': 8200, 'consensus_projection': 22.1},
    {'name': 'Lamar Jackson', 'position': 'QB', 'salary': 7900, 'consensus_projection': 20.5}
]

def test_qb_projections():
    import joblib
    import numpy as np
    
    # Load QB model
    qb_model = joblib.load('models/qb_model_proper.pkl')
    print("Testing QB projections with realistic features:")
    
    for qb in sample_qbs:
        # Create realistic QB features
        features = []
        for feature_name in qb_model['feature_columns']:
            if 'fp_avg_3' in feature_name:
                features.append(qb['consensus_projection'])  # Use consensus as baseline
            elif 'fp_trend_3' in feature_name:
                features.append(1.0 if qb['name'] == 'Josh Allen' else -0.5)
            elif 'years_exp' in feature_name:
                features.append(6 if qb['name'] == 'Josh Allen' else 6)
            elif 'completions_avg_3' in feature_name:
                features.append(24 if qb['name'] == 'Josh Allen' else 20)
            elif 'attempts_avg_3' in feature_name:
                features.append(35 if qb['name'] == 'Josh Allen' else 30)
            elif 'passing_yards_avg_3' in feature_name:
                features.append(280 if qb['name'] == 'Josh Allen' else 230)
            elif 'carries_avg_8' in feature_name:
                features.append(6 if qb['name'] == 'Josh Allen' else 8)
            elif 'rushing_yards_avg_8' in feature_name:
                features.append(35 if qb['name'] == 'Josh Allen' else 60)
            else:
                features.append(0.0)
        
        # Make prediction
        features_scaled = qb_model['scaler'].transform([features])
        prediction = qb_model['model'].predict(features_scaled)[0]
        
        print(f"  {qb['name']}: {prediction:.1f} (vs consensus {qb['consensus_projection']:.1f})")

if __name__ == "__main__":
    test_qb_projections()