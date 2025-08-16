import joblib
import numpy as np

def test_working_wr_features():
    # Load WR model
    wr_model = joblib.load('models/wr_model_proper.pkl')
    print("Testing WR with working feature ranges:")
    
    # Use the feature ranges that gave us 19.3 in diagnosis
    # Start with model means and boost key features
    scaler = wr_model['scaler']
    
    wrs = [
        {'name': 'Tyreek Hill', 'consensus': 16.8, 'boost': 1.2},
        {'name': 'Stefon Diggs', 'consensus': 15.9, 'boost': 1.0}, 
        {'name': 'A.J. Brown', 'consensus': 14.2, 'boost': 0.8}
    ]
    
    for wr in wrs:
        # Start with model's learned means
        features = scaler.mean_.copy()
        
        # Boost the key features that matter
        for i, feature_name in enumerate(wr_model['feature_columns']):
            if 'fp_avg_3' in feature_name:
                features[i] = wr['consensus'] + wr['boost']  # Use consensus + boost
            elif 'fp_trend_3' in feature_name:
                features[i] = wr['boost']  # Positive trend
            elif 'years_exp' in feature_name:
                features[i] = 7.0  # Good experience
            elif 'targets_avg' in feature_name:
                features[i] = 8.0 + wr['boost']  # Good target volume
            elif 'receiving_yards_avg' in feature_name:
                features[i] = 85.0 + (wr['boost'] * 10)  # Good yardage
        
        # Make prediction
        features_scaled = scaler.transform([features])
        prediction = wr_model['model'].predict(features_scaled)[0]
        
        print(f"  {wr['name']}: {prediction:.1f} (target: {wr['consensus']:.1f})")

if __name__ == "__main__":
    test_working_wr_features()