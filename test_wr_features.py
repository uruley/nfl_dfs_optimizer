import joblib
import numpy as np

def test_wr_projections():
    # Load WR model
    wr_model = joblib.load('models/wr_model_proper.pkl')
    print("Testing WR projections:")
    
    # Test with better WR features
    wrs = [
        {'name': 'Tyreek Hill', 'consensus': 16.8},
        {'name': 'Stefon Diggs', 'consensus': 15.9}, 
        {'name': 'A.J. Brown', 'consensus': 14.2}
    ]
    
    for wr in wrs:
        features = []
        for feature_name in wr_model['feature_columns']:
            if 'fp_avg_3' in feature_name:
                features.append(wr['consensus'])  # Use consensus as baseline
            elif 'fp_trend_3' in feature_name:
                features.append(1.0 if wr['name'] == 'Tyreek Hill' else 0.0)
            elif 'targets_avg_' in feature_name:
                features.append(8.5 if wr['name'] == 'Stefon Diggs' else 7.5)
            elif 'target_share_avg_5' in feature_name:
                features.append(0.22 if wr['name'] == 'Tyreek Hill' else 0.18)
            elif 'years_exp' in feature_name:
                features.append(8 if wr['name'] == 'Tyreek Hill' else 5)
            else:
                features.append(0.0)
        
        # Make prediction
        features_scaled = wr_model['scaler'].transform([features])
        prediction = wr_model['model'].predict(features_scaled)[0]
        
        print(f"  {wr['name']}: {prediction:.1f} (vs consensus {wr['consensus']:.1f})")

if __name__ == "__main__":
    test_wr_projections()