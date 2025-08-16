import joblib
import numpy as np

def diagnose_wr_model():
    # Load WR model
    wr_model = joblib.load('models/wr_model_proper.pkl')
    
    print("WR Model Feature Analysis:")
    print("=" * 40)
    
    # Check the scaler's learned parameters
    scaler = wr_model['scaler']
    feature_names = wr_model['feature_columns']
    
    print("Feature ranges the model expects:")
    for i, feature in enumerate(feature_names[:10]):
        mean_val = scaler.mean_[i]
        scale_val = scaler.scale_[i]
        print(f"{feature:25} - Mean: {mean_val:6.2f}, Scale: {scale_val:6.2f}")
    
    print("\nTesting with model's expected ranges:")
    
    # Use the model's learned means as baseline
    test_features = scaler.mean_.copy()
    
    # Adjust the top features to realistic high values
    for i, feature in enumerate(feature_names):
        if 'fp_avg_3' in feature:
            test_features[i] = 16.0  # Good WR fantasy avg
        elif 'fp_trend_3' in feature:
            test_features[i] = 1.0   # Positive trend
        elif 'years_exp' in feature:
            test_features[i] = 6.0   # Experienced player
    
    # Make prediction with realistic ranges
    features_scaled = scaler.transform([test_features])
    prediction = wr_model['model'].predict(features_scaled)[0]
    
    print(f"\nTest prediction with model means: {prediction:.1f}")
    
    # Try with even higher baseline
    test_features2 = test_features.copy()
    for i, feature in enumerate(feature_names):
        if 'fp_avg_3' in feature:
            test_features2[i] = 18.0  # Higher baseline
        elif 'targets_avg' in feature:
            test_features2[i] = 8.0   # Good target volume
        elif 'receiving_yards_avg' in feature:
            test_features2[i] = 80.0  # Good yardage
    
    features_scaled2 = scaler.transform([test_features2])
    prediction2 = wr_model['model'].predict(features_scaled2)[0]
    
    print(f"Test prediction with higher values: {prediction2:.1f}")

if __name__ == "__main__":
    diagnose_wr_model()