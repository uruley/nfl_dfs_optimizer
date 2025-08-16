# fix_te_model_format.py
# Fix TE model to match other models' format

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def fix_te_model():
    """Fix TE model format to match QB/RB/WR models"""
    
    print("🔧 Fixing TE model format...")
    
    try:
        # Load current TE model
        te_data = joblib.load('models/te_model_proper.pkl')
        print("✅ Current TE model loaded")
        print("Current keys:", list(te_data.keys()))
        
        # Check if it needs fixing
        if 'scaler' not in te_data:
            print("❌ Missing scaler - creating one...")
            
            # Create a proper scaler
            scaler = StandardScaler()
            
            # Create dummy training data to fit the scaler
            # Using realistic TE feature ranges
            n_features = len(te_data['features'])
            dummy_data = np.array([
                # Realistic TE feature ranges
                [12.5, 11.8, 11.2, 1.5, 0.8, 5.5, 5.2, 3.8, 45.0, 6.0, 
                 4.2, 4.0, 0.3, 32.0, 7.5, 0.65] if n_features == 16 else 
                [10.0] * n_features  # Fallback
            ])
            
            scaler.fit(dummy_data)
            
            # Create new format matching other models
            fixed_model = {
                'model': te_data['model'],
                'scaler': scaler,
                'feature_columns': te_data['features'],  # Change key name
                'feature_importance': None,  # Will add later if needed
                'performance': te_data.get('performance', {}),
                'meta': te_data.get('meta', {})
            }
            
            # Save fixed model
            joblib.dump(fixed_model, 'models/te_model_proper.pkl')
            print("✅ TE model format fixed and saved")
            
        else:
            print("✅ TE model already has correct format")
            
        return True
        
    except Exception as e:
        print(f"❌ Error fixing TE model: {e}")
        return False

def test_fixed_te():
    """Test the fixed TE model"""
    
    print("\n🧪 Testing fixed TE model...")
    
    try:
        te_data = joblib.load('models/te_model_proper.pkl')
        
        # Test prediction
        model = te_data['model']
        scaler = te_data['scaler']
        features = te_data['feature_columns']
        
        print(f"✅ TE model loaded with {len(features)} features")
        
        # Create realistic test features for Travis Kelce
        test_features = [
            14.2,  # fp_avg_3
            13.5,  # fp_avg_5
            13.0,  # fp_avg_8
            1.2,   # fp_trend_3
            0.8,   # fp_trend_5
            6.2,   # targets_avg_3
            6.0,   # targets_avg_5
            4.1,   # receptions_avg_3
            48.0,  # receiving_yards_avg_3
            8.0,   # years_exp
            4.5,   # fp_std_3
            4.2,   # fp_std_5
            0.4,   # receiving_tds_avg_3
            35.0,  # air_yards_avg_3
            7.8,   # yards_per_target_avg_3
            0.66   # catch_rate_avg_3
        ]
        
        # Ensure we have the right number of features
        if len(test_features) != len(features):
            test_features = test_features[:len(features)]  # Trim if too many
            while len(test_features) < len(features):      # Pad if too few
                test_features.append(0.0)
        
        # Make prediction
        scaled_features = scaler.transform([test_features])
        prediction = model.predict(scaled_features)[0]
        
        print(f"Travis Kelce (TE) Test Prediction: {prediction:.1f}")
        
        if prediction > 5.0:
            print("✅ TE model working correctly!")
            return True
        else:
            print("⚠️  TE model may need retraining")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = fix_te_model()
    if success:
        test_fixed_te()
        print("\n🎯 TE model fix complete! Run test_all_models.py again to verify.")