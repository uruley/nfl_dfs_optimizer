# dict_inspector.py - Inspect your dictionary-based models
import joblib
import pandas as pd
import numpy as np

def inspect_model_dict():
    """Inspect the dictionary structure of your models"""
    
    print("🔍 INSPECTING MODEL DICTIONARY STRUCTURE")
    print("=" * 60)
    
    try:
        # Load QB model as example
        qb_model = joblib.load('models/qb_model_proper.pkl')
        
        print("📋 QB Model Dictionary Keys:")
        for key in qb_model.keys():
            print(f"   '{key}': {type(qb_model[key])}")
        
        print(f"\n🔑 Dictionary Contents:")
        for key, value in qb_model.items():
            print(f"\n--- {key} ---")
            if key == 'model' and hasattr(value, 'feature_names_in_'):
                print(f"  Model type: {type(value)}")
                print(f"  Features: {len(value.feature_names_in_)}")
                print(f"  Feature names: {list(value.feature_names_in_)[:5]}...")
            elif key == 'feature_columns':
                print(f"  Feature columns ({len(value)}): {value[:5]}...")
            elif key == 'scaler':
                print(f"  Scaler type: {type(value)}")
                if hasattr(value, 'feature_names_in_'):
                    print(f"  Scaler features: {len(value.feature_names_in_)}")
            else:
                print(f"  Type: {type(value)}")
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    print(f"  Length: {len(value)}")
                    print(f"  First few: {value[:3]}")
                elif isinstance(value, str):
                    print(f"  Value: {value}")
                elif isinstance(value, (int, float)):
                    print(f"  Value: {value}")
        
        return qb_model
        
    except Exception as e:
        print(f"Error inspecting model: {e}")
        return None

def test_model_prediction(model_dict):
    """Test how to use the dictionary model for predictions"""
    
    print(f"\n🎯 TESTING MODEL PREDICTION")
    print("=" * 40)
    
    if 'model' in model_dict and 'feature_columns' in model_dict:
        try:
            actual_model = model_dict['model']
            feature_names = model_dict['feature_columns']
            
            print(f"✅ Found model and feature columns")
            print(f"   Model type: {type(actual_model)}")
            print(f"   Features needed: {len(feature_names)}")
            print(f"   Feature names: {feature_names[:5]}...")
            
            # Create test data with correct feature names
            test_data = pd.DataFrame([np.random.random(len(feature_names))], 
                                   columns=feature_names)
            
            # Check if there's a scaler
            if 'scaler' in model_dict:
                print(f"   Found scaler: {type(model_dict['scaler'])}")
                scaled_data = model_dict['scaler'].transform(test_data)
                test_pred = actual_model.predict(scaled_data)[0]
                print(f"   ✅ Test prediction (with scaling): {test_pred:.2f}")
            else:
                test_pred = actual_model.predict(test_data)[0]
                print(f"   ✅ Test prediction (no scaling): {test_pred:.2f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            return False
    else:
        print("   ❌ Model dictionary doesn't have expected structure")
        return False

if __name__ == "__main__":
    model_dict = inspect_model_dict()
    if model_dict:
        test_model_prediction(model_dict)