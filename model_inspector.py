#!/usr/bin/env python3
"""
Quick script to inspect your existing model formats
"""

import joblib
from pathlib import Path

def inspect_model(filepath, position):
    """Inspect a model file and show its structure"""
    
    if not Path(filepath).exists():
        print(f"❌ {position} model not found: {filepath}")
        return
    
    try:
        model_data = joblib.load(filepath)
        print(f"\n📊 {position} MODEL STRUCTURE")
        print("-" * 40)
        print(f"Type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print(f"Keys: {list(model_data.keys())}")
            
            # Check for model
            if 'model' in model_data:
                model = model_data['model']
                print(f"Model type: {type(model)}")
                
                # Check for feature names
                if hasattr(model, 'feature_names_in_'):
                    features = list(model.feature_names_in_)
                    print(f"Features (sklearn): {len(features)}")
                    print(f"First 5: {features[:5]}")
                else:
                    print("No sklearn feature_names_in_ found")
            
            # Check for explicit features list
            if 'features' in model_data:
                features = model_data['features']
                print(f"Features (explicit): {len(features)}")
                print(f"First 5: {features[:5]}")
            
            # Check for performance data
            if 'performance' in model_data:
                perf = model_data['performance']
                print(f"Performance data: {perf}")
            else:
                print("No performance data found")
                
        else:
            print("Model is not a dictionary - older format")
            
    except Exception as e:
        print(f"❌ Error inspecting {position} model: {e}")

def main():
    """Inspect all model files"""
    
    models = {
        'QB': 'models/qb_model_proper.pkl',
        'RB': 'models/rb_model_proper.pkl', 
        'WR': 'models/wr_model_proper.pkl',
        'TE': 'models/te_model_proper.pkl'
    }
    
    print("🔍 MODEL FORMAT INSPECTOR")
    print("=" * 50)
    
    for position, filepath in models.items():
        inspect_model(filepath, position)
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("- If models lack 'features' key, they need to be retrained")
    print("- Or we can extract features from sklearn's feature_names_in_")
    print("- TE model should have the new format with 'performance' data")

if __name__ == "__main__":
    main()