import joblib
import pandas as pd

def inspect_qb_model():
    """Inspect what features the QB model expects"""
    
    print("🔍 INSPECTING QB MODEL")
    print("="*50)
    
    # Load the model
    model = joblib.load('models/qb_model.pkl')
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"📊 Expected features: {model.n_features_in_}")
    
    # Try to get feature names if available
    if hasattr(model, 'feature_names_in_'):
        print(f"\n📋 Feature names ({len(model.feature_names_in_)}):")
        for i, feature in enumerate(model.feature_names_in_):
            print(f"  {i+1:2d}. {feature}")
    else:
        print(f"\n⚠️  Feature names not stored in model")
        print(f"Model expects {model.n_features_in_} features but names unknown")
    
    # Check if it's part of a pipeline or has other info
    print(f"\n🔧 Model attributes:")
    for attr in dir(model):
        if not attr.startswith('_') and not callable(getattr(model, attr)):
            try:
                value = getattr(model, attr)
                if isinstance(value, (int, float, str, bool)):
                    print(f"  {attr}: {value}")
            except:
                pass

if __name__ == "__main__":
    inspect_qb_model()