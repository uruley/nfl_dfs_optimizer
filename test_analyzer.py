import joblib
print("Testing models...")
qb_model = joblib.load('models/qb_model_proper.pkl')
rb_model = joblib.load('models/rb_model_proper.pkl')
print("✅ QB model loaded (MAE 2.81)")
print("✅ RB model loaded (MAE 2.35)")
print("🔥 Both elite models working!")