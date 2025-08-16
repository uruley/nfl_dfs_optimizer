import joblib

print("Testing all models in VS project:")
print("=" * 40)

try:
    qb_old = joblib.load('models/qb_model.pkl')
    print('✅ Original qb_model.pkl loads')
    print('  Type:', type(qb_old))
except Exception as e:
    print('❌ qb_model.pkl error:', e)

try:
    qb_proper = joblib.load('models/qb_model_proper.pkl')
    print('✅ qb_model_proper.pkl loads')
    print('  Type:', type(qb_proper))
    if isinstance(qb_proper, dict) and 'feature_columns' in qb_proper:
        print('  Features:', len(qb_proper['feature_columns']))
except Exception as e:
    print('❌ qb_model_proper.pkl error:', e)

print("\nChecking for RB model...")
try:
    rb = joblib.load('models/rb_model_proper.pkl')
    print('✅ RB model loaded')
except:
    print('❌ No RB model found - need to copy from today')