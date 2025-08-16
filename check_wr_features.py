import joblib

wr_model = joblib.load('models/wr_model_proper.pkl')
print('WR Model Features:')
for i, feature in enumerate(wr_model['feature_columns'][:15]):
    print(f'  {i+1:2d}. {feature}')
print(f'  ... and {len(wr_model["feature_columns"]) - 15} more')