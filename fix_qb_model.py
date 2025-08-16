🏆 SUCCESS! Real DraftKings Showdown lineup generated!
PS C:\Users\ruley\dfs_optimizer> python Scripts\qb_model_integration.py
🏈 TESTING QB MODEL INTEGRATION
==================================================
📊 Found 7 QBs in showdown data
❌ Error loading QB model: STACK_GLOBAL requires str
Available model files:
  - qb_model.pkl
❌ QB model integration failed
PS C:\Users\ruley\dfs_optimizer> python Scripts\fix_qb_model.py
🔍 DIAGNOSING QB MODEL ISSUE
==================================================
✅ Model file exists: models/qb_model.pkl
📊 File size: 18,964,833 bytes

🔧 Trying different loading methods...
❌ Method 1 (pickle): STACK_GLOBAL requires str
✅ Method 2 (joblib): SUCCESS
   Model type: <class 'sklearn.ensemble._forest.RandomForestRegressor'>

✅ Model loading successful!

🧪 Testing working model...
❌ Working model test failed: [Errno 2] No such file or directory: 'models/qb_model_working.pkl'

🎯 NEXT STEPS:
1. If working model created successfully:
   - Modify qb_model_integration.py to use 'models/qb_model_working.pkl'
2. Test QB integration again
3. Integrate with showdown optimizer
PS C:\Users\ruley\dfs_optimizer> python Scripts\qb_model_integration.py
🏈 TESTING QB MODEL INTEGRATION
==================================================
📊 Found 7 QBs in showdown data
❌ Error loading QB model: STACK_GLOBAL requires str
Available model files:
  - qb_model.pkl
❌ QB model integration failed
PS C:\Users\ruley\dfs_optimizer>