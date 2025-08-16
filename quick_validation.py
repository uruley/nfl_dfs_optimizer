import pandas as pd
import joblib
import nfl_data_py as nfl
from Scripts.live_qb_features import create_live_qb_features

def quick_validation_test():
    """
    Quick test: Can your model predict 2023 QB performances?
    """
    
    print("🔍 QUICK ML MODEL VALIDATION")
    print("="*50)
    
    # Load 2023 data
    weekly_data = nfl.import_weekly_data([2023])
    qb_data = weekly_data[weekly_data['position'] == 'QB'].copy()
    
    # Focus on known QBs
    target_qbs = ['Jalen Hurts', 'Dak Prescott']
    
    for qb_name in target_qbs:
        print(f"\n📊 Testing {qb_name}...")
        
        qb_games = qb_data[qb_data['player_display_name'].str.contains(qb_name.split()[1], case=False, na=False)]
        
        if not qb_games.empty:
            # Get recent games (weeks 10-17 as test period)
            test_games = qb_games[(qb_games['week'] >= 10) & (qb_games['week'] <= 17)]
            
            print(f"  📈 2023 Week 10-17 actual performances:")
            for _, game in test_games.iterrows():
                week = game['week']
                actual_fp = game['fantasy_points']
                print(f"    Week {week}: {actual_fp:.1f} points")
            
            if len(test_games) > 0:
                avg_actual = test_games['fantasy_points'].mean()
                print(f"  📊 Average actual: {avg_actual:.1f}")
                
                # Compare to your current ML prediction
                if qb_name == 'Jalen Hurts':
                    ml_pred = 14.5
                elif qb_name == 'Dak Prescott':
                    ml_pred = 30.7
                
                difference = abs(ml_pred - avg_actual)
                print(f"  🤖 Your ML prediction: {ml_pred:.1f}")
                print(f"  📏 Difference: {difference:.1f} points")
                
                if difference < 5:
                    print(f"  ✅ Good prediction (within 5 points)")
                else:
                    print(f"  ⚠️  Large difference (>5 points)")

def check_consensus_accuracy():
    """Check how accurate consensus projections were in 2023"""
    
    print(f"\n📊 CONSENSUS ACCURACY CHECK")
    print("="*30)
    
    # Simulate consensus projections (would need real historical consensus data)
    print("📝 Note: Need historical consensus projections for full comparison")
    print("🔍 Your model says:")
    print("  • Hurts was overvalued (ML: 14.5 vs consensus: 22.4)")
    print("  • Dak was undervalued (ML: 30.7 vs consensus: 16.4)")
    print("\n💡 Next step: Test these predictions against 2023 actual results")

if __name__ == "__main__":
    quick_validation_test()
    check_consensus_accuracy()