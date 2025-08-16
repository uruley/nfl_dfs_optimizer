import pandas as pd
import nfl_data_py as nfl

def simple_validation_test():
    """
    Simple test: Check 2023 actual performance vs your ML predictions
    """
    
    print("🔍 SIMPLE ML VALIDATION TEST")
    print("="*50)
    
    # Load 2023 NFL data
    print("📥 Loading 2023 NFL data...")
    weekly_data = nfl.import_weekly_data([2023])
    qb_data = weekly_data[weekly_data['position'] == 'QB'].copy()
    
    print(f"✅ Loaded {len(qb_data)} QB performances from 2023")
    
    # Test your key predictions
    predictions = {
        'Jalen Hurts': {'ml_projection': 14.5, 'consensus': 22.4},
        'Dak Prescott': {'ml_projection': 30.7, 'consensus': 16.4}
    }
    
    for qb_name, preds in predictions.items():
        print(f"\n📊 TESTING {qb_name.upper()}:")
        print(f"  🤖 Your ML prediction: {preds['ml_projection']:.1f}")
        print(f"  📰 Consensus projection: {preds['consensus']:.1f}")
        
        # Find QB in 2023 data
        qb_games = qb_data[qb_data['player_display_name'].str.contains(qb_name.split()[1], case=False, na=False)]
        
        if not qb_games.empty:
            print(f"  ✅ Found {len(qb_games)} games in 2023")
            
            # Show recent performance (last 8 games)
            recent_games = qb_games.sort_values(['season', 'week']).tail(8)
            
            print(f"  📈 Last 8 games of 2023:")
            total_points = 0
            for _, game in recent_games.iterrows():
                week = game['week']
                actual_fp = game['fantasy_points']
                total_points += actual_fp
                print(f"    Week {week}: {actual_fp:.1f} points")
            
            # Calculate averages
            avg_actual = recent_games['fantasy_points'].mean()
            season_avg = qb_games['fantasy_points'].mean()
            
            print(f"\n  📊 2023 Performance Summary:")
            print(f"    Recent 8 games avg: {avg_actual:.1f}")
            print(f"    Full season avg: {season_avg:.1f}")
            
            # Compare to your ML prediction
            ml_diff = abs(preds['ml_projection'] - avg_actual)
            consensus_diff = abs(preds['consensus'] - avg_actual)
            
            print(f"\n  🎯 Accuracy Test (vs recent avg):")
            print(f"    ML prediction error: {ml_diff:.1f} points")
            print(f"    Consensus error: {consensus_diff:.1f} points")
            
            if ml_diff < consensus_diff:
                print(f"    ✅ Your ML model was MORE accurate!")
                advantage = consensus_diff - ml_diff
                print(f"    🏆 ML advantage: {advantage:.1f} points better")
            else:
                print(f"    ❌ Consensus was more accurate")
                disadvantage = ml_diff - consensus_diff  
                print(f"    📉 ML disadvantage: {disadvantage:.1f} points worse")
                
        else:
            print(f"  ❌ {qb_name} not found in 2023 data")
    
    # Overall assessment
    print(f"\n🎯 VALIDATION SUMMARY:")
    print(f"This test shows whether your ML model predictions")
    print(f"would have been closer to actual 2023 performance")
    print(f"than consensus projections.")

def check_model_sanity():
    """Quick sanity check on predictions"""
    
    print(f"\n🧠 SANITY CHECK:")
    print(f"="*30)
    print(f"🤖 Your ML model says:")
    print(f"  • Hurts: 22.4 → 14.5 (FADE by -7.9)")
    print(f"  • Dak: 16.4 → 30.7 (BOOST by +14.3)")
    
    print(f"\n💭 Analysis:")
    print(f"  📈 Model heavily favors Dak over Hurts")
    print(f"  🎯 Suggests Dak is massive value, Hurts overpriced")
    print(f"  ⚠️  Dak 30.7 projection seems very high")
    print(f"  🔍 Need to validate against actual 2023 results")

if __name__ == "__main__":
    simple_validation_test()
    check_model_sanity()