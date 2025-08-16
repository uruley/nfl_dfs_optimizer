#!/usr/bin/env python3
"""
Updated Working Optimizer - Now with all 4 elite models!
Tests QB, RB, WR, and TE models with sample player projections
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_all_models():
    """Load all four trained models"""
    models = {}
    model_files = {
        'QB': 'models/qb_model_proper.pkl',
        'RB': 'models/rb_model_proper.pkl', 
        'WR': 'models/wr_model_proper.pkl',
        'TE': 'models/te_model_proper.pkl'
    }
    
    print("Loading elite ML models...")
    
    for position, filepath in model_files.items():
        try:
            if Path(filepath).exists():
                model_data = joblib.load(filepath)
                models[position] = model_data
                
                perf = model_data['performance']
                print(f"✅ {position} model loaded - MAE: {perf['mae']:.2f}, Corr: {perf['correlation']:.3f}")
            else:
                print(f"❌ {position} model not found at {filepath}")
                if position == 'TE':
                    print("   Run: python Scripts/te_ml_training.py")
                    
        except Exception as e:
            print(f"❌ Error loading {position} model: {e}")
    
    return models

def create_sample_player_data():
    """Create sample players with realistic feature values"""
    
    # Sample players with estimated feature values
    sample_players = [
        # Elite QBs
        {
            'name': 'Josh Allen', 'position': 'QB', 'salary': 8200, 'team': 'BUF',
            'fp_avg_3': 22.4, 'fp_avg_5': 21.8, 'fp_trend_3': 2.1,
            'pass_yards_avg_3': 280, 'pass_tds_avg_3': 2.2, 'rush_yards_avg_3': 42,
            'years_exp': 6, 'games_played': 45, 'fp_std_5': 6.2
        },
        {
            'name': 'Lamar Jackson', 'position': 'QB', 'salary': 7800, 'team': 'BAL',
            'fp_avg_3': 24.1, 'fp_avg_5': 23.2, 'fp_trend_3': 1.8,
            'pass_yards_avg_3': 260, 'pass_tds_avg_3': 1.9, 'rush_yards_avg_3': 68,
            'years_exp': 7, 'games_played': 78, 'fp_std_5': 7.8
        },
        
        # Elite RBs  
        {
            'name': 'Christian McCaffrey', 'position': 'RB', 'salary': 9000, 'team': 'SF',
            'fp_avg_3': 21.2, 'fp_avg_5': 20.1, 'fp_trend_3': 1.4,
            'rush_yards_avg_3': 98, 'rush_tds_avg_3': 1.1, 'targets_avg_3': 6.8,
            'years_exp': 7, 'games_played': 82, 'fp_std_5': 4.2
        },
        {
            'name': 'Austin Ekeler', 'position': 'RB', 'salary': 7400, 'team': 'WSH',
            'fp_avg_3': 16.8, 'fp_avg_5': 15.9, 'fp_trend_3': -0.8,
            'rush_yards_avg_3': 72, 'rush_tds_avg_3': 0.8, 'targets_avg_3': 5.2,
            'years_exp': 7, 'games_played': 91, 'fp_std_5': 5.1
        },
        
        # Elite WRs
        {
            'name': 'Tyreek Hill', 'position': 'WR', 'salary': 8600, 'team': 'MIA',
            'fp_avg_3': 18.9, 'fp_avg_5': 19.4, 'fp_trend_3': 2.4,
            'targets_avg_3': 9.2, 'receptions_avg_3': 6.8, 'receiving_yards_avg_3': 112,
            'years_exp': 8, 'games_played': 108, 'fp_std_5': 6.8
        },
        {
            'name': 'A.J. Brown', 'position': 'WR', 'salary': 7800, 'team': 'PHI',
            'fp_avg_3': 17.2, 'fp_avg_5': 16.1, 'fp_trend_3': 0.9,
            'targets_avg_3': 8.1, 'receptions_avg_3': 5.4, 'receiving_yards_avg_3': 89,
            'years_exp': 5, 'games_played': 67, 'fp_std_5': 5.9
        },
        
        # Elite TEs
        {
            'name': 'Travis Kelce', 'position': 'TE', 'salary': 7200, 'team': 'KC',
            'fp_avg_3': 16.8, 'fp_avg_5': 17.2, 'fp_trend_3': 0.6,
            'targets_avg_3': 8.9, 'receptions_avg_3': 6.7, 'receiving_yards_avg_3': 78,
            'td_rate_avg_5': 0.18, 'snap_pct_avg_3': 0.89, 'target_share_avg_3': 0.24,
            'years_exp': 11, 'games_played': 152, 'fp_std_5': 4.1
        },
        {
            'name': 'Mark Andrews', 'position': 'TE', 'salary': 6400, 'team': 'BAL',
            'fp_avg_3': 13.1, 'fp_avg_5': 12.8, 'fp_trend_3': -1.2,
            'targets_avg_3': 6.8, 'receptions_avg_3': 4.9, 'receiving_yards_avg_3': 58,
            'td_rate_avg_5': 0.15, 'snap_pct_avg_3': 0.76, 'target_share_avg_3': 0.19,
            'years_exp': 6, 'games_played': 89, 'fp_std_5': 5.2
        }
    ]
    
    return pd.DataFrame(sample_players)

def generate_projections(models, players_df):
    """Generate projections for all players using appropriate models"""
    
    projections = []
    
    for _, player in players_df.iterrows():
        position = player['position']
        
        if position not in models:
            print(f"⚠️ No model available for {position} - {player['name']}")
            continue
            
        model_data = models[position]
        model = model_data['model']
        required_features = model_data['features']
        
        # Prepare feature vector
        feature_values = []
        missing_features = []
        
        for feature in required_features:
            if feature in player.index and pd.notna(player[feature]):
                feature_values.append(player[feature])
            else:
                # Use position-specific defaults for missing features
                default_value = get_default_feature_value(position, feature)
                feature_values.append(default_value)
                missing_features.append(feature)
        
        if len(feature_values) != len(required_features):
            print(f"⚠️ Feature mismatch for {player['name']}")
            continue
        
        # Generate projection
        try:
            feature_array = np.array(feature_values).reshape(1, -1)
            projection = model.predict(feature_array)[0]
            
            # Calculate edge vs salary-based expectation
            salary_expectation = estimate_salary_expectation(player['salary'], position)
            edge = projection - salary_expectation
            
            projections.append({
                'name': player['name'],
                'position': position,
                'team': player['team'],
                'salary': player['salary'],
                'projection': projection,
                'salary_expectation': salary_expectation,
                'edge': edge,
                'missing_features': len(missing_features)
            })
            
        except Exception as e:
            print(f"❌ Projection error for {player['name']}: {e}")
    
    return pd.DataFrame(projections)

def get_default_feature_value(position, feature):
    """Get reasonable default values for missing features by position"""
    
    defaults = {
        'QB': {
            'fp_avg_3': 18.0, 'fp_avg_5': 17.5, 'fp_trend_3': 0.0,
            'pass_yards_avg_3': 250, 'pass_tds_avg_3': 1.8, 'rush_yards_avg_3': 15,
            'years_exp': 5, 'games_played': 50, 'fp_std_5': 5.0
        },
        'RB': {
            'fp_avg_3': 12.0, 'fp_avg_5': 11.8, 'fp_trend_3': 0.0,
            'rush_yards_avg_3': 65, 'rush_tds_avg_3': 0.6, 'targets_avg_3': 3.5,
            'years_exp': 4, 'games_played': 40, 'fp_std_5': 4.5
        },
        'WR': {
            'fp_avg_3': 11.0, 'fp_avg_5': 10.8, 'fp_trend_3': 0.0,
            'targets_avg_3': 6.5, 'receptions_avg_3': 4.2, 'receiving_yards_avg_3': 58,
            'years_exp': 4, 'games_played': 45, 'fp_std_5': 4.8
        },
        'TE': {
            'fp_avg_3': 8.5, 'fp_avg_5': 8.2, 'fp_trend_3': 0.0,
            'targets_avg_3': 4.8, 'receptions_avg_3': 3.2, 'receiving_yards_avg_3': 38,
            'td_rate_avg_5': 0.12, 'snap_pct_avg_3': 0.65, 'target_share_avg_3': 0.15,
            'years_exp': 4, 'games_played': 45, 'fp_std_5': 3.8
        }
    }
    
    return defaults.get(position, {}).get(feature, 0.0)

def estimate_salary_expectation(salary, position):
    """Estimate expected fantasy points based on salary"""
    
    # Rough salary-to-points expectations by position
    salary_multipliers = {
        'QB': 0.0025,  # $8000 salary ≈ 20 points
        'RB': 0.0022,  # $7000 salary ≈ 15.4 points  
        'WR': 0.0021,  # $7000 salary ≈ 14.7 points
        'TE': 0.0018   # $6000 salary ≈ 10.8 points
    }
    
    return salary * salary_multipliers.get(position, 0.002)

def print_projection_summary(projections_df):
    """Print formatted projection summary"""
    
    print("\n🎯 ELITE MODEL PROJECTIONS")
    print("=" * 80)
    
    # Sort by projection descending
    projections_df = projections_df.sort_values('projection', ascending=False)
    
    for _, proj in projections_df.iterrows():
        edge_indicator = "🔥" if proj['edge'] > 3 else "⬆️" if proj['edge'] > 1 else "➡️" if proj['edge'] > -1 else "⬇️"
        
        print(f"{edge_indicator} {proj['name']} ({proj['position']}) - {proj['team']}")
        print(f"   Projection: {proj['projection']:.1f} | Salary: ${proj['salary']:,} | Edge: {proj['edge']:+.1f}")
        
        if proj['missing_features'] > 0:
            print(f"   ⚠️ {proj['missing_features']} features estimated")
        print()
    
    # Summary stats
    print("📊 PROJECTION SUMMARY")
    print("-" * 40)
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_projs = projections_df[projections_df['position'] == position]
        if len(pos_projs) > 0:
            avg_proj = pos_projs['projection'].mean()
            avg_edge = pos_projs['edge'].mean()
            print(f"{position}: Avg Proj {avg_proj:.1f} | Avg Edge {avg_edge:+.1f}")
    
    # Top edges
    top_edges = projections_df.nlargest(3, 'edge')
    print(f"\n🚀 TOP EDGES:")
    for _, proj in top_edges.iterrows():
        print(f"   {proj['name']} ({proj['position']}): +{proj['edge']:.1f} points")

def main():
    """Main testing function"""
    print("🏈 ELITE DFS MODEL TESTING")
    print("=" * 50)
    
    # Load all models
    models = load_all_models()
    
    if len(models) == 0:
        print("❌ No models loaded. Train models first!")
        return
    
    print(f"\n✅ Loaded {len(models)}/4 elite models")
    
    # Create sample player data
    players_df = create_sample_player_data()
    print(f"✅ Created {len(players_df)} sample players")
    
    # Generate projections
    projections_df = generate_projections(models, players_df)
    
    if len(projections_df) == 0:
        print("❌ No projections generated")
        return
    
    # Print results
    print_projection_summary(projections_df)
    
    # Model coverage report
    positions_with_models = set(models.keys())
    positions_in_data = set(players_df['position'].unique())
    
    print(f"\n📋 MODEL COVERAGE")
    print("-" * 30)
    print(f"Positions with models: {sorted(positions_with_models)}")
    print(f"Sample data positions: {sorted(positions_in_data)}")
    
    coverage = len(positions_with_models & positions_in_data) / len(positions_in_data)
    print(f"Coverage: {coverage:.1%}")
    
    if coverage == 1.0:
        print("🎉 COMPLETE MODEL COVERAGE ACHIEVED!")
        print("\nNext steps:")
        print("1. Run real lineup optimization")
        print("2. Implement live slate loading")  
        print("3. Add backtesting framework")

if __name__ == "__main__":
    main()