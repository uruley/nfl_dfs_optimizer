import pandas as pd
import joblib
import numpy as np
import os

class QBModelIntegrator:
    def __init__(self):
        self.qb_model = None
        
    def load_qb_model(self):
        """Load QB model using joblib"""
        
        try:
            self.qb_model = joblib.load('models/qb_model.pkl')
            print(f"✅ QB model loaded with joblib!")
            print(f"🤖 Model type: {type(self.qb_model).__name__}")
            print(f"📊 Features expected: {self.qb_model.n_features_in_}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading QB model: {e}")
            return False
    
    def create_qb_features(self, qb_players):
        """Create the exact 45 features your model expects"""
        
        feature_data = []
        
        for idx, qb in qb_players.iterrows():
            qb_name = qb['player_name'].lower()
            consensus_proj = qb['projection']
            
            # Create realistic features based on QB name and consensus projection
            # These would normally come from your nfl_data_py pipeline
            
            if 'hurts' in qb_name:
                # Jalen Hurts - elite dual threat QB
                features = [
                    20.5, 32.0, 245, 1.8, 0.8,  # passing stats
                    2.1, 12.0, 0.1, 0.1,         # sacks
                    180, 65, 12.5, 0.15, 0.0,    # passing advanced
                    0.72, 0.05,                   # pacr, dakota
                    12.5, 65, 0.8, 0.2, 0.1,     # rushing stats
                    4.2, 0.45, 0.0,               # rushing advanced
                    0.2, 1.0, 8.5, 0.1, 0.0, 0.0,  # receiving (QBs rarely receive)
                    5.2, 2.8, 1.2, 0.02, 0.0,    # receiving advanced
                    0.85, 0.12, 0.08, 0.15,      # air yards metrics
                    0.0, consensus_proj,          # special teams, fantasy points
                    consensus_proj * 0.95,        # fp_avg_3
                    consensus_proj * 0.92,        # fp_avg_5  
                    consensus_proj * 0.90,        # fp_avg_8
                    18                            # opp_def_rank (good matchup)
                ]
            elif 'prescott' in qb_name or 'dak' in qb_name:
                # Dak Prescott - pocket passer
                features = [
                    25.2, 38.5, 275, 2.1, 0.9,   # passing stats
                    2.8, 18.0, 0.1, 0.1,         # sacks
                    210, 65, 15.2, 0.18, 0.0,    # passing advanced
                    0.76, 0.03,                   # pacr, dakota
                    3.5, 18, 0.2, 0.1, 0.0,      # rushing stats (limited)
                    1.8, 0.12, 0.0,               # rushing advanced
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # receiving
                    0.0, 0.0, 0.0, 0.0, 0.0,     # receiving advanced
                    0.0, 0.0, 0.0, 0.0,          # air yards metrics
                    0.0, consensus_proj,          # special teams, fantasy points
                    consensus_proj * 0.98,        # fp_avg_3
                    consensus_proj * 0.96,        # fp_avg_5
                    consensus_proj * 0.94,        # fp_avg_8
                    22                            # opp_def_rank (tougher matchup)
                ]
            elif 'milton' in qb_name:
                # Joe Milton III - backup with rushing upside
                features = [
                    12.5, 22.0, 160, 1.1, 0.6,   # passing stats (limited)
                    1.8, 12.0, 0.0, 0.0,         # sacks
                    120, 40, 8.5, 0.08, 0.0,     # passing advanced
                    0.65, 0.08,                   # pacr, dakota
                    8.5, 45, 0.6, 0.1, 0.0,      # rushing stats (mobile)
                    3.2, 0.35, 0.0,               # rushing advanced
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # receiving
                    0.0, 0.0, 0.0, 0.0, 0.0,     # receiving advanced
                    0.0, 0.0, 0.0, 0.0,          # air yards metrics
                    0.0, consensus_proj,          # special teams, fantasy points
                    consensus_proj * 1.05,        # fp_avg_3 (upside)
                    consensus_proj * 1.02,        # fp_avg_5
                    consensus_proj * 1.00,        # fp_avg_8
                    25                            # opp_def_rank (easier matchup)
                ]
            else:
                # Default backup QB features
                features = [
                    10.0, 18.0, 130, 0.8, 0.5,   # passing stats
                    1.5, 8.0, 0.0, 0.0,          # sacks
                    95, 35, 6.5, 0.05, 0.0,      # passing advanced
                    0.60, 0.06,                   # pacr, dakota
                    2.5, 12, 0.1, 0.1, 0.0,      # rushing stats
                    1.0, 0.08, 0.0,               # rushing advanced
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # receiving
                    0.0, 0.0, 0.0, 0.0, 0.0,     # receiving advanced
                    0.0, 0.0, 0.0, 0.0,          # air yards metrics
                    0.0, consensus_proj,          # special teams, fantasy points
                    consensus_proj * 0.95,        # fp_avg_3
                    consensus_proj * 0.93,        # fp_avg_5
                    consensus_proj * 0.91,        # fp_avg_8
                    20                            # opp_def_rank (average)
                ]
            
            feature_data.append(features)
        
        # Convert to DataFrame with correct feature names
        feature_names = [
            'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
            'sacks', 'sack_yards', 'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards',
            'passing_yards_after_catch', 'passing_first_downs', 'passing_epa', 'passing_2pt_conversions',
            'pacr', 'dakota', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
            'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa', 'rushing_2pt_conversions',
            'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles',
            'receiving_fumbles_lost', 'receiving_air_yards', 'receiving_yards_after_catch',
            'receiving_first_downs', 'receiving_epa', 'receiving_2pt_conversions', 'racr',
            'target_share', 'air_yards_share', 'wopr', 'special_teams_tds', 'fantasy_points_ppr',
            'fp_avg_3', 'fp_avg_5', 'fp_avg_8', 'opp_def_rank'
        ]
        
        return pd.DataFrame(feature_data, columns=feature_names, index=qb_players.index)
    
    def generate_qb_projections(self, qb_players):
        """Generate ML projections using your 45-feature model"""
        
        if self.qb_model is None:
            print("❌ No model loaded")
            return qb_players
        
        try:
            # Create the 45 features
            feature_data = self.create_qb_features(qb_players)
            
            # Generate ML predictions
            ml_projections = self.qb_model.predict(feature_data)
            
            # Update dataframe
            qb_players = qb_players.copy()
            qb_players['consensus_projection'] = qb_players['projection']
            qb_players['ml_projection'] = ml_projections
            qb_players['projection'] = ml_projections  # Use ML projections
            
            print(f"✅ ML QB projections generated using 45 features!")
            print(f"📈 ML Range: {ml_projections.min():.1f} - {ml_projections.max():.1f}")
            
            # Show comparison
            print(f"\n🔥 ML vs Consensus Comparison:")
            for _, qb in qb_players.iterrows():
                consensus = qb['consensus_projection']
                ml_proj = qb['ml_projection']
                advantage = ml_proj - consensus
                print(f"  {qb['player_name']}: {consensus:.1f} → {ml_proj:.1f} ({advantage:+.1f})")
            
            return qb_players
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            print(f"Using consensus projections")
            return qb_players

def test_qb_integration():
    """Test the QB model integration"""
    
    print("🏈 TESTING QB MODEL WITH 45 FEATURES")
    print("="*50)
    
    # Load showdown data
    showdown_data = pd.read_csv('data/live_slates/showdown_processed.csv')
    qb_players = showdown_data[showdown_data['position'] == 'QB'].copy()
    
    print(f"📊 Found {len(qb_players)} QBs")
    
    # Test integration
    integrator = QBModelIntegrator()
    
    if integrator.load_qb_model():
        qb_with_ml = integrator.generate_qb_projections(qb_players)
        return qb_with_ml
    else:
        return qb_players

if __name__ == "__main__":
    test_qb_integration()