import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import sys
import os

def load_your_qb_model():
    """Load your trained QB model with better error handling"""
    
    model_path = 'models/qb_model.pkl'
    
    try:
        # Try standard pickle loading first
        with open(model_path, 'rb') as f:
            qb_model = pickle.load(f)
        print(f"✅ QB model loaded from {model_path}")
        return qb_model
        
    except Exception as e:
        print(f"⚠️  Pickle error: {e}")
        
        # Try joblib if pickle fails
        try:
            import joblib
            qb_model = joblib.load(model_path.replace('.pkl', '.joblib'))
            print(f"✅ QB model loaded with joblib")
            return qb_model
        except:
            pass
        
        # Try loading with different pickle protocol
        try:
            import pickle5
            with open(model_path, 'rb') as f:
                qb_model = pickle5.load(f)
            print(f"✅ QB model loaded with pickle5")
            return qb_model
        except:
            pass
    
    print(f"❌ Could not load QB model - using fallback projections")
    return None

def create_week1_qb_data():
    """Create Week 1 QB data with your feature structure"""
    
    # This simulates Week 1 2024 data - replace with your actual data loading
    week1_qbs = pd.DataFrame({
        'player_name': [
            'Josh Allen', 'Lamar Jackson', 'Patrick Mahomes', 'Josh Jacobs',
            'Tua Tagovailoa', 'Dak Prescott', 'Russell Wilson', 'Kyler Murray',
            'Geno Smith', 'Derek Carr', 'Kirk Cousins', 'Jared Goff',
            'Brock Purdy', 'C.J. Stroud', 'Anthony Richardson', 'Caleb Williams'
        ],
        'team': [
            'BUF', 'BAL', 'KC', 'LV', 'MIA', 'DAL', 'PIT', 'ARI',
            'SEA', 'NO', 'ATL', 'DET', 'SF', 'HOU', 'IND', 'CHI'
        ],
        'salary': [
            8400, 8100, 8300, 7800, 7200, 7500, 7000, 7100,
            6600, 6800, 6400, 6700, 7600, 6300, 6000, 6100
        ],
        # Add your feature columns here - these should match your training features
        'fantasy_points_3game_avg': [24.2, 22.1, 21.8, 20.5, 18.9, 19.2, 17.3, 18.1, 16.8, 17.9, 16.2, 18.4, 19.6, 15.2, 14.1, 13.8],
        'fantasy_points_5game_avg': [23.8, 21.9, 22.1, 19.8, 19.1, 18.9, 17.8, 17.6, 17.2, 17.4, 16.8, 17.9, 19.2, 14.8, 13.9, 14.2],
        'passing_yards_3game_avg': [285, 245, 275, 260, 240, 265, 220, 235, 215, 250, 230, 245, 270, 210, 195, 200],
        'passing_tds_3game_avg': [2.3, 1.8, 2.1, 1.9, 1.6, 1.9, 1.4, 1.7, 1.3, 1.8, 1.5, 1.7, 2.0, 1.2, 1.1, 1.2],
        'rushing_yards_3game_avg': [45, 65, 25, 35, 15, 20, 25, 55, 18, 22, 12, 15, 28, 30, 40, 35],
    })
    
    return week1_qbs

def generate_qb_projections(qb_model, qb_data):
    """Generate QB projections using your model"""
    
    if qb_model is None:
        # Fallback: use simple projections
        qb_data['projection'] = np.random.uniform(18, 26, len(qb_data))
        print("⚠️ Using random projections (model not loaded)")
    else:
        # Use your actual model
        try:
            # Get feature columns (exclude non-feature columns)
            feature_cols = [col for col in qb_data.columns 
                           if col not in ['player_name', 'team', 'salary']]
            
            feature_data = qb_data[feature_cols].fillna(0)
            predictions = qb_model.predict(feature_data)
            qb_data['projection'] = predictions
            
            print(f"✅ Generated ML QB projections!")
            print(f"📈 Range: {predictions.min():.1f} - {predictions.max():.1f} points")
            
        except Exception as e:
            print(f"❌ Error using QB model: {e}")
            qb_data['projection'] = np.random.uniform(18, 26, len(qb_data))
    
    qb_data['position'] = 'QB'
    return qb_data

def add_other_positions():
    """Add other positions with consensus projections"""
    
    other_players = pd.DataFrame({
        'player_name': [
            # RBs
            'Christian McCaffrey', 'Saquon Barkley', 'Josh Jacobs', 'Aaron Jones', 'Derrick Henry', 'Tony Pollard', 'Najee Harris', 'Austin Ekeler',
            # WRs
            'Stefon Diggs', 'Davante Adams', 'Tyreek Hill', 'Mike Evans', 'Chris Godwin', 'CeeDee Lamb', 'A.J. Brown', 'DK Metcalf', 'Cooper Kupp', 'Ja\'Marr Chase',
            # TEs
            'Travis Kelce', 'Mark Andrews', 'T.J. Hockenson', 'Kyle Pitts', 'George Kittle',
            # DSTs
            'Bills DST', 'Ravens DST', '49ers DST', 'Cowboys DST'
        ],
        'position': (
            ['RB'] * 8 + ['WR'] * 10 + ['TE'] * 5 + ['DST'] * 4
        ),
        'salary': [
            # RBs
            7200, 6600, 6000, 5400, 5800, 4800, 4400, 4000,
            # WRs
            6200, 5800, 5600, 5200, 4800, 5400, 5000, 4600, 4400, 5600,
            # TEs
            5400, 4600, 4000, 3600, 4200,
            # DSTs
            2400, 2200, 2000, 1800
        ],
        'projection': [
            # RBs
            22.5, 19.8, 18.1, 16.4, 18.7, 14.8, 14.2, 13.5,
            # WRs
            17.5, 17.1, 16.8, 16.2, 15.1, 16.9, 16.5, 15.8, 15.4, 16.7,
            # TEs
            15.2, 12.5, 11.8, 11.2, 13.1,
            # DSTs
            10.5, 9.8, 9.2, 8.5
        ],
        'team': [
            # RBs
            'SF', 'NYG', 'LV', 'MIN', 'BAL', 'DAL', 'PIT', 'LAC',
            # WRs
            'BUF', 'LV', 'MIA', 'TB', 'TB', 'DAL', 'PHI', 'SEA', 'LAR', 'CIN',
            # TEs
            'KC', 'BAL', 'MIN', 'ATL', 'SF',
            # DSTs
            'BUF', 'BAL', 'SF', 'DAL'
        ]
    })
    
    return other_players

def create_full_player_pool():
    """Create complete player pool with your QB projections"""
    
    print("🏈 Creating Week 1 Player Pool with Your QB Edge!")
    print("="*60)
    
    # Load your QB model
    qb_model = load_your_qb_model()
    
    # Create QB data
    qb_data = create_week1_qb_data()
    qb_projections = generate_qb_projections(qb_model, qb_data)
    
    # Add other positions
    other_players = add_other_positions()
    
    # Combine everything
    full_pool = pd.concat([qb_projections, other_players], ignore_index=True)
    
    # Save for optimizer
    output_file = 'data/processed/week1_player_pool_with_qb_model.csv'
    full_pool.to_csv(output_file, index=False)
    
    print(f"\n📊 FULL PLAYER POOL CREATED:")
    print(f"   Total players: {len(full_pool)}")
    print(f"   Positions: {full_pool['position'].value_counts().to_dict()}")
    print(f"   Your QB advantage: ML projections vs consensus!")
    print(f"   Saved to: {output_file}")
    
    # Show your top QB projections
    top_qbs = qb_projections.nlargest(5, 'projection')[['player_name', 'salary', 'projection']]
    print(f"\n🏆 Your Top 5 QB Projections:")
    print(top_qbs.to_string(index=False))
    
    return output_file

if __name__ == "__main__":
    create_full_player_pool()