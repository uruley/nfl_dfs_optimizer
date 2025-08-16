# dfs_ui.py
# Professional DFS Lineup Generator UI
# Integrates with your 4 elite ML models

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import time

sys.path.append('.')  # Add current directory to path
from ml_model_integration import integrate_ml_with_ui

# Page config
st.set_page_config(
    page_title="NFL DFS Optimizer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-box {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_models():
    """Load all 4 ML models"""
    models = {}
    model_info = {}
    
    positions = ['qb', 'rb', 'wr', 'te']
    
    for pos in positions:
        try:
            model_path = f'models/{pos}_model_proper.pkl'
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                models[pos] = model_data
                
                # Extract model info
                if isinstance(model_data, dict):
                    if 'performance' in model_data:
                        perf = model_data['performance']
                        mae = perf.get('mae', 'N/A')
                        corr = perf.get('correlation', 'N/A')
                        model_info[pos] = {'mae': mae, 'correlation': corr}
                    else:
                        # For models without performance data
                        model_info[pos] = {'mae': 'Trained', 'correlation': 'Available'}
                
        except Exception as e:
            st.error(f"Error loading {pos.upper()} model: {e}")
    
    return models, model_info

def detect_contest_type(df):
    """Auto-detect contest type from DraftKings CSV"""
    
    # Check salary cap
    if 'Salary' in df.columns:
        max_salary = df['Salary'].max()
        if max_salary > 15000:  # Showdown captain pricing
            return "Showdown"
        elif max_salary < 12000:  # Lower salary caps
            return "Primetime/TNF"
    
    # Check roster construction
    if 'Roster Position' in df.columns:
        positions = df['Roster Position'].unique()
        if 'CPT' in str(positions) or 'FLEX' in str(positions):
            return "Showdown"
    
    # Check number of players
    if len(df) < 20:
        return "Showdown"
    elif len(df) < 50:
        return "Primetime/TNF"
    
    return "Main Slate"

def predict_player_score(player_row, position, models):
    """Generate ML prediction for a single player"""
    
    # DEBUG - see all player data
    st.write(f"DEBUG: {player_row}")
    
    pos_key = position.lower()
    if pos_key not in models:
        return None
    
    model_data = models[pos_key]
    
    try:
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Get feature columns
        if 'feature_columns' in model_data:
            feature_cols = model_data['feature_columns']
        elif 'features' in model_data:
            feature_cols = model_data['features']
        else:
            return None
        
        # Create realistic features based on position
        features = create_realistic_features(player_row, position, feature_cols)
        
        # Scale and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        
        return max(0, prediction)  # Ensure non-negative
        
    except Exception as e:
        st.error(f"Prediction error for {player_row.get('Name', 'Unknown')}: {e}")
        return None

def create_realistic_features(player_row, position, feature_cols):
    """Create realistic feature values for prediction"""
    
    # Base feature values by position
    feature_defaults = {
        'qb': {
            'fp_avg_3': 22.5, 'fp_trend_3': 1.2, 'years_exp': 6,
            'fp_avg_5': 21.8, 'fp_avg_8': 21.2, 'fp_consistency_8': 4.2,
            'passing_epa_avg_5': 0.15, 'air_yards_per_attempt_avg_5': 8.5,
            'completions_avg_3': 22.0, 'carries_avg_8': 6.5
        },
        'rb': {
            'fp_avg_3': 15.2, 'fp_trend_3': 0.8, 'years_exp': 4,
            'fp_avg_5': 14.8, 'fp_avg_8': 14.5, 'carries_avg_8': 18.5,
            'fp_consistency_8': 4.8, 'rushing_yards_avg_8': 85.0,
            'target_share_avg_5': 0.12, 'wopr_avg_5': 0.15
        },
        'wr': {
            'fp_avg_3': 13.5, 'fp_trend_3': 0.5, 'years_exp': 5,
            'targets_avg_8': 8.5, 'fp_avg_5': 13.2, 'fp_avg_8': 13.0,
            'target_share_avg_5': 0.22, 'air_yards_share_avg_5': 0.25,
            'catch_rate_avg_5': 0.68, 'yards_per_target_avg_8': 9.2
        },
        'te': {
            'fp_avg_3': 9.5, 'fp_avg_5': 9.2, 'fp_avg_8': 9.0,
            'fp_trend_3': 0.3, 'targets_avg_3': 5.5, 'years_exp': 6,
            'receptions_avg_3': 3.8, 'receiving_yards_avg_3': 42.0
        }
    }
    
    defaults = feature_defaults.get(position.lower(), {})
    
    # Create feature array
    features = []
    for col in feature_cols:
        if col in defaults:
            # Add some salary-based variance
            base_value = defaults[col]
            salary = player_row.get('Salary', 5000)
            
            # Higher salary players get better projections
            if salary > 8000:
                multiplier = 1.2
            elif salary > 6000:
                multiplier = 1.0
            else:
                multiplier = 0.8
            
            features.append(base_value * multiplier)
        else:
            features.append(0.0)
    
    return features

def generate_lineups(slate_df, models, num_lineups=10, strategy="GPP"):
    """Generate optimized lineups"""
    
    # DEBUG - check salary column first
    st.write("DEBUG: Salary column info:")
    st.write(f"Salary column exists: {'Salary' in slate_df.columns}")
    st.write(f"First 5 salary values: {slate_df['Salary'].head().tolist()}")
    st.write(f"Salary data type: {slate_df['Salary'].dtype}")
    
    # Add ML projections
    slate_with_projections = slate_df.copy()
    
    for idx, row in slate_with_projections.iterrows():
        position = row.get('Roster Position', 'FLEX')
        
        # Map DK positions to our model positions
        if position in ['QB']:
            pos_key = 'QB'
        elif position in ['RB', 'RB/WR']:
            pos_key = 'RB'
        elif position in ['WR', 'WR/RB']:
            pos_key = 'WR'
        elif position in ['TE']:
            pos_key = 'TE'
        elif position in ['FLEX']:
            pos_key = 'RB'  # Default FLEX to RB
        else:
            pos_key = 'RB'  # Default
        
        prediction = predict_player_score(row, pos_key, models)
        slate_with_projections.at[idx, 'ML_Projection'] = prediction if prediction else 8.0
        
        # FIX: Use actual salary, not default
        actual_salary = row['Salary']  # Don't use .get() with default
        slate_with_projections.at[idx, 'Value'] = (prediction if prediction else 8.0) / (actual_salary / 1000)
    
    # Simple greedy optimization for demo
    lineups = []
    
    for i in range(num_lineups):
        lineup = select_optimal_lineup(slate_with_projections, strategy)
        if lineup is not None:
            lineups.append(lineup)
    
    return lineups, slate_with_projections

def select_optimal_lineup(slate_df, strategy="GPP"):
    """Select optimal lineup using greedy algorithm"""
    
    available_players = slate_df.copy()
    lineup = []
    total_salary = 0
    salary_cap = 50000
    
    # Position requirements for main slate
    position_needs = {
        'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DST': 1
    }
    
    # Fill required positions
    for pos, count in position_needs.items():
        if pos == 'DST':
            continue  # Skip DST for now
            
        for _ in range(count):
            eligible = available_players[
                (available_players['Roster Position'].str.contains(pos) |
                 ((pos == 'FLEX') & available_players['Roster Position'].isin(['RB', 'WR', 'TE'])))
            ]
            
            if len(eligible) == 0:
                continue
              
            # Sort by value for cash, projection for GPP
            if strategy == "Cash":
                eligible = eligible.sort_values('Value', ascending=False)
            else:
                eligible = eligible.sort_values('ML_Projection', ascending=False)
            
            for _, player in eligible.iterrows():
                if total_salary + player['Salary'] <= salary_cap:
                    lineup.append(player)
                    total_salary += player['Salary']
                    available_players = available_players[available_players['Name'] != player['Name']]
                    break
    
    return lineup if len(lineup) >= 6 else None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏈 NFL DFS Optimizer</h1>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading ML models..."):
        models, model_info = load_models()
    
    # Sidebar - Model Status
    st.sidebar.markdown("## 🤖 Model Status")
    
    for pos, info in model_info.items():
        mae = info.get('mae', 'N/A')
        corr = info.get('correlation', 'N/A')
        
        if isinstance(mae, float):
            status = "🔥 Elite" if mae < 3.0 else "✅ Good"
            st.sidebar.markdown(f"**{pos.upper()}**: {status}")
            st.sidebar.write(f"MAE: {mae:.2f}")
            if isinstance(corr, float):
                st.sidebar.write(f"Correlation: {corr:.3f}")
        else:
            st.sidebar.markdown(f"**{pos.upper()}**: ✅ Loaded")
    
    st.sidebar.markdown("---")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload Slate", "🎯 Generate Lineups", "📊 Analysis"])
    
    with tab1:
        st.subheader("Upload DraftKings CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a DraftKings CSV file",
            type="csv",
            help="Upload your DraftKings contest CSV file"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} players")
                
                # Auto-detect contest type
                contest_type = detect_contest_type(df)
                st.info(f"🎯 Detected Contest Type: **{contest_type}**")
                
                # Store in session state
                st.session_state['slate_df'] = df
                st.session_state['contest_type'] = contest_type
                
                # Preview
                st.subheader("Player Preview")
                st.dataframe(df.head(10))
                
                # Basic stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Players", len(df))
                with col2:
                    if 'Salary' in df.columns:
                        st.metric("Avg Salary", f"${df['Salary'].mean():.0f}")
                with col3:
                    if 'Roster Position' in df.columns:
                        st.metric("Positions", len(df['Roster Position'].unique()))
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with tab2:
        st.subheader("Generate Optimized Lineups")
        
        if 'slate_df' not in st.session_state:
            st.warning("⚠️ Please upload a slate file first")
            return
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            strategy = st.selectbox(
                "Optimization Strategy",
                ["GPP", "Cash"],
                help="GPP: Tournament play, Cash: Safe/consistent"
            )
        
        with col2:
            num_lineups = st.slider(
                "Number of Lineups",
                min_value=1,
                max_value=50,
                value=10,
                help="How many lineups to generate"
            )
        
        if st.button("🚀 Generate Lineups", type="primary"):
            with st.spinner("Generating lineups with ML projections..."):
                try:
                    lineups, slate_with_proj = generate_lineups(
                        st.session_state['slate_df'], 
                        models, 
                        num_lineups, 
                        strategy
                    )
                    
                    if lineups:
                        st.success(f"✅ Generated {len(lineups)} optimized lineups!")
                        
                        # Show all lineups
                        for i, lineup in enumerate(lineups, 1):
                            st.subheader(f"💎 Lineup #{i}")
                            lineup_df = pd.DataFrame(lineup)
                            
                            if 'ML_Projection' in lineup_df.columns:
                                lineup_display = lineup_df[['Name', 'Roster Position', 'Salary', 'ML_Projection', 'Value']].round(2)
                                st.dataframe(lineup_display, use_container_width=True)
                                

                                # Quick stats
                                total_salary = lineup_df['Salary'].sum()
                                total_projection = lineup_df['ML_Projection'].sum()
                                st.write(f"**Total:** ${total_salary:,} salary, {total_projection:.1f} projection")
                            

                            st.write("---")  # Separator between lineups
                        
                    else:
                        st.error("❌ Could not generate any valid lineups")
                        
                except Exception as e:
                    st.error(f"Error generating lineups: {e}")
    
    with tab3:
        st.subheader("📊 Projection Analysis")
        
        if 'slate_with_projections' not in st.session_state:
            st.warning("⚠️ Please generate lineups first")
            return
        
        slate_df = st.session_state['slate_with_projections']
        
        # Top projections by position
        positions = slate_df['Roster Position'].unique()
        
        for pos in sorted(positions):
            if pos in ['QB', 'RB', 'WR', 'TE']:
                pos_players = slate_df[slate_df['Roster Position'] == pos].copy()
                
                if len(pos_players) > 0:
                    # Sort by projection
                    pos_players = pos_players.sort_values('ML_Projection', ascending=False)
                    
                    st.subheader(f"🎯 Top {pos} Projections")
                    
                    display_cols = ['Name', 'Salary', 'ML_Projection', 'Value']
                    available_cols = [col for col in display_cols if col in pos_players.columns]
                    
                    st.dataframe(
                        pos_players[available_cols].head(5).round(2),
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()