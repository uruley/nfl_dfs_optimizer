import pandas as pd
import sys
import os
from pulp import *
sys.path.append('Scripts')
from live_qb_features import create_live_qb_features

def generate_ml_powered_showdown():
    """
    Generate showdown lineup using your ML QB projections
    """
    
    print("🏈 ML-POWERED SHOWDOWN LINEUP GENERATION")
    print("="*60)
    
    # Load base showdown data
    showdown_data = pd.read_csv('data/live_slates/showdown_processed.csv')
    print(f"✅ Loaded {len(showdown_data)} players")
    
    # Get ML QB projections
    print(f"\n🤖 Generating ML QB projections...")
    qb_features, ml_projections = create_live_qb_features()
    
    # Update QB projections in showdown data
    qb_indices = showdown_data[showdown_data['position'] == 'QB'].index
    
    for i, idx in enumerate(qb_indices):
        old_proj = showdown_data.loc[idx, 'projection']
        new_proj = ml_projections[i]
        showdown_data.loc[idx, 'consensus_projection'] = old_proj
        showdown_data.loc[idx, 'projection'] = new_proj
        
        qb_name = showdown_data.loc[idx, 'player_name']
        advantage = new_proj - old_proj
        print(f"  📈 {qb_name}: {old_proj:.1f} → {new_proj:.1f} ({advantage:+.1f})")
    
    # Generate lineup with ML projections
    lineup = optimize_showdown_with_ml(showdown_data)
    
    if not lineup.empty:
        print(f"\n📋 ML-POWERED SHOWDOWN LINEUP:")
        display_cols = ['player_name', 'role', 'position', 'team', 'actual_salary', 'fantasy_points']
        print(lineup[display_cols].to_string(index=False))
        
        total_salary = lineup['actual_salary'].sum()
        total_points = lineup['fantasy_points'].sum()
        
        print(f"\n💰 Total Salary: ${total_salary:,.0f}")
        print(f"🎯 Total Points: {total_points:.1f}")
        print(f"💸 Remaining: ${50000 - total_salary:,.0f}")
        
        # Show the ML edge
        print(f"\n🔥 YOUR ML EDGE:")
        qb_in_lineup = lineup[lineup['position'] == 'QB']
        if not qb_in_lineup.empty:
            qb_name = qb_in_lineup.iloc[0]['player_name']
            qb_role = qb_in_lineup.iloc[0]['role']
            qb_proj = qb_in_lineup.iloc[0]['fantasy_points']
            
            # Find consensus projection
            qb_match = showdown_data[showdown_data['player_name'] == qb_name]
            if not qb_match.empty:
                consensus = qb_match.iloc[0]['consensus_projection']
                ml_proj = qb_match.iloc[0]['projection']
                
                if qb_role == 'Captain':
                    consensus_capt = consensus * 1.5
                    ml_capt = ml_proj * 1.5
                    advantage = ml_capt - consensus_capt
                    print(f"  🎯 {qb_name} ({qb_role}): Consensus={consensus_capt:.1f}, ML={ml_capt:.1f} (+{advantage:.1f})")
                else:
                    advantage = ml_proj - consensus
                    print(f"  🎯 {qb_name} ({qb_role}): Consensus={consensus:.1f}, ML={ml_proj:.1f} (+{advantage:.1f})")
        
        print(f"\n🏆 SUCCESS! ML-powered lineup generated!")
        
        return lineup
    else:
        print("❌ ML optimization failed")
        return pd.DataFrame()

def optimize_showdown_with_ml(player_pool):
    """Showdown optimizer using ML projections"""
    
    # Create optimization problem
    prob = LpProblem("ML_Showdown", LpMaximize)
    
    # Variables
    captain_vars = {}
    flex_vars = {}
    
    for idx, player in player_pool.iterrows():
        captain_vars[idx] = LpVariable(f"captain_{idx}", cat='Binary')
        flex_vars[idx] = LpVariable(f"flex_{idx}", cat='Binary')
    
    # Objective: Maximize ML-projected points
    objective = []
    for idx, player in player_pool.iterrows():
        ml_projection = player['projection']  # Now using ML projections!
        
        # Captain gets 1.5x points
        objective.append(ml_projection * 1.5 * captain_vars[idx])
        # FLEX gets 1x points  
        objective.append(ml_projection * flex_vars[idx])
    
    prob += lpSum(objective)
    
    # Constraints
    prob += lpSum([captain_vars[idx] for idx in captain_vars]) == 1  # 1 captain
    prob += lpSum([flex_vars[idx] for idx in flex_vars]) == 5        # 5 flex
    
    # Player can't be both captain and flex
    for idx in captain_vars:
        prob += captain_vars[idx] + flex_vars[idx] <= 1
    
    # Salary constraint
    salary_constraint = []
    for idx, player in player_pool.iterrows():
        captain_salary = player['cpt_salary']
        flex_salary = player['flex_salary']
        
        salary_constraint.append(captain_salary * captain_vars[idx])
        salary_constraint.append(flex_salary * flex_vars[idx])
    
    prob += lpSum(salary_constraint) <= 50000
    
    # Solve
    try:
        prob.solve(PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:  # Optimal
            # Extract lineup
            captain_idx = None
            flex_indices = []
            
            for idx in captain_vars:
                if captain_vars[idx].value() == 1:
                    captain_idx = idx
                if flex_vars[idx].value() == 1:
                    flex_indices.append(idx)
            
            # Build lineup
            lineup_data = []
            
            if captain_idx is not None:
                captain = player_pool.loc[captain_idx].copy()
                captain['role'] = 'Captain'
                captain['fantasy_points'] = captain['projection'] * 1.5
                captain['actual_salary'] = captain['cpt_salary']
                lineup_data.append(captain)
            
            for idx in flex_indices:
                flex_player = player_pool.loc[idx].copy()
                flex_player['role'] = 'FLEX'
                flex_player['fantasy_points'] = flex_player['projection']
                flex_player['actual_salary'] = flex_player['flex_salary']
                lineup_data.append(flex_player)
            
            return pd.DataFrame(lineup_data)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    generate_ml_powered_showdown()