import pandas as pd
import sys
import os
from pulp import *

# Add Scripts directory to path
sys.path.append('Scripts')

class ShowdownOptimizer:
    """
    DraftKings Showdown optimizer - 1 Captain + 5 FLEX players
    """
    
    def __init__(self, salary_cap=50000):
        self.salary_cap = salary_cap
        
    def optimize_showdown(self, player_pool):
        """
        Optimize showdown lineup: 1 Captain (1.5x points) + 5 FLEX
        """
        
        print(f"🏈 Optimizing Showdown with {len(player_pool)} players...")
        
        # Debug info
        if 'position' in player_pool.columns:
            pos_counts = player_pool['position'].value_counts()
            print(f"📊 Available positions: {pos_counts.to_dict()}")
        
        # Create optimization problem
        prob = LpProblem("Showdown_Lineup", LpMaximize)
        
        # Create variables
        captain_vars = {}
        flex_vars = {}
        
        for idx, player in player_pool.iterrows():
            captain_vars[idx] = LpVariable(f"captain_{idx}", cat='Binary')
            flex_vars[idx] = LpVariable(f"flex_{idx}", cat='Binary')
        
        # Objective: Maximize points (captain gets 1.5x multiplier)
        objective = []
        for idx, player in player_pool.iterrows():
            projection = player['projection']
            # Captain gets 1.5x points
            objective.append(projection * 1.5 * captain_vars[idx])
            # FLEX gets 1x points
            objective.append(projection * flex_vars[idx])
        
        prob += lpSum(objective)
        
        # CONSTRAINT 1: Exactly 1 captain
        prob += lpSum([captain_vars[idx] for idx in captain_vars]) == 1
        
        # CONSTRAINT 2: Exactly 5 flex players
        prob += lpSum([flex_vars[idx] for idx in flex_vars]) == 5
        
        # CONSTRAINT 3: Player can't be both captain and flex
        for idx in captain_vars:
            prob += captain_vars[idx] + flex_vars[idx] <= 1
        
        # CONSTRAINT 4: Salary cap
        salary_constraint = []
        for idx, player in player_pool.iterrows():
            captain_salary = player['cpt_salary']
            flex_salary = player['flex_salary']
            
            salary_constraint.append(captain_salary * captain_vars[idx])
            salary_constraint.append(flex_salary * flex_vars[idx])
        
        prob += lpSum(salary_constraint) <= self.salary_cap
        
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
                
                # Build lineup dataframe
                lineup_data = []
                
                # Add captain
                if captain_idx is not None:
                    captain = player_pool.loc[captain_idx].copy()
                    captain['role'] = 'Captain'
                    captain['fantasy_points'] = captain['projection'] * 1.5
                    captain['actual_salary'] = captain['cpt_salary']
                    lineup_data.append(captain)
                
                # Add flex players
                for idx in flex_indices:
                    flex_player = player_pool.loc[idx].copy()
                    flex_player['role'] = 'FLEX'
                    flex_player['fantasy_points'] = flex_player['projection']
                    flex_player['actual_salary'] = flex_player['flex_salary']
                    lineup_data.append(flex_player)
                
                lineup = pd.DataFrame(lineup_data)
                
                total_salary = lineup['actual_salary'].sum()
                total_projection = lineup['fantasy_points'].sum()
                
                print(f"✅ Showdown lineup optimized!")
                print(f"   Total projection: {total_projection:.1f}")
                print(f"   Total salary: ${total_salary:,.0f}")
                
                return lineup
                
            else:
                print(f"❌ Optimization failed: {LpStatus[prob.status]}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error during optimization: {e}")
            return pd.DataFrame()

def generate_showdown_lineups():
    """
    Generate showdown lineups using the processed data
    """
    
    print("🏈 GENERATING SHOWDOWN LINEUPS")
    print("="*50)
    
    # Load processed showdown data
    showdown_data = pd.read_csv('data/live_slates/showdown_processed.csv')
    print(f"✅ Loaded {len(showdown_data)} players")
    
    # Show top players
    print(f"\n🏆 Top projected players:")
    top_5 = showdown_data.nlargest(5, 'projection')[['player_name', 'position', 'projection']]
    print(top_5.to_string(index=False))
    
    # Generate lineup
    optimizer = ShowdownOptimizer(salary_cap=50000)
    lineup = optimizer.optimize_showdown(showdown_data)
    
    if not lineup.empty:
        print(f"\n📋 SHOWDOWN LINEUP:")
        display_cols = ['player_name', 'role', 'position', 'team', 'actual_salary', 'fantasy_points']
        print(lineup[display_cols].to_string(index=False))
        
        total_salary = lineup['actual_salary'].sum()
        total_points = lineup['fantasy_points'].sum()
        
        print(f"\n💰 Total Salary: ${total_salary:,.0f}")
        print(f"🎯 Total Points: {total_points:.1f}")
        print(f"💸 Remaining: ${50000 - total_salary:,.0f}")
        
        print(f"\n🏆 SUCCESS! Real DraftKings Showdown lineup generated!")
        
    else:
        print("❌ Showdown optimization failed")

if __name__ == "__main__":
    generate_showdown_lineups()