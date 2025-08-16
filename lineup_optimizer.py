import pandas as pd
import numpy as np
from pulp import *
import warnings
warnings.filterwarnings('ignore')

class SimpleDraftKingsOptimizer:
    """
    Simplified working DraftKings optimizer - no FLEX complications
    """
    
    def __init__(self):
        self.salary_cap = 50000
        
    def optimize_lineup(self, player_pool: pd.DataFrame):
        """
        Simple optimization: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX (any RB/WR/TE), 1 DST
        """
        
        print(f"🔄 Optimizing lineup with {len(player_pool)} players...")
        
        # Debug: show what positions we have
        pos_counts = player_pool['position'].value_counts()
        print(f"📊 Available positions: {pos_counts.to_dict()}")
        
        # Create problem
        prob = LpProblem("DraftKings_Simple", LpMaximize)
        
        # Create variables for each player
        player_vars = {}
        for idx, player in player_pool.iterrows():
            player_vars[idx] = LpVariable(f"player_{idx}", cat='Binary')
        
        # Objective: maximize points
        prob += lpSum([player_pool.loc[idx, 'projection'] * player_vars[idx] for idx in player_vars])
        
        # Constraint 1: Salary cap
        prob += lpSum([player_pool.loc[idx, 'salary'] * player_vars[idx] for idx in player_vars]) <= self.salary_cap
        
        # Constraint 2: Exactly 1 QB
        qb_indices = player_pool[player_pool['position'] == 'QB'].index
        if len(qb_indices) > 0:
            prob += lpSum([player_vars[idx] for idx in qb_indices]) == 1
        
        # Constraint 3: At least 2 RBs, at most 4 RBs (2 starters + up to 2 FLEX)
        rb_indices = player_pool[player_pool['position'] == 'RB'].index
        if len(rb_indices) > 0:
            prob += lpSum([player_vars[idx] for idx in rb_indices]) >= 2
            prob += lpSum([player_vars[idx] for idx in rb_indices]) <= 4
        
        # Constraint 4: At least 3 WRs, at most 5 WRs (3 starters + up to 2 FLEX)
        wr_indices = player_pool[player_pool['position'] == 'WR'].index
        if len(wr_indices) > 0:
            prob += lpSum([player_vars[idx] for idx in wr_indices]) >= 3
            prob += lpSum([player_vars[idx] for idx in wr_indices]) <= 5
        
        # Constraint 5: At least 1 TE, at most 3 TEs (1 starter + up to 2 FLEX)
        te_indices = player_pool[player_pool['position'] == 'TE'].index
        if len(te_indices) > 0:
            prob += lpSum([player_vars[idx] for idx in te_indices]) >= 1
            prob += lpSum([player_vars[idx] for idx in te_indices]) <= 3
        
        # Constraint 6: Exactly 1 DST
        dst_indices = player_pool[player_pool['position'] == 'DST'].index
        if len(dst_indices) > 0:
            prob += lpSum([player_vars[idx] for idx in dst_indices]) == 1
        
        # Constraint 7: Total exactly 9 players
        prob += lpSum([player_vars[idx] for idx in player_vars]) == 9
        
        # Solve
        try:
            prob.solve(PULP_CBC_CMD(msg=0))
            
            if prob.status == 1:  # Optimal
                # Extract lineup
                selected = []
                for idx in player_vars:
                    if player_vars[idx].value() == 1:
                        selected.append(idx)
                
                lineup = player_pool.loc[selected].copy()
                total_salary = lineup['salary'].sum()
                total_projection = lineup['projection'].sum()
                
                print(f"✅ Lineup optimized!")
                print(f"   Total projection: {total_projection:.1f}")
                print(f"   Total salary: ${total_salary:,}")
                print(f"   Players: {', '.join(lineup['player_name'].tolist())}")
                
                return lineup
            else:
                print(f"❌ Optimization failed: {LpStatus[prob.status]}")
                self.debug_constraints(player_pool)
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error during optimization: {e}")
            return pd.DataFrame()
    
    def debug_constraints(self, player_pool):
        """Debug why optimization is failing"""
        
        print(f"\n🔍 DEBUGGING CONSTRAINTS:")
        
        pos_counts = player_pool['position'].value_counts()
        print(f"Position counts: {pos_counts.to_dict()}")
        
        # Check if we have minimum required players
        requirements = {'QB': 1, 'RB': 3, 'WR': 4, 'TE': 2, 'DST': 1}  # Need extra for FLEX
        
        for pos, min_needed in requirements.items():
            available = pos_counts.get(pos, 0)
            status = "✅" if available >= min_needed else "❌"
            print(f"   {pos}: {available} available, {min_needed} needed {status}")
        
        # Check salary constraints
        cheapest_lineup_cost = (
            player_pool[player_pool['position'] == 'QB']['salary'].min() +
            player_pool[player_pool['position'] == 'RB']['salary'].nsmallest(3).sum() +  # 2 RB + 1 FLEX
            player_pool[player_pool['position'] == 'WR']['salary'].nsmallest(4).sum() +  # 3 WR + 1 FLEX  
            player_pool[player_pool['position'] == 'TE']['salary'].nsmallest(2).sum() +  # 1 TE + 1 FLEX
            player_pool[player_pool['position'] == 'DST']['salary'].min()
        )
        
        print(f"   Cheapest possible lineup: ${cheapest_lineup_cost:,}")
        print(f"   Salary cap: ${self.salary_cap:,}")
        print(f"   Under cap: {'✅' if cheapest_lineup_cost <= self.salary_cap else '❌'}")

def test_simple_optimizer():
    """Test with corrected sample data"""
    
    print("🏈 Testing Simple DraftKings Optimizer")
    print("=" * 50)
    
    # Create sample data with enough players for FLEX
    sample_players = pd.DataFrame({
        'player_name': [
            # QBs (1 needed)
            'Josh Allen', 'Lamar Jackson', 'Patrick Mahomes',
            # RBs (3 needed: 2 starters + 1 FLEX option)  
            'Christian McCaffrey', 'Saquon Barkley', 'Josh Jacobs', 'Aaron Jones', 'Tony Pollard',
            # WRs (4 needed: 3 starters + 1 FLEX option)
            'Stefon Diggs', 'Davante Adams', 'Tyreek Hill', 'Mike Evans', 'Chris Godwin', 'CeeDee Lamb',
            # TEs (2 needed: 1 starter + 1 FLEX option)
            'Travis Kelce', 'Mark Andrews', 'T.J. Hockenson',
            # DSTs (1 needed)
            'Bills DST', 'Ravens DST'
        ],
        'position': [
            'QB', 'QB', 'QB',
            'RB', 'RB', 'RB', 'RB', 'RB', 
            'WR', 'WR', 'WR', 'WR', 'WR', 'WR',
            'TE', 'TE', 'TE',
            'DST', 'DST'
        ],
        'salary': [
            8200, 7800, 8000,  # QBs
            8800, 7600, 6200, 5400, 4800,  # RBs (reduced expensive ones)
            7200, 6800, 6400, 5800, 5200, 4600,  # WRs (reduced)
            6800, 5400, 4200,  # TEs (reduced)
            2800, 2400  # DSTs
        ],
        'projection': [
            23.5, 22.1, 21.8,  # QBs
            21.2, 18.5, 16.8, 15.2, 13.8,  # RBs
            16.1, 15.8, 15.2, 14.9, 13.5, 14.2,  # WRs
            13.8, 11.2, 9.8,  # TEs
            9.2, 8.1  # DSTs
        ],
        'team': [
            'BUF', 'BAL', 'KC',
            'SF', 'NYG', 'LV', 'MIN', 'DAL',
            'BUF', 'LV', 'MIA', 'TB', 'TB', 'DAL', 
            'KC', 'BAL', 'MIN',
            'BUF', 'BAL'
        ]
    })
    
    print(f"📊 Created sample player pool: {len(sample_players)} players")
    
    # Test optimization
    optimizer = SimpleDraftKingsOptimizer()
    lineup = optimizer.optimize_lineup(sample_players)
    
    if not lineup.empty:
        print(f"\n📋 GENERATED LINEUP:")
        lineup_display = lineup[['player_name', 'position', 'salary', 'projection']].copy()
        lineup_display = lineup_display.sort_values('position')
        print(lineup_display.to_string(index=False))
        
        print(f"\n✅ SUCCESS! Optimizer is working correctly.")
        return True
    else:
        print(f"\n❌ FAILED! Still having issues.")
        return False

if __name__ == "__main__":
    success = test_simple_optimizer()
    
    if success:
        print(f"\n🚀 Next steps:")
        print(f"1. Integration test passed")
        print(f"2. Ready to load your QB projections")
        print(f"3. Generate real lineups!")