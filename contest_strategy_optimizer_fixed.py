"""
Contest Strategy & Lineup Builder
Generates optimized lineups for different contest types with ML projections
"""

import pandas as pd
import numpy as np
from pulp import *
import joblib
import warnings
warnings.filterwarnings('ignore')

class ContestStrategyOptimizer:
    def __init__(self):
        """Initialize the contest strategy optimizer"""
        self.contest_configs = {
            'gpp': {
                'name': 'GPP Tournament',
                'objective': 'ceiling',
                'ownership_penalty': 0.3,
                'min_salary_usage': 0.98,
                'stack_bonus': 0.1,
                'contrarian_boost': 0.15
            },
            'cash': {
                'name': 'Cash Game',
                'objective': 'floor',
                'ownership_penalty': 0.0,
                'min_salary_usage': 0.95,
                'stack_bonus': 0.05,
                'contrarian_boost': 0.0
            },
            'showdown_single': {
                'name': 'Single Entry Showdown',
                'objective': 'projection',
                'ownership_penalty': 0.1,
                'min_salary_usage': 0.97,
                'captain_multiplier': 1.5,
                'stack_bonus': 0.08
            },
            'showdown_multi': {
                'name': 'Multi Entry Showdown',
                'objective': 'mixed',
                'ownership_penalty': 0.2,
                'min_salary_usage': 0.96,
                'captain_multiplier': 1.5,
                'stack_bonus': 0.12,
                'lineup_diversity': 0.3
            }
        }
        
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load trained ML models"""
        try:
            # Load QB model
            qb_model = joblib.load('models/qb_model_proper.pkl')
            self.models['QB'] = qb_model
            print("✅ QB model loaded")
        except:
            print("⚠️ QB model not found")
        
        try:
            # Load RB model
            rb_model = joblib.load('models/rb_model_proper.pkl')
            self.models['RB'] = rb_model
            print("✅ RB model loaded")
            
            # Debug: Show what features the RB model expects
            if 'feature_columns' in rb_model:
                print(f"📊 RB model expects {len(rb_model['feature_columns'])} features:")
                for i, feature in enumerate(rb_model['feature_columns'][:10]):  # Show first 10
                    print(f"  {i+1}. {feature}")
                if len(rb_model['feature_columns']) > 10:
                    print(f"  ... and {len(rb_model['feature_columns']) - 10} more")
            
        except Exception as e:
            print(f"⚠️ RB model error: {e}")
    
    def generate_projections(self, player_data):
        """Generate ML projections for all players"""
        projections = []
        
        for _, player in player_data.iterrows():
            position = player['position']
            
            # Use ML model if available, otherwise use consensus
            if position in self.models:
                model_data = self.models[position]
                
                # Prepare features (this would use your actual feature engineering)
                features = self.prepare_player_features(player)
                features_scaled = model_data['scaler'].transform([features])
                ml_projection = model_data['model'].predict(features_scaled)[0]
            else:
                # Fallback to consensus for positions without ML models
                ml_projection = player.get('consensus_projection', 10.0)
            
            # Calculate ceiling and floor based on projection and consistency
            consistency = player.get('consistency', 0.7)
            ceiling = ml_projection * (1.4 + (0.2 * (1 - consistency)))
            floor = ml_projection * (0.4 + (0.3 * consistency))
            
            # Calculate value
            value = ml_projection / (player['salary'] / 1000)
            
            projections.append({
                'player_id': player.get('player_id', player['name']),
                'name': player['name'],
                'position': position,
                'team': player['team'],
                'salary': player['salary'],
                'ml_projection': ml_projection,
                'consensus': player.get('consensus_projection', ml_projection * 0.95),
                'ceiling': ceiling,
                'floor': floor,
                'ownership': player.get('ownership', 15.0),
                'value': value,
                'opponent': player.get('opponent', 'UNK'),
                'home_away': player.get('home_away', 'H'),
                'weather_impact': player.get('weather_impact', 0),
                'injury_status': player.get('injury_status', 'Healthy')
            })
        
        return pd.DataFrame(projections)
    
    def prepare_player_features(self, player):
        """Prepare features for ML model prediction with proper QB handling"""
        """Prepare features for ML model prediction with proper QB handling"""
        """Prepare features for ML model prediction with proper QB handling"""
        position = player['position']
        
        # Get the expected feature columns for this position
        if position in self.models and 'feature_columns' in self.models[position]:
            expected_features = self.models[position]['feature_columns']
        else:
            # Default features if model not available
            expected_features = ['fp_avg_3', 'fp_avg_5', 'fp_avg_8']
        
        # Create feature vector matching the expected features
        features = []
        
        for feature_name in expected_features:
            if feature_name.startswith('fp_avg_'):
                # Fantasy points averages
                window = feature_name.split('_')[-1]
                features.append(player.get(feature_name, 10.0))
            elif feature_name.startswith('carries_avg_'):
                # Carries averages (RB specific)
                features.append(player.get(feature_name, 15.0 if position == 'RB' else 0))
            elif feature_name.startswith('rushing_yards_avg_'):
                # Rushing yards averages
                features.append(player.get(feature_name, 60.0 if position == 'RB' else 0))
            elif feature_name.startswith('targets_avg_'):
                # Target averages
                features.append(player.get(feature_name, 5.0))
            elif feature_name.startswith('receptions_avg_'):
                # Reception averages
                features.append(player.get(feature_name, 4.0))
            elif feature_name.startswith('receiving_yards_avg_'):
                # Receiving yards averages
                features.append(player.get(feature_name, 30.0))
            elif 'ypc_avg' in feature_name:
                # Yards per carry
                features.append(player.get(feature_name, 4.2))
            elif 'target_share' in feature_name:
                # Target share
                features.append(player.get(feature_name, 0.15))
            elif 'air_yards_share' in feature_name:
                # Air yards share
                features.append(player.get(feature_name, 0.10))
            elif 'consistency' in feature_name:
                # Consistency metrics
                features.append(player.get(feature_name, 0.7))
            elif 'trend' in feature_name:
                # Trend metrics
                features.append(player.get(feature_name, 0.0))
            elif 'snap_pct' in feature_name:
                # Snap percentage
                features.append(player.get(feature_name, 0.75))
            elif 'redzone' in feature_name:
                # Red zone usage
                features.append(player.get(feature_name, 2.0))
            elif 'team_total' in feature_name:
                # Team total implied
                features.append(player.get(feature_name, 22.5))
            elif 'spread' in feature_name:
                # Spread impact
                features.append(player.get(feature_name, 0.0))
            elif 'years_exp' in feature_name:
                # Years of experience
                features.append(player.get(feature_name, 3.0))
            else:
                # Default for unknown features
                features.append(0.0)
        
        return features
    
    def calculate_contest_score(self, player, contest_type):
        """Calculate player score based on contest strategy"""
        config = self.contest_configs[contest_type]
        
        base_score = 0
        
        if config['objective'] == 'ceiling':
            base_score = player['ceiling']
        elif config['objective'] == 'floor':
            base_score = player['floor']
        elif config['objective'] == 'projection':
            base_score = player['ml_projection']
        elif config['objective'] == 'mixed':
            base_score = (player['ml_projection'] * 0.6 + player['ceiling'] * 0.4)
        
        # Apply ownership penalty for GPP
        ownership_penalty = (player['ownership'] / 100) * config['ownership_penalty']
        
        # Apply contrarian boost for low-owned players
        if player['ownership'] < 10:
            base_score += config.get('contrarian_boost', 0)
        
        final_score = base_score - ownership_penalty
        return final_score
    
    def optimize_main_slate(self, projections, contest_type='gpp', num_lineups=1):
        """Optimize main slate lineups"""
        print(f"\n🎯 Optimizing {num_lineups} {contest_type.upper()} lineup(s)")
        
        lineups = []
        used_players = set()
        
        for lineup_num in range(num_lineups):
            print(f"  Generating lineup {lineup_num + 1}...")
            
            # Create optimization problem
            prob = LpProblem(f"DFS_Lineup_{lineup_num}", LpMaximize)
            
            # Decision variables
            player_vars = {}
            for idx, player in projections.iterrows():
                player_vars[idx] = LpVariable(f"player_{idx}", cat='Binary')
            
            # Objective function
            contest_scores = [
                self.calculate_contest_score(projections.iloc[idx], contest_type) 
                for idx in projections.index
            ]
            
            prob += lpSum([
                player_vars[idx] * contest_scores[idx] 
                for idx in projections.index
            ])
            
            # Constraints
            # Salary cap
            prob += lpSum([
                player_vars[idx] * projections.iloc[idx]['salary'] 
                for idx in projections.index
            ]) <= 50000
            
            # Minimum salary usage
            min_salary = 50000 * self.contest_configs[contest_type]['min_salary_usage']
            prob += lpSum([
                player_vars[idx] * projections.iloc[idx]['salary'] 
                for idx in projections.index
            ]) >= min_salary
            
            # Roster construction (1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX, 1 DST)
            positions = projections['position'].value_counts()
            
            # QB constraint
            qb_players = projections[projections['position'] == 'QB'].index
            prob += lpSum([player_vars[idx] for idx in qb_players]) == 1
            
            # RB constraint
            rb_players = projections[projections['position'] == 'RB'].index
            prob += lpSum([player_vars[idx] for idx in rb_players]) >= 2
            prob += lpSum([player_vars[idx] for idx in rb_players]) <= 3
            
            # WR constraint
            wr_players = projections[projections['position'] == 'WR'].index
            prob += lpSum([player_vars[idx] for idx in wr_players]) >= 3
            prob += lpSum([player_vars[idx] for idx in wr_players]) <= 4
            
            # TE constraint
            te_players = projections[projections['position'] == 'TE'].index
            prob += lpSum([player_vars[idx] for idx in te_players]) >= 1
            prob += lpSum([player_vars[idx] for idx in te_players]) <= 2
            
            # Total players
            prob += lpSum([player_vars[idx] for idx in projections.index]) == 9
            
            # Diversity constraint for multiple lineups
            if lineup_num > 0 and contest_type in ['gpp', 'showdown_multi']:
                diversity_constraint = int(len(used_players) * 0.3)
                prob += lpSum([
                    player_vars[idx] for idx in used_players
                ]) <= max(6, 9 - diversity_constraint)
            
            # Solve
            prob.solve(PULP_CBC_CMD(msg=0))
            
            if prob.status == 1:  # Optimal solution found
                lineup = []
                total_salary = 0
                total_projection = 0
                
                for idx in projections.index:
                    if player_vars[idx].value() == 1:
                        player = projections.iloc[idx]
                        lineup.append(player)
                        total_salary += player['salary']
                        total_projection += player['ml_projection']
                        used_players.add(idx)
                
                lineups.append({
                    'lineup_num': lineup_num + 1,
                    'players': lineup,
                    'total_salary': total_salary,
                    'total_projection': total_projection,
                    'contest_type': contest_type
                })
            else:
                print(f"    ❌ No optimal solution found for lineup {lineup_num + 1}")
        
        return lineups
    
    def optimize_showdown(self, projections, contest_type='showdown_single', num_lineups=1):
        """Optimize showdown lineups with captain selection"""
        print(f"\n🏆 Optimizing {num_lineups} {contest_type.upper()} lineup(s)")
        
        lineups = []
        used_combinations = set()
        
        for lineup_num in range(num_lineups):
            print(f"  Generating lineup {lineup_num + 1}...")
            
            # Create optimization problem
            prob = LpProblem(f"Showdown_Lineup_{lineup_num}", LpMaximize)
            
            # Decision variables - separate for captain and flex
            captain_vars = {}
            flex_vars = {}
            
            for idx, player in projections.iterrows():
                captain_vars[idx] = LpVariable(f"captain_{idx}", cat='Binary')
                flex_vars[idx] = LpVariable(f"flex_{idx}", cat='Binary')
            
            # Objective function
            captain_multiplier = self.contest_configs[contest_type]['captain_multiplier']
            
            prob += lpSum([
                captain_vars[idx] * self.calculate_contest_score(projections.iloc[idx], contest_type) * captain_multiplier +
                flex_vars[idx] * self.calculate_contest_score(projections.iloc[idx], contest_type)
                for idx in projections.index
            ])
            
            # Constraints
            # Salary cap
            prob += lpSum([
                captain_vars[idx] * projections.iloc[idx]['salary'] * captain_multiplier +
                flex_vars[idx] * projections.iloc[idx]['salary']
                for idx in projections.index
            ]) <= 50000
            
            # Exactly 1 captain
            prob += lpSum([captain_vars[idx] for idx in projections.index]) == 1
            
            # Exactly 5 flex players
            prob += lpSum([flex_vars[idx] for idx in projections.index]) == 5
            
            # Player can't be both captain and flex
            for idx in projections.index:
                prob += captain_vars[idx] + flex_vars[idx] <= 1
            
            # Diversity for multiple lineups
            if lineup_num > 0 and len(used_combinations) > 0:
                # Add constraint to ensure different lineups
                pass
            
            # Solve
            prob.solve(PULP_CBC_CMD(msg=0))
            
            if prob.status == 1:  # Optimal solution found
                lineup = {'captain': None, 'flex': []}
                total_salary = 0
                total_projection = 0
                
                for idx in projections.index:
                    if captain_vars[idx].value() == 1:
                        player = projections.iloc[idx].copy()
                        player['role'] = 'Captain'
                        player['salary_used'] = player['salary'] * captain_multiplier
                        player['projection_used'] = player['ml_projection'] * captain_multiplier
                        lineup['captain'] = player
                        total_salary += player['salary_used']
                        total_projection += player['projection_used']
                    
                    elif flex_vars[idx].value() == 1:
                        player = projections.iloc[idx].copy()
                        player['role'] = 'Flex'
                        player['salary_used'] = player['salary']
                        player['projection_used'] = player['ml_projection']
                        lineup['flex'].append(player)
                        total_salary += player['salary_used']
                        total_projection += player['projection_used']
                
                # Create combination signature for diversity
                captain_id = lineup['captain']['player_id']
                flex_ids = sorted([p['player_id'] for p in lineup['flex']])
                combination = (captain_id, tuple(flex_ids))
                used_combinations.add(combination)
                
                lineups.append({
                    'lineup_num': lineup_num + 1,
                    'captain': lineup['captain'],
                    'flex': lineup['flex'],
                    'total_salary': total_salary,
                    'total_projection': total_projection,
                    'contest_type': contest_type
                })
            else:
                print(f"    ❌ No optimal solution found for lineup {lineup_num + 1}")
        
        return lineups
    
    def export_to_draftkings(self, lineups, contest_format='main_slate'):
        """Export lineups to DraftKings CSV format"""
        if contest_format == 'main_slate':
            return self._export_main_slate_csv(lineups)
        else:
            return self._export_showdown_csv(lineups)
    
    def _export_main_slate_csv(self, lineups):
        """Export main slate lineups to CSV"""
        export_data = []
        
        for lineup in lineups:
            row = {'Entry ID': lineup['lineup_num']}
            
            # Sort players by position for consistent ordering
            players = sorted(lineup['players'], key=lambda x: (x['position'], -x['salary']))
            
            position_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
            
            for player in players:
                pos = player['position']
                if pos in position_counts:
                    position_counts[pos] += 1
                    if pos == 'RB' and position_counts[pos] <= 2:
                        row[f'{pos}{position_counts[pos]}'] = f"{player['name']} ({player['player_id']})"
                    elif pos == 'WR' and position_counts[pos] <= 3:
                        row[f'{pos}{position_counts[pos]}'] = f"{player['name']} ({player['player_id']})"
                    elif pos in ['QB', 'TE'] and position_counts[pos] == 1:
                        row[pos] = f"{player['name']} ({player['player_id']})"
                    else:
                        # FLEX position
                        row['FLEX'] = f"{player['name']} ({player['player_id']})"
            
            # Add DST (placeholder)
            row['DST'] = "Defense/Special Teams"
            
            export_data.append(row)
        
        return pd.DataFrame(export_data)
    
    def _export_showdown_csv(self, lineups):
        """Export showdown lineups to CSV"""
        export_data = []
        
        for lineup in lineups:
            row = {'Entry ID': lineup['lineup_num']}
            
            # Captain
            captain = lineup['captain']
            row['CPT'] = f"{captain['name']} ({captain['player_id']})"
            
            # Flex players
            flex_players = sorted(lineup['flex'], key=lambda x: -x['salary'])
            for i, player in enumerate(flex_players, 1):
                row[f'FLEX{i}'] = f"{player['name']} ({player['player_id']})"
            
            export_data.append(row)
        
        return pd.DataFrame(export_data)
    
    def generate_research_report(self, projections, contest_type):
        """Generate detailed research report for contest strategy"""
        config = self.contest_configs[contest_type]
        
        report = f"""
=== {config['name']} Research Report ===

STRATEGY OVERVIEW:
- Primary Objective: {config['objective']}
- Ownership Penalty: {config['ownership_penalty']}
- Min Salary Usage: {config['min_salary_usage']*100:.1f}%

TOP PLAYS BY POSITION:
"""
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_players = projections[projections['position'] == position]
            if len(pos_players) > 0:
                pos_players['contest_score'] = pos_players.apply(
                    lambda x: self.calculate_contest_score(x, contest_type), axis=1
                )
                top_players = pos_players.nlargest(3, 'contest_score')
                
                report += f"\n{position}:\n"
                for _, player in top_players.iterrows():
                    edge = player['ml_projection'] - player['consensus']
                    report += f"  {player['name']} (${player['salary']}) - "
                    report += f"Proj: {player['ml_projection']:.1f}, "
                    report += f"Own: {player['ownership']:.1f}%, "
                    report += f"Edge: {edge:+.1f}\n"
        
        # Key insights
        report += f"\nKEY INSIGHTS:\n"
        
        # High value plays
        high_value = projections[projections['value'] > 2.5]
        if len(high_value) > 0:
            report += f"- {len(high_value)} players with value > 2.5\n"
        
        # Low ownership pivots
        low_owned = projections[projections['ownership'] < 15]
        if len(low_owned) > 0:
            report += f"- {len(low_owned)} players under 15% ownership\n"
        
        # Strong ML edges
        strong_edges = projections[projections['ml_projection'] - projections['consensus'] > 2]
        if len(strong_edges) > 0:
            report += f"- {len(strong_edges)} players with 2+ point projection edge\n"
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Sample player data
    sample_data = pd.DataFrame([
        {'name': 'Josh Allen', 'position': 'QB', 'team': 'BUF', 'salary': 8200, 'consensus_projection': 22.1, 'ownership': 28.5},
        {'name': 'Christian McCaffrey', 'position': 'RB', 'team': 'SF', 'salary': 9000, 'consensus_projection': 18.2, 'ownership': 35.2},
        {'name': 'Austin Ekeler', 'position': 'RB', 'team': 'LAC', 'salary': 7800, 'consensus_projection': 15.8, 'ownership': 28.1},
        {'name': 'Tyreek Hill', 'position': 'WR', 'team': 'MIA', 'salary': 8600, 'consensus_projection': 16.8, 'ownership': 32.1},
        {'name': 'Travis Kelce', 'position': 'TE', 'team': 'KC', 'salary': 7200, 'consensus_projection': 14.2, 'ownership': 26.8},
    ])
    
    optimizer = ContestStrategyOptimizer()
    projections = optimizer.generate_projections(sample_data)
    
    print("Sample projections generated:")
    print(projections[['name', 'position', 'ml_projection', 'ceiling', 'floor', 'value']].head())
    
    # Generate research report
    report = optimizer.generate_research_report(projections, 'gpp')
    print(report)