# Scripts/complete_backtesting_pipeline.py

import pandas as pd
import numpy as np
from pathlib import Path
import nfl_data_py as nfl

def build_backtest_pipeline():
    """
    Complete backtesting pipeline using your existing components:
    1. Load your salary projections (salaries_merged.csv)
    2. Get actual fantasy results from nfl_data_py
    3. Generate lineups using your optimizer
    4. Calculate ROI against simulated contests
    """
    
    print("🏈 Starting Complete Backtesting Pipeline")
    print("=" * 50)
    
    # Step 1: Load your existing salary/projection data
    merged_file = Path("data/salaries_merged.csv")
    if not merged_file.exists():
        print("❌ Run merge script first: python Scripts/merge_real_and_proj_salaries.py")
        return
    
    df = pd.read_csv(merged_file)
    print(f"✅ Loaded {len(df)} players from merged salary data")
    
    # Step 2: Get actual fantasy results
    print("\n📊 Loading actual fantasy results from nfl_data_py...")
    
    # Load actual weekly data for comparison
    seasons = [2024]  # Match your data
    weeks = df['week'].unique()
    
    actual_results = []
    for season in seasons:
        print(f"Loading {season} data...")
        weekly_data = nfl.import_weekly_data([season])
        
        # Filter to relevant weeks and add fantasy points
        for week in weeks:
            week_data = weekly_data[
                (weekly_data['season'] == season) & 
                (weekly_data['week'] == week)
            ].copy()
            
            if week_data.empty:
                continue
                
            # Calculate fantasy points (DraftKings scoring)
            week_data['actual_fpts'] = (
                # Passing
                week_data['passing_yards'] * 0.04 +
                week_data['passing_tds'] * 4 +
                week_data['interceptions'] * -1 +
                # Rushing  
                week_data['rushing_yards'] * 0.1 +
                week_data['rushing_tds'] * 6 +
                # Receiving
                week_data['receiving_yards'] * 0.1 +
                week_data['receptions'] * 1 +
                week_data['receiving_tds'] * 6 +
                # Bonuses
                (week_data['passing_yards'] >= 300) * 3 +
                (week_data['rushing_yards'] >= 100) * 3 +
                (week_data['receiving_yards'] >= 100) * 3
            ).fillna(0)
            
            # Clean up for merging
            week_data['name'] = week_data['player_display_name']
            week_data = week_data[['season', 'week', 'name', 'position', 'actual_fpts']].copy()
            
            actual_results.append(week_data)
    
    if not actual_results:
        print("❌ No actual results found for your weeks")
        return
        
    actual_df = pd.concat(actual_results, ignore_index=True)
    print(f"✅ Loaded actual results for {len(actual_df)} player-games")
    
    # Step 3: Merge projections with actual results
    print("\n🔗 Merging projections with actual results...")
    
    # Normalize names for better matching
    for data in [df, actual_df]:
        data['name_clean'] = data['name'].str.strip().str.lower()
    
    backtest_df = df.merge(
        actual_df, 
        on=['season', 'week', 'name_clean'], 
        how='inner',
        suffixes=('', '_actual')
    )
    
    print(f"✅ Matched {len(backtest_df)} players with actual results")
    
    if len(backtest_df) < 50:
        print("⚠️ Low match rate - check name formatting")
        # Show some examples
        print("\nProjection names sample:")
        print(df['name'].head(10).tolist())
        print("\nActual names sample:")
        print(actual_df['name'].head(10).tolist())
    
    # Step 4: Analyze projection accuracy
    print("\n📈 Analyzing Projection Accuracy...")
    
    if 'proj_fpts' in backtest_df.columns:
        backtest_df['fpts_error'] = backtest_df['proj_fpts'] - backtest_df['actual_fpts']
        backtest_df['abs_error'] = backtest_df['fpts_error'].abs()
        
        overall_mae = backtest_df['abs_error'].mean()
        overall_corr = backtest_df['proj_fpts'].corr(backtest_df['actual_fpts'])
        
        print(f"🎯 Overall Projection Accuracy:")
        print(f"   MAE: {overall_mae:.2f} fantasy points")
        print(f"   Correlation: {overall_corr:.3f}")
        
        # By position
        print(f"\n📊 Accuracy by Position:")
        pos_accuracy = backtest_df.groupby('pos').agg({
            'abs_error': 'mean',
            'proj_fpts': lambda x: x.corr(backtest_df.loc[x.index, 'actual_fpts'])
        }).round(3)
        print(pos_accuracy)
    
    # Step 5: Simulate lineup optimization
    print("\n🎯 Simulating Lineup Optimization...")
    
    # Group by week for lineup generation
    weekly_results = []
    
    for (season, week), week_data in backtest_df.groupby(['season', 'week']):
        if len(week_data) < 20:  # Need minimum players
            continue
            
        # Simulate basic lineup optimization (simplified)
        lineup = simulate_optimal_lineup(week_data)
        
        if lineup is not None:
            weekly_results.append({
                'season': season,
                'week': week,
                'projected_score': lineup['proj_fpts'].sum() if 'proj_fpts' in lineup.columns else 0,
                'actual_score': lineup['actual_fpts'].sum(),
                'players': len(lineup),
                'total_salary': lineup['salary_final'].sum() if 'salary_final' in lineup.columns else 0
            })
    
    if weekly_results:
        results_df = pd.DataFrame(weekly_results)
        print(f"✅ Generated {len(results_df)} weekly lineups")
        
        # Calculate performance metrics
        avg_actual = results_df['actual_score'].mean()
        avg_projected = results_df['projected_score'].mean()
        projection_bias = avg_projected - avg_actual
        
        print(f"\n🏆 Lineup Performance:")
        print(f"   Average Actual Score: {avg_actual:.1f}")
        print(f"   Average Projected Score: {avg_projected:.1f}")
        print(f"   Projection Bias: {projection_bias:+.1f}")
        
        # Estimate contest performance
        print(f"\n💰 Estimated Contest Performance:")
        cash_line = 135  # Typical DraftKings cash line
        tournament_line = 160  # Competitive tournament score
        
        cash_rate = (results_df['actual_score'] >= cash_line).mean() * 100
        tournament_rate = (results_df['actual_score'] >= tournament_line).mean() * 100
        
        print(f"   Cash Rate: {cash_rate:.1f}% (>= {cash_line} pts)")
        print(f"   Tournament Rate: {tournament_rate:.1f}% (>= {tournament_line} pts)")
        
        # ROI estimation
        if cash_rate > 50:
            estimated_roi = (cash_rate - 55) * 2  # Rough estimate accounting for rake
            print(f"   Estimated ROI: {estimated_roi:+.1f}%")
        else:
            estimated_roi = -(55 - cash_rate) * 2
            print(f"   Estimated ROI: {estimated_roi:+.1f}% (LOSING)")
        
        # Save results
        results_df.to_csv("data/backtest_results.csv", index=False)
        backtest_df.to_csv("data/backtest_detailed.csv", index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   data/backtest_results.csv - Weekly lineup performance")
        print(f"   data/backtest_detailed.csv - Player-level accuracy")
        
        return results_df, backtest_df
    
    else:
        print("❌ Could not generate any lineups")
        return None, backtest_df

def simulate_optimal_lineup(week_data):
    """
    Simplified lineup optimizer for backtesting
    """
    try:
        # Basic position requirements for DraftKings
        positions_needed = {
            'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DST': 1
        }
        
        lineup_indices = []
        week_data = week_data.copy().reset_index(drop=True)
        
        # Sort by value (projected points per $1000 salary)
        if 'proj_fpts' in week_data.columns and 'salary_final' in week_data.columns:
            week_data['value'] = week_data['proj_fpts'] / (week_data['salary_final'] / 1000 + 0.001)  # Avoid division by zero
            week_data = week_data.sort_values('value', ascending=False).reset_index(drop=True)
        else:
            # Fallback: use actual_fpts as proxy for projections
            if 'actual_fpts' in week_data.columns and 'salary_final' in week_data.columns:
                week_data['value'] = week_data['actual_fpts'] / (week_data['salary_final'] / 1000 + 0.001)
                week_data = week_data.sort_values('value', ascending=False).reset_index(drop=True)
        
        # Select players by position
        for pos, count in positions_needed.items():
            if pos == 'FLEX':
                # FLEX can be RB, WR, or TE (not already selected)
                available_data = week_data[
                    week_data['pos'].isin(['RB', 'WR', 'TE']) &
                    ~week_data.index.isin(lineup_indices)
                ]
            else:
                available_data = week_data[
                    (week_data['pos'] == pos) &
                    ~week_data.index.isin(lineup_indices)
                ]
            
            if len(available_data) >= count:
                selected_indices = available_data.head(count).index.tolist()
                lineup_indices.extend(selected_indices)
        
        if len(lineup_indices) >= 8:  # Minimum viable lineup
            return week_data.loc[lineup_indices]
        else:
            print(f"⚠️ Only {len(lineup_indices)} players selected, need 9")
            return None
            
    except Exception as e:
        print(f"⚠️ Lineup optimization error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    weekly_results, detailed_results = build_backtest_pipeline()
    
    if weekly_results is not None:
        print("\n🎉 Backtesting Complete!")
        print("Check data/backtest_results.csv for weekly performance")
        print("Check data/backtest_detailed.csv for player-level accuracy")
    else:
        print("\n❌ Backtesting failed - check data alignment")