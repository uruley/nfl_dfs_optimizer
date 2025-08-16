import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ManualSalaryLoader:
    """
    Load manually downloaded DraftKings CSV files
    """
    
    def __init__(self):
        self.data_dir = 'data/live_slates'
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs('data/raw', exist_ok=True)
        
    def create_sample_dk_csv(self):
        """
        Create a sample DraftKings CSV format for testing
        """
        
        # Create simple, balanced sample data (8 per position)
        players = []
        
        # 8 QBs
        qb_data = [
            ('Josh Allen', 7200, 'BUF', 'BUF@MIA 1:00PM ET', 23.2),
            ('Lamar Jackson', 6800, 'BAL', 'BAL@CIN 1:00PM ET', 21.8),
            ('Patrick Mahomes', 7000, 'KC', 'KC@DEN 4:25PM ET', 21.5),
            ('Dak Prescott', 6400, 'DAL', 'DAL@NYG 1:00PM ET', 19.8),
            ('Tua Tagovailoa', 5800, 'MIA', 'MIA@BUF 1:00PM ET', 18.9),
            ('Russell Wilson', 5600, 'PIT', 'PIT@CLE 1:00PM ET', 17.2),
            ('Kyler Murray', 5900, 'ARI', 'ARI@SEA 4:25PM ET', 18.5),
            ('Geno Smith', 5200, 'SEA', 'SEA@ARI 4:25PM ET', 16.4)
        ]
        
        # 8 RBs
        rb_data = [
            ('Christian McCaffrey', 7200, 'SF', 'SF@LAR 4:25PM ET', 20.1),
            ('Saquon Barkley', 6600, 'NYG', 'NYG@DAL 1:00PM ET', 17.8),
            ('Josh Jacobs', 6000, 'LV', 'LV@KC 4:25PM ET', 16.2),
            ('Aaron Jones', 5400, 'MIN', 'MIN@GB 1:00PM ET', 14.9),
            ('Derrick Henry', 5800, 'BAL', 'BAL@CIN 1:00PM ET', 16.8),
            ('Tony Pollard', 4800, 'DAL', 'DAL@NYG 1:00PM ET', 13.2),
            ('Najee Harris', 4400, 'PIT', 'PIT@CLE 1:00PM ET', 12.8),
            ('Austin Ekeler', 4000, 'LAC', 'LAC@LV 4:25PM ET', 12.1)
        ]
        
        # 8 WRs
        wr_data = [
            ('Stefon Diggs', 6200, 'BUF', 'BUF@MIA 1:00PM ET', 15.2),
            ('Davante Adams', 5800, 'LV', 'LV@KC 4:25PM ET', 14.8),
            ('Tyreek Hill', 5600, 'MIA', 'MIA@BUF 1:00PM ET', 14.5),
            ('Mike Evans', 5200, 'TB', 'TB@NO 1:00PM ET', 13.9),
            ('Chris Godwin', 4800, 'TB', 'TB@NO 1:00PM ET', 12.8),
            ('CeeDee Lamb', 5400, 'DAL', 'DAL@NYG 1:00PM ET', 14.2),
            ('A.J. Brown', 5000, 'PHI', 'PHI@WAS 1:00PM ET', 13.6),
            ('DK Metcalf', 4600, 'SEA', 'SEA@ARI 4:25PM ET', 12.9)
        ]
        
        # 8 TEs
        te_data = [
            ('Travis Kelce', 5400, 'KC', 'KC@DEN 4:25PM ET', 12.8),
            ('Mark Andrews', 4600, 'BAL', 'BAL@CIN 1:00PM ET', 10.2),
            ('T.J. Hockenson', 4000, 'MIN', 'MIN@GB 1:00PM ET', 9.8),
            ('Kyle Pitts', 3600, 'ATL', 'ATL@CAR 1:00PM ET', 8.9),
            ('George Kittle', 4200, 'SF', 'SF@LAR 4:25PM ET', 10.5),
            ('Darren Waller', 3400, 'NYG', 'NYG@DAL 1:00PM ET', 8.2),
            ('Dallas Goedert', 3200, 'PHI', 'PHI@WAS 1:00PM ET', 7.8),
            ('Pat Freiermuth', 3000, 'PIT', 'PIT@CLE 1:00PM ET', 7.2)
        ]
        
        # 8 DSTs
        dst_data = [
            ('Bills DST', 2400, 'BUF', 'BUF@MIA 1:00PM ET', 9.2),
            ('Ravens DST', 2200, 'BAL', 'BAL@CIN 1:00PM ET', 8.8),
            ('49ers DST', 2000, 'SF', 'SF@LAR 4:25PM ET', 8.1),
            ('Cowboys DST', 1800, 'DAL', 'DAL@NYG 1:00PM ET', 7.5),
            ('Steelers DST', 1600, 'PIT', 'PIT@CLE 1:00PM ET', 7.0),
            ('Eagles DST', 1800, 'PHI', 'PHI@WAS 1:00PM ET', 7.2),
            ('Seahawks DST', 1600, 'SEA', 'SEA@ARI 4:25PM ET', 6.8),
            ('Chiefs DST', 1700, 'KC', 'KC@DEN 4:25PM ET', 7.1)
        ]
        
        # Combine all data
        all_data = [
            ('QB', qb_data), ('RB', rb_data), ('WR', wr_data), 
            ('TE', te_data), ('DST', dst_data)
        ]
        
        # Build the dataframe
        names, ids, positions, salaries, teams, games, avg_points = [], [], [], [], [], [], []
        
        player_id = 1001
        for position, position_data in all_data:
            for name, salary, team, game, avg_point in position_data:
                names.append(name)
                ids.append(player_id)
                positions.append(position)
                salaries.append(salary)
                teams.append(team)
                games.append(game)
                avg_points.append(avg_point)
                player_id += 1
        
        sample_data = pd.DataFrame({
            'Name': names,
            'ID': ids,
            'Position': positions,
            'Salary': salaries,
            'GameInfo': games,
            'TeamAbbrev': teams,
            'AvgPointsPerGame': avg_points
        })
        
        # Save sample CSV
        sample_file = 'data/raw/sample_DKSalaries.csv'
        sample_data.to_csv(sample_file, index=False)
        
        print(f"✅ Created sample DraftKings CSV: {sample_file}")
        print(f"📊 Contains {len(sample_data)} players")
        print(f"💡 This simulates a real DraftKings download")
        
        return sample_data, sample_file
    
    def load_dk_csv(self, csv_path=None):
        """
        Load DraftKings CSV (real or sample)
        """
        
        # Try common locations for DraftKings CSV
        if csv_path is None:
            possible_paths = [
                'data/raw/DKSalaries.csv',           # Standard location
                'data/raw/sample_DKSalaries.csv',    # Our sample
                'Downloads/DKSalaries.csv',          # Downloaded to Downloads
                'DKSalaries.csv'                     # Current directory
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
        
        if csv_path is None or not os.path.exists(csv_path):
            print(f"❌ No DraftKings CSV found!")
            print(f"\n💡 HOW TO GET REAL DRAFTKINGS DATA:")
            print(f"1. Go to DraftKings.com (when NFL season starts)")
            print(f"2. Navigate to NFL contests")
            print(f"3. Click any contest (Main Slate, Early, etc.)")
            print(f"4. Look for 'Export' or 'Download CSV' button")
            print(f"5. Save as 'data/raw/DKSalaries.csv'")
            print(f"\n🧪 FOR NOW: Creating sample data for testing...")
            
            return self.create_sample_dk_csv()
        
        try:
            # Load the CSV
            data = pd.read_csv(csv_path)
            
            print(f"✅ Loaded DraftKings CSV: {csv_path}")
            print(f"📊 Players: {len(data)}")
            
            # Show what columns we have
            print(f"📋 Columns: {list(data.columns)}")
            
            # Show sample
            print(f"\n📈 Sample data:")
            display_cols = ['Name', 'Position', 'Salary', 'TeamAbbrev']
            available_cols = [col for col in display_cols if col in data.columns]
            print(data[available_cols].head().to_string(index=False))
            
            return data, csv_path
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None, None
    
    def format_for_optimizer(self, dk_data):
        """
        Convert DraftKings CSV format to your optimizer format
        """
        
        if dk_data is None or dk_data.empty:
            return pd.DataFrame()
        
        # Map DraftKings columns to your format
        formatted = pd.DataFrame()
        
        formatted['player_name'] = dk_data['Name']
        formatted['position'] = dk_data['Position']
        formatted['salary'] = dk_data['Salary'].astype(int)
        formatted['team'] = dk_data['TeamAbbrev']
        
        # Add game info if available
        if 'GameInfo' in dk_data.columns:
            formatted['game_info'] = dk_data['GameInfo']
        
        # Add average points as baseline projection
        if 'AvgPointsPerGame' in dk_data.columns:
            formatted['projection'] = dk_data['AvgPointsPerGame']
        else:
            # Default projections by position
            position_defaults = {'QB': 18.0, 'RB': 12.0, 'WR': 10.0, 'TE': 8.0, 'DST': 6.0}
            formatted['projection'] = formatted['position'].map(position_defaults).fillna(8.0)
        
        # Detect slate type
        unique_games = formatted['game_info'].nunique() if 'game_info' in formatted.columns else 8
        if unique_games <= 2:
            slate_type = "early_slate"
        elif unique_games <= 4:
            slate_type = "afternoon_slate"
        else:
            slate_type = "main_slate"
        
        formatted['slate_type'] = slate_type
        
        print(f"🔧 Formatted for optimizer:")
        print(f"   Players: {len(formatted)}")
        print(f"   Slate type: {slate_type}")
        print(f"   Position breakdown: {formatted['position'].value_counts().to_dict()}")
        
        return formatted
    
    def save_for_optimizer(self, formatted_data):
        """
        Save formatted data for your optimizer
        """
        
        if formatted_data.empty:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        slate_type = formatted_data['slate_type'].iloc[0]
        
        output_file = f"{self.data_dir}/dk_slate_{slate_type}_{timestamp}.csv"
        formatted_data.to_csv(output_file, index=False)
        
        print(f"💾 Saved optimizer-ready data: {output_file}")
        
        return output_file

def test_manual_loader():
    """
    Test the manual CSV loading system
    """
    
    print("🏈 TESTING MANUAL DRAFTKINGS CSV LOADER")
    print("="*60)
    
    loader = ManualSalaryLoader()
    
    # Load DraftKings data
    dk_data, source_file = loader.load_dk_csv()
    
    if dk_data is not None:
        # Format for optimizer
        formatted_data = loader.format_for_optimizer(dk_data)
        
        # Save formatted data
        output_file = loader.save_for_optimizer(formatted_data)
        
        print(f"\n🏆 SUCCESS!")
        print(f"📁 Source: {source_file}")
        print(f"📁 Output: {output_file}")
        print(f"🚀 Ready for your optimizer!")
        
        return output_file
    else:
        print(f"\n❌ Failed to load data")
        return None

if __name__ == "__main__":
    test_manual_loader()