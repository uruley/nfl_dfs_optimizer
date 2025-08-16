import pandas as pd
import os
from datetime import datetime
import requests
import warnings
warnings.filterwarnings('ignore')

class LiveSalaryManager:
    """
    Manages live DraftKings salary data for different slates
    """
    
    def __init__(self):
        self.data_dir = 'data/live_slates'
        os.makedirs(self.data_dir, exist_ok=True)
        
    def method_1_scraper_package(self):
        """
        Method 1: Use dfs-salary-scraper package
        """
        try:
            from dfs_salary_scraper import DraftKingsNFL
            
            print("🔄 Attempting to scrape DraftKings salaries...")
            
            slate = DraftKingsNFL()
            data = slate.get_salaries()
            
            if not data.empty:
                # Save raw salary data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                filename = f"{self.data_dir}/dk_salaries_{timestamp}.csv"
                data.to_csv(filename, index=False)
                
                print(f"✅ Scraped {len(data)} players")
                print(f"📁 Saved to: {filename}")
                
                # Show sample
                print(f"\n📊 Sample data:")
                print(data.head()[['Name', 'Salary', 'Position', 'Team']].to_string(index=False))
                
                return data, filename
            else:
                print("❌ No salary data found - may be off-season")
                return None, None
                
        except ImportError:
            print("❌ dfs-salary-scraper not installed")
            print("Run: pip install dfs-salary-scraper")
            return None, None
        except Exception as e:
            print(f"❌ Scraping error: {e}")
            return None, None
    
    def method_2_manual_csv(self, csv_path=None):
        """
        Method 2: Load manually downloaded DraftKings CSV
        """
        
        if csv_path is None:
            # Check for common CSV locations
            possible_paths = [
                'data/raw/DKSalaries.csv',
                'data/raw/dk_salaries.csv', 
                'Downloads/DKSalaries.csv',
                'DKSalaries.csv'
            ]
            
            csv_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
        
        if csv_path is None or not os.path.exists(csv_path):
            print(f"❌ No CSV found. Download from DraftKings:")
            print(f"1. Go to DraftKings.com")
            print(f"2. Choose an NFL contest")
            print(f"3. Click 'Download Player CSV'")
            print(f"4. Save as 'data/raw/DKSalaries.csv'")
            return None, None
        
        try:
            data = pd.read_csv(csv_path)
            
            print(f"✅ Loaded {len(data)} players from {csv_path}")
            
            # Standardize column names
            column_mapping = {
                'Name': 'player_name',
                'ID': 'player_id', 
                'Position': 'position',
                'Salary': 'salary',
                'GameInfo': 'game_info',
                'AvgPointsPerGame': 'avg_points',
                'TeamAbbrev': 'team'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in data.columns:
                    data = data.rename(columns={old_name: new_name})
            
            # Show sample
            display_cols = [col for col in ['player_name', 'salary', 'position', 'team'] if col in data.columns]
            print(f"\n📊 Sample data:")
            print(data[display_cols].head().to_string(index=False))
            
            return data, csv_path
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None, None
    
    def method_3_draftfast(self):
        """
        Method 3: Use draftfast package to download salary CSV
        """
        try:
            from draftfast.csv_parse import salary_download
            
            print("🔄 Using draftfast to download DraftKings CSV...")
            
            # Download salary CSV
            result = salary_download.download_csv(site='dk', sport='nfl')
            
            # Look for downloaded file
            csv_files = [f for f in os.listdir('.') if f.startswith('DKSalaries') and f.endswith('.csv')]
            
            if csv_files:
                latest_csv = max(csv_files, key=os.path.getmtime)
                data = pd.read_csv(latest_csv)
                
                print(f"✅ Downloaded {len(data)} players via draftfast")
                print(f"📁 File: {latest_csv}")
                
                return data, latest_csv
            else:
                print("❌ No CSV file found after download")
                return None, None
                
        except ImportError:
            print("❌ draftfast not installed")
            print("Run: pip install draftfast")
            return None, None
        except Exception as e:
            print(f"❌ Draftfast error: {e}")
            return None, None
    
    def detect_slate_type(self, data):
        """
        Detect what type of slate this is based on the data
        """
        
        if data is None or data.empty:
            return "unknown"
        
        total_players = len(data)
        unique_teams = data['team'].nunique() if 'team' in data.columns else 0
        
        # Heuristics for slate detection
        if total_players < 20:
            return "showdown"  # Single game slate
        elif unique_teams <= 4:
            return "early_slate"  # 2 games
        elif unique_teams <= 8:
            return "afternoon_slate"  # 4 games  
        elif unique_teams >= 10:
            return "main_slate"  # Sunday main slate
        else:
            return "unknown"
    
    def format_for_optimizer(self, salary_data, slate_type):
        """
        Format DraftKings salary data for your optimizer
        """
        
        if salary_data is None or salary_data.empty:
            return pd.DataFrame()
        
        # Create standard format
        formatted = pd.DataFrame()
        
        # Map columns to standard names
        if 'player_name' in salary_data.columns:
            formatted['player_name'] = salary_data['player_name']
        elif 'Name' in salary_data.columns:
            formatted['player_name'] = salary_data['Name']
        
        if 'position' in salary_data.columns:
            formatted['position'] = salary_data['position']
        elif 'Position' in salary_data.columns:
            formatted['position'] = salary_data['Position']
        
        if 'salary' in salary_data.columns:
            formatted['salary'] = salary_data['salary']
        elif 'Salary' in salary_data.columns:
            formatted['salary'] = salary_data['Salary']
        
        if 'team' in salary_data.columns:
            formatted['team'] = salary_data['team']
        elif 'TeamAbbrev' in salary_data.columns:
            formatted['team'] = salary_data['TeamAbbrev']
        
        # Add placeholder projections (you'll replace with your ML model)
        formatted['projection'] = 0.0  # Will be filled by your QB model + consensus
        
        # Add slate info
        formatted['slate_type'] = slate_type
        
        print(f"🔧 Formatted {len(formatted)} players for {slate_type}")
        
        return formatted
    
    def get_live_slate_data(self):
        """
        Try all methods to get current slate data
        """
        
        print("🏈 GETTING LIVE DRAFTKINGS SLATE DATA")
        print("="*50)
        
        data = None
        source = None
        
        # Try Method 1: Scraper package
        print("\n🔍 Method 1: Trying dfs-salary-scraper...")
        data, source = self.method_1_scraper_package()
        
        if data is None:
            # Try Method 2: Manual CSV
            print("\n🔍 Method 2: Looking for manual CSV...")
            data, source = self.method_2_manual_csv()
        
        if data is None:
            # Try Method 3: Draftfast
            print("\n🔍 Method 3: Trying draftfast download...")
            data, source = self.method_3_draftfast()
        
        if data is None:
            print("\n❌ All methods failed!")
            print("\n💡 SOLUTIONS:")
            print("1. Install packages: pip install dfs-salary-scraper draftfast")
            print("2. Manually download CSV from DraftKings")
            print("3. Wait for NFL season to start (live slates)")
            return None, None, None
        
        # Detect slate type
        slate_type = self.detect_slate_type(data)
        
        # Format for optimizer
        formatted_data = self.format_for_optimizer(data, slate_type)
        
        # Save formatted data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = f"{self.data_dir}/formatted_slate_{slate_type}_{timestamp}.csv"
        formatted_data.to_csv(output_file, index=False)
        
        print(f"\n✅ SUCCESS!")
        print(f"📊 Slate Type: {slate_type}")
        print(f"📁 Formatted data: {output_file}")
        print(f"🎯 Ready for your optimizer!")
        
        return formatted_data, slate_type, output_file

def test_live_salary_pipeline():
    """
    Test the live salary pipeline
    """
    
    manager = LiveSalaryManager()
    data, slate_type, output_file = manager.get_live_slate_data()
    
    if data is not None:
        print(f"\n🏆 PIPELINE TEST SUCCESS!")
        print(f"Next steps:")
        print(f"1. Add your QB projections to this data")
        print(f"2. Run your optimizer with: {output_file}")
        print(f"3. Generate lineups for {slate_type}")
    else:
        print(f"\n❌ PIPELINE TEST FAILED")
        print(f"Check the solutions above")

if __name__ == "__main__":
    test_live_salary_pipeline()