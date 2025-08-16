# show_file_structure.py
# Complete file structure analysis for DFS optimizer project

import os
import sys
from pathlib import Path
import joblib

def analyze_project_structure():
    """Analyze complete project structure and model status"""
    
    print("🏈 DFS Optimizer Project Analysis")
    print("=" * 60)
    
    # Show current working directory
    current_dir = os.getcwd()
    print(f"📍 Current Directory: {current_dir}")
    print()
    
    # Check for both possible project locations
    locations_to_check = [
        current_dir,
        r"C:\Users\ruley\dfs_optimizer",
        r"C:\Users\ruley\source\repos\dfs_optimizer"
    ]
    
    for location in locations_to_check:
        if os.path.exists(location):
            print(f"📂 FOUND PROJECT: {location}")
            analyze_directory(location)
            print("-" * 60)
        else:
            print(f"❌ Not found: {location}")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("- Copy this output and share with Claude")
    print("- Choose which location to use as main project")
    print("- We'll consolidate everything in Visual Studio")

def analyze_directory(directory):
    """Analyze a specific directory structure"""
    
    print(f"\n📁 Structure for: {directory}")
    print("-" * 40)
    
    # Show directory tree
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}📁 {os.path.basename(root)}/")
        
        # Show files with details
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
                if size > 1024*1024:  # > 1MB
                    size_str = f"{size/(1024*1024):.1f}MB"
                elif size > 1024:  # > 1KB
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                
                # Special markers for important files
                marker = ""
                if file.endswith('.pkl'):
                    marker = " 🤖"
                elif file.endswith('.py'):
                    marker = " 🐍"
                elif file.endswith('.html'):
                    marker = " 🌐"
                elif file.endswith('.csv'):
                    marker = " 📊"
                
                print(f"{subindent}📄 {file} ({size_str}){marker}")
                
            except:
                print(f"{subindent}📄 {file} (size unknown)")
    
    # Check for models specifically
    models_dir = os.path.join(directory, 'models')
    if os.path.exists(models_dir):
        print(f"\n🤖 MODEL ANALYSIS for {directory}:")
        analyze_models(models_dir)
    
    # Check for scripts
    scripts_dir = os.path.join(directory, 'Scripts')
    if os.path.exists(scripts_dir):
        print(f"\n🐍 SCRIPTS ANALYSIS for {directory}:")
        analyze_scripts(scripts_dir)

def analyze_models(models_dir):
    """Analyze model files in detail"""
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        try:
            model_data = joblib.load(model_path)
            print(f"  ✅ {model_file}:")
            
            # Check model format
            if isinstance(model_data, dict):
                keys = list(model_data.keys())
                print(f"     Keys: {keys}")
                
                if 'performance' in model_data:
                    perf = model_data['performance']
                    mae = perf.get('mae', 'N/A')
                    corr = perf.get('correlation', 'N/A')
                    print(f"     Performance: MAE={mae}, Correlation={corr}")
                
                if 'feature_columns' in model_data:
                    print(f"     Features: {len(model_data['feature_columns'])}")
                elif 'features' in model_data:
                    print(f"     Features: {len(model_data['features'])}")
                    
            else:
                print(f"     Type: {type(model_data)}")
                
        except Exception as e:
            print(f"  ❌ {model_file}: Error loading - {e}")

def analyze_scripts(scripts_dir):
    """Analyze Python scripts"""
    
    script_files = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]
    
    for script in script_files:
        script_path = os.path.join(scripts_dir, script)
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = len(content.split('\n'))
                
            # Check what type of script it is
            script_type = "Unknown"
            if 'MLTrainer' in content:
                script_type = "ML Training"
            elif 'optimizer' in script.lower() or 'lineup' in script.lower():
                script_type = "Optimizer"
            elif 'test' in script.lower():
                script_type = "Test Script"
            elif 'dashboard' in script.lower():
                script_type = "Dashboard"
                
            print(f"  📝 {script} ({lines} lines) - {script_type}")
            
        except Exception as e:
            print(f"  ❌ {script}: Error reading - {e}")

def check_python_environment():
    """Check Python environment and packages"""
    
    print(f"\n🐍 PYTHON ENVIRONMENT:")
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.executable}")
    
    # Check key packages
    packages = ['pandas', 'numpy', 'scikit-learn', 'joblib', 'nfl_data_py']
    for package in packages:
        try:
            __import__(package)
            print(f"  ✅ {package} installed")
        except ImportError:
            print(f"  ❌ {package} NOT installed")

if __name__ == "__main__":
    try:
        analyze_project_structure()
        check_python_environment()
        
        print("\n" + "=" * 60)
        print("🎯 ANALYSIS COMPLETE!")
        print("Share this output with Claude to get your project organized!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please share this error with Claude")