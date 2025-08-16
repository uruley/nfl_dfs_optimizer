import pandas as pd
import numpy as np
import nfl_data_py as nfl
import joblib
from datetime import datetime
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ContinuousLearningSystem:
    """
    System that learns and improves from each NFL week's results
    This is what makes ML truly powerful - continuous improvement!
    """
    
    def __init__(self):
        self.models = {}
        self.performance_history = {}
        self.learning_log = []
        
    def load_current_models(self):
        """Load current trained models"""
        
        positions = ['QB']  # Expand as you add more positions
        
        for position in positions:
            try:
                model_path = f'models/{position.lower()}_model_proper.pkl'
                model_data = joblib.load(model_path)
                
                self.models[position] = {
                    'model': model_data['model'],
                    'feature_columns': model_data['feature_columns'],
                    'training_date': model_data.get('training_date'),
                    'training_samples': model_data.get('training_samples', 0)
                }
                
                print(f"✅ Loaded {position} model from {model_data.get('training_date', 'unknown date')}")
                
            except Exception as e:
                print(f"❌ Error loading {position} model: {e}")
    
    def track_week_predictions(self, week_predictions, week_actuals):
        """
        Track how well our predictions performed vs actual results
        This is the feedback loop that enables learning!
        """
        
        print(f"\n📊 TRACKING WEEK PERFORMANCE")
        print("="*40)
        
        for position in week_predictions.keys():
            if position not in week_actuals:
                continue
            
            pred_data = week_predictions[position]
            actual_data = week_actuals[position]
            
            # Calculate accuracy metrics
            predictions = pred_data['predictions']
            actuals = actual_data['actual_scores']
            consensus = pred_data['consensus']
            
            # ML model performance
            ml_mae = mean_absolute_error(actuals, predictions)
            ml_correlation = np.corrcoef(actuals, predictions)[0, 1] if len(actuals) > 1 else 0
            
            # Consensus performance (for comparison)
            consensus_mae = mean_absolute_error(actuals, consensus)
            consensus_correlation = np.corrcoef(actuals, consensus)[0, 1] if len(actuals) > 1 else 0
            
            # Track performance
            week_performance = {
                'date': datetime.now().isoformat(),
                'position': position,
                'ml_mae': ml_mae,
                'ml_correlation': ml_correlation,
                'consensus_mae': consensus_mae,
                'consensus_correlation': consensus_correlation,
                'ml_advantage': consensus_mae - ml_mae,  # Positive = ML better
                'num_predictions': len(predictions)
            }
            
            # Store in performance history
            if position not in self.performance_history:
                self.performance_history[position] = []
            
            self.performance_history[position].append(week_performance)
            
            # Show results
            print(f"\n🎯 {position} Performance This Week:")
            print(f"  ML Model: MAE={ml_mae:.2f}, Correlation={ml_correlation:.3f}")
            print(f"  Consensus: MAE={consensus_mae:.2f}, Correlation={consensus_correlation:.3f}")
            
            if week_performance['ml_advantage'] > 0:
                print(f"  ✅ ML Better by {week_performance['ml_advantage']:.2f} points!")
            else:
                print(f"  ❌ Consensus Better by {-week_performance['ml_advantage']:.2f} points")
    
    def check_learning_progress(self, position, weeks_back=4):
        """Check if model is improving over time"""
        
        if position not in self.performance_history or len(self.performance_history[position]) < weeks_back:
            return None
        
        recent_performance = self.performance_history[position][-weeks_back:]
        
        # Calculate trend in ML advantage
        advantages = [p['ml_advantage'] for p in recent_performance]
        
        # Simple trend calculation
        if len(advantages) >= 2:
            recent_avg = np.mean(advantages[-2:])
            older_avg = np.mean(advantages[:-2]) if len(advantages) > 2 else advantages[0]
            trend = recent_avg - older_avg
        else:
            trend = 0
        
        avg_advantage = np.mean(advantages)
        
        return {
            'avg_advantage': avg_advantage,
            'trend': trend,
            'is_improving': trend > 0,
            'weeks_analyzed': len(advantages)
        }
    
    def retrain_model_with_new_data(self, position, new_week_data):
        """
        Retrain model with new week's data
        This is where continuous learning happens!
        """
        
        print(f"\n🔄 RETRAINING {position} MODEL WITH NEW DATA")
        print("="*50)
        
        try:
            # Load current training data
            current_model_data = self.models[position]
            
            # Load all historical data including new week
            recent_seasons = [2022, 2023, 2024]  # Adjust based on current year
            all_data = nfl.import_weekly_data(recent_seasons)
            
            # Filter to position
            pos_data = all_data[all_data['position'] == position].copy()
            
            # Create features using the same process as original training
            from Scripts.proper_ml_training import ProperMLTraining
            trainer = ProperMLTraining()
            
            # Generate updated features (including new week)
            updated_features = trainer.create_historical_features(all_data, position)
            
            if updated_features.empty:
                print(f"❌ No updated features for {position}")
                return False
            
            # Prepare training data
            feature_cols = current_model_data['feature_columns']
            X = updated_features[feature_cols].fillna(0)
            y = updated_features['target_fantasy_points']
            
            # Train updated model
            updated_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )
            
            updated_model.fit(X, y)
            
            # Validate improvement on recent data
            recent_data = updated_features[updated_features['season'] >= 2023]
            if len(recent_data) > 10:
                recent_X = recent_data[feature_cols].fillna(0)
                recent_y = recent_data['target_fantasy_points']
                
                # Old model performance
                old_predictions = current_model_data['model'].predict(recent_X)
                old_mae = mean_absolute_error(recent_y, old_predictions)
                
                # New model performance
                new_predictions = updated_model.predict(recent_X)
                new_mae = mean_absolute_error(recent_y, new_predictions)
                
                improvement = old_mae - new_mae
                
                print(f"📊 Validation Results:")
                print(f"  Old Model MAE: {old_mae:.2f}")
                print(f"  New Model MAE: {new_mae:.2f}")
                print(f"  Improvement: {improvement:.2f} points")
                
                # Only update if model actually improved
                if improvement > 0:
                    print(f"✅ Model improved! Saving updated model...")
                    
                    # Save updated model
                    updated_model_data = {
                        'model': updated_model,
                        'feature_columns': feature_cols,
                        'position': position,
                        'training_samples': len(updated_features),
                        'training_date': datetime.now().isoformat(),
                        'improvement_over_previous': improvement
                    }
                    
                    model_path = f'models/{position.lower()}_model_proper.pkl'
                    joblib.dump(updated_model_data, model_path)
                    
                    # Update in memory
                    self.models[position] = {
                        'model': updated_model,
                        'feature_columns': feature_cols,
                        'training_date': updated_model_data['training_date'],
                        'training_samples': updated_model_data['training_samples']
                    }
                    
                    # Log the learning event
                    self.learning_log.append({
                        'date': datetime.now().isoformat(),
                        'position': position,
                        'action': 'model_retrained',
                        'improvement': improvement,
                        'new_training_samples': len(updated_features)
                    })
                    
                    return True
                    
                else:
                    print(f"⚠️ No improvement detected. Keeping current model.")
                    return False
            
        except Exception as e:
            print(f"❌ Error retraining {position} model: {e}")
            return False
    
    def generate_learning_report(self):
        """Generate report on learning progress"""
        
        print(f"\n📈 LEARNING PROGRESS REPORT")
        print("="*50)
        
        for position in self.performance_history.keys():
            print(f"\n🎯 {position} Learning Summary:")
            
            # Recent performance trend
            progress = self.check_learning_progress(position)
            
            if progress:
                print(f"  📊 Average ML Advantage: {progress['avg_advantage']:+.2f} points")
                print(f"  📈 Recent Trend: {progress['trend']:+.2f}")
                
                if progress['is_improving']:
                    print(f"  ✅ Model is IMPROVING over time!")
                else:
                    print(f"  ⚠️ Model performance declining")
            
            # Show all weeks
            history = self.performance_history[position]
            print(f"  📅 Performance History ({len(history)} weeks):")
            
            for i, week in enumerate(history[-4:], 1):  # Show last 4 weeks
                advantage = week['ml_advantage']
                status = "✅" if advantage > 0 else "❌"
                print(f"    Week {i}: {status} ML Advantage = {advantage:+.2f}")
        
        # Show learning events
        if self.learning_log:
            print(f"\n🔄 Learning Events:")
            for event in self.learning_log[-3:]:  # Show last 3 events
                print(f"  {event['date'][:10]}: {event['action']} ({event['position']}) - Improvement: {event['improvement']:+.2f}")
    
    def automated_weekly_update(self, week_results):
        """
        Automated function to run after each NFL week
        This is the main continuous learning workflow
        """
        
        print(f"\n🤖 AUTOMATED WEEKLY LEARNING UPDATE")
        print("="*60)
        
        # Step 1: Track this week's performance
        if 'predictions' in week_results and 'actuals' in week_results:
            self.track_week_predictions(week_results['predictions'], week_results['actuals'])
        
        # Step 2: Check if models need retraining
        for position in self.models.keys():
            progress = self.check_learning_progress(position)
            
            if progress and not progress['is_improving'] and progress['avg_advantage'] < -1.0:
                print(f"\n⚠️ {position} model performance declining. Retraining...")
                self.retrain_model_with_new_data(position, week_results.get('new_data'))
        
        # Step 3: Generate learning report
        self.generate_learning_report()
        
        # Step 4: Save learning state
        self.save_learning_state()
    
    def save_learning_state(self):
        """Save learning history and progress"""
        
        learning_state = {
            'performance_history': self.performance_history,
            'learning_log': self.learning_log,
            'last_updated': datetime.now().isoformat()
        }
        
        os.makedirs('models', exist_ok=True)
        
        import json
        with open('models/learning_state.json', 'w') as f:
            json.dump(learning_state, f, indent=2)
        
        print(f"💾 Learning state saved to models/learning_state.json")
    
    def load_learning_state(self):
        """Load previous learning history"""
        
        try:
            import json
            with open('models/learning_state.json', 'r') as f:
                learning_state = json.load(f)
            
            self.performance_history = learning_state.get('performance_history', {})
            self.learning_log = learning_state.get('learning_log', [])
            
            print(f"✅ Loaded learning state from {learning_state.get('last_updated', 'unknown')}")
            
        except Exception as e:
            print(f"⚠️ No previous learning state found: {e}")

def demo_continuous_learning():
    """Demo the continuous learning system"""
    
    print("🧪 DEMO: CONTINUOUS LEARNING SYSTEM")
    print("="*50)
    
    # Initialize learning system
    learner = ContinuousLearningSystem()
    learner.load_current_models()
    learner.load_learning_state()
    
    if not learner.models:
        print("❌ No models loaded! Run proper_ml_training.py first")
        return
    
    # Simulate week results (in real usage, this would come from actual NFL results)
    simulated_week_results = {
        'predictions': {
            'QB': {
                'predictions': np.array([25.3, 18.7, 22.1]),  # Your ML predictions
                'consensus': np.array([22.4, 16.4, 20.2])     # Consensus projections
            }
        },
        'actuals': {
            'QB': {
                'actual_scores': np.array([24.1, 19.2, 21.8])  # Actual fantasy scores
            }
        }
    }
    
    # Run automated learning update
    learner.automated_weekly_update(simulated_week_results)
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. After each real NFL week, call automated_weekly_update()")
    print(f"2. System will automatically track performance and retrain if needed")
    print(f"3. Models will continuously improve over time!")

if __name__ == "__main__":
    demo_continuous_learning()