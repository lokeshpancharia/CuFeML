#!/usr/bin/env python3
"""
Model Training Script for Cu-Fe HEA Compressive Strength Prediction

This script loads the engineered features and trains multiple regression models
including Random Forest, XGBoost, SVR, and ensemble methods with cross-validation
and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

class ModelTrainer:
    """
    Comprehensive model training class for Cu-Fe HEA compressive strength prediction.
    """
    
    def __init__(self, data_path='features_data.csv', random_state=42):
        """
        Initialize the model trainer.
        
        Args:
            data_path (str): Path to the features data CSV file
            random_state (int): Random state for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
        # Create results directory
        Path('models').mkdir(exist_ok=True)
        Path('results').mkdir(exist_ok=True)
        
    def load_data(self):
        """Load and prepare the features data for training."""
        print("Loading features data...")
        
        # Load the data
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded dataset with shape: {self.df.shape}")
        
        # Separate features and target
        target_col = 'compressive_strength'
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Remove non-feature columns that shouldn't be used for prediction
        exclude_cols = [target_col, 'Test Type']  # Test Type has many missing values
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        self.X = self.df[feature_cols].copy()
        self.y = self.df[target_col].copy()
        
        # Handle any remaining missing values in features
        if self.X.isnull().sum().sum() > 0:
            print("Handling missing values in features...")
            self.X = self.X.fillna(self.X.median())
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        print(f"Target statistics:")
        print(self.y.describe())
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=None
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
    def define_models(self):
        """Define the regression models and their hyperparameter grids."""
        print("Defining models and hyperparameter grids...")
        
        # Random Forest
        rf_param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [10, 20, None],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }
        
        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(random_state=self.random_state))
        ])
        
        self.models['Random Forest'] = {
            'pipeline': rf_pipeline,
            'param_grid': rf_param_grid
        }
        
        # Support Vector Regression
        svr_param_grid = {
            'regressor__C': [0.1, 1, 10, 100],
            'regressor__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'regressor__epsilon': [0.01, 0.1, 0.2]
        }
        
        svr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SVR(kernel='rbf'))
        ])
        
        self.models['SVR'] = {
            'pipeline': svr_pipeline,
            'param_grid': svr_param_grid
        }
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            xgb_param_grid = {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [3, 6, 10],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__subsample': [0.8, 0.9, 1.0]
            }
            
            xgb_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', xgb.XGBRegressor(random_state=self.random_state))
            ])
            
            self.models['XGBoost'] = {
                'pipeline': xgb_pipeline,
                'param_grid': xgb_param_grid
            }
        
    def train_models(self):
        """Train all models with hyperparameter tuning and cross-validation."""
        print("Training models with hyperparameter tuning...")
        
        # Cross-validation setup
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for model_name, model_config in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model_config['pipeline'],
                model_config['param_grid'],
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit the model
            grid_search.fit(self.X_train, self.y_train)
            
            # Store the best model
            best_model = grid_search.best_estimator_
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                best_model, self.X_train, self.y_train,
                cv=cv, scoring='neg_mean_squared_error'
            )
            
            # Test predictions
            y_pred = best_model.predict(self.X_test)
            
            # Calculate metrics
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            test_mae = mean_absolute_error(self.y_test, y_pred)
            test_r2 = r2_score(self.y_test, y_pred)
            
            cv_rmse = np.sqrt(-cv_scores)
            
            # Store results
            self.results[model_name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'cv_scores': cv_scores.tolist(),
                'predictions': y_pred.tolist()
            }
            
            print(f"{model_name} Results:")
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  CV RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
            print(f"  Test RMSE: {test_rmse:.2f}")
            print(f"  Test MAE: {test_mae:.2f}")
            print(f"  Test R²: {test_r2:.3f}")
            
    def create_ensemble(self):
        """Create an ensemble model from the best performing individual models."""
        print("\nCreating ensemble model...")
        
        # Get the trained models (without the results metadata)
        trained_models = [(name, results['model']) for name, results in self.results.items()]
        
        # Create voting regressor
        ensemble = VotingRegressor(trained_models)
        ensemble.fit(self.X_train, self.y_train)
        
        # Cross-validation for ensemble
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            ensemble, self.X_train, self.y_train,
            cv=cv, scoring='neg_mean_squared_error'
        )
        
        # Test predictions
        y_pred = ensemble.predict(self.X_test)
        
        # Calculate metrics
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        test_mae = mean_absolute_error(self.y_test, y_pred)
        test_r2 = r2_score(self.y_test, y_pred)
        
        cv_rmse = np.sqrt(-cv_scores)
        
        # Store ensemble results
        self.results['Ensemble'] = {
            'model': ensemble,
            'best_params': 'Voting ensemble of all models',
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'cv_scores': cv_scores.tolist(),
            'predictions': y_pred.tolist()
        }
        
        print(f"Ensemble Results:")
        print(f"  CV RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
        print(f"  Test RMSE: {test_rmse:.2f}")
        print(f"  Test MAE: {test_mae:.2f}")
        print(f"  Test R²: {test_r2:.3f}")
        
    def select_best_model(self):
        """Select the best performing model based on cross-validation RMSE."""
        print("\nSelecting best model...")
        
        best_cv_rmse = float('inf')
        best_model_name = None
        
        for model_name, results in self.results.items():
            cv_rmse = results['cv_rmse_mean']
            if cv_rmse < best_cv_rmse:
                best_cv_rmse = cv_rmse
                best_model_name = model_name
        
        self.best_model = {
            'name': best_model_name,
            'model': self.results[best_model_name]['model'],
            'cv_rmse': best_cv_rmse
        }
        
        print(f"Best model: {best_model_name}")
        print(f"Best CV RMSE: {best_cv_rmse:.2f}")
        
    def save_models(self):
        """Save the trained models and results."""
        print("\nSaving models and results...")
        
        # Save individual models
        for model_name, results in self.results.items():
            model_filename = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(results['model'], f)
            print(f"Saved {model_name} model to {model_filename}")
        
        # Save best model separately
        if self.best_model:
            best_model_filename = "models/best_model.pkl"
            with open(best_model_filename, 'wb') as f:
                pickle.dump(self.best_model['model'], f)
            print(f"Saved best model to {best_model_filename}")
        
        # Prepare results for JSON serialization (remove model objects)
        results_for_json = {}
        for model_name, results in self.results.items():
            results_for_json[model_name] = {
                key: value for key, value in results.items() 
                if key != 'model'  # Exclude the model object
            }
        
        # Add metadata
        results_for_json['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'dataset_shape': self.df.shape,
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'feature_count': self.X.shape[1],
            'best_model': self.best_model['name'] if self.best_model else None,
            'random_state': self.random_state
        }
        
        # Save results
        results_filename = "results/model_comparison_results.json"
        with open(results_filename, 'w') as f:
            json.dump(results_for_json, f, indent=2)
        print(f"Saved model comparison results to {results_filename}")
        
    def print_summary(self):
        """Print a summary of all model performances."""
        print("\n" + "="*80)
        print("MODEL TRAINING SUMMARY")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'CV RMSE': f"{results['cv_rmse_mean']:.2f} ± {results['cv_rmse_std']:.2f}",
                'Test RMSE': f"{results['test_rmse']:.2f}",
                'Test MAE': f"{results['test_mae']:.2f}",
                'Test R²': f"{results['test_r2']:.3f}"
            })
        
        # Convert to DataFrame for nice printing
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        if self.best_model:
            print(f"\nBest performing model: {self.best_model['name']}")
            print(f"Best CV RMSE: {self.best_model['cv_rmse']:.2f}")
        
        print("\nFiles saved:")
        print("- Individual models: models/[model_name]_model.pkl")
        print("- Best model: models/best_model.pkl")
        print("- Results: results/model_comparison_results.json")
        
    def run_training_pipeline(self):
        """Run the complete model training pipeline."""
        print("Starting Cu-Fe HEA Compressive Strength Model Training Pipeline")
        print("="*80)
        
        try:
            # Load data
            self.load_data()
            
            # Define models
            self.define_models()
            
            # Train models
            self.train_models()
            
            # Create ensemble
            self.create_ensemble()
            
            # Select best model
            self.select_best_model()
            
            # Save models and results
            self.save_models()
            
            # Print summary
            self.print_summary()
            
            print("\nModel training completed successfully!")
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise


def main():
    """Main function to run the model training."""
    trainer = ModelTrainer()
    trainer.run_training_pipeline()


if __name__ == "__main__":
    main()