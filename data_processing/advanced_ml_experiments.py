"""
Advanced Machine Learning Experiments for Compressive Strength Prediction
Goal: Achieve R² ≥ 0.85 through comprehensive experimentation

This script includes:
1. Advanced feature engineering
2. Multiple ML algorithms
3. Hyperparameter optimization
4. Ensemble methods
5. Cross-validation strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CompressiveStrengthPredictor:
    def __init__(self, data_path="data_processing/merged_database_cleaned.csv"):
        """Initialize the predictor with data loading and preprocessing."""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        print("=== Loading and Preparing Data ===")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape}")
        
        # Prepare features and target
        exclude_cols = ['Compressive strength (MPa)', 'Source', 'Phases present', 'Unique ID', 'Composition']
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        self.X = self.df[feature_cols].copy()
        self.y = self.df['Compressive strength (MPa)'].copy()
        
        # Handle missing values
        self.X = self.X.fillna(self.X.median())
        
        print(f"Features: {self.X.shape[1]}")
        print(f"Target range: {self.y.min():.1f} - {self.y.max():.1f} MPa")
        
        return self.X, self.y
    
    def advanced_feature_engineering(self):
        """Create advanced engineered features."""
        print("\n=== Advanced Feature Engineering ===")
        
        X_enhanced = self.X.copy()
        
        # 1. Element interaction features
        elements = ['Cu', 'Fe', 'Al', 'Ni', 'Co', 'Cr', 'Ti', 'Mn', 'Mo']
        existing_elements = [e for e in elements if e in X_enhanced.columns]
        
        print("Creating element interaction features...")
        for i, elem1 in enumerate(existing_elements):
            for elem2 in existing_elements[i+1:]:
                # Multiplicative interactions
                X_enhanced[f'{elem1}x{elem2}'] = X_enhanced[elem1] * X_enhanced[elem2]
                # Ratio interactions (avoid division by zero)
                X_enhanced[f'{elem1}/{elem2}'] = np.where(
                    X_enhanced[elem2] > 0.001,
                    X_enhanced[elem1] / X_enhanced[elem2],
                    0
                )
        
        # 2. Polynomial features for key elements
        key_elements = ['Cu', 'Fe', 'Al', 'Ni', 'Mo']
        existing_key = [e for e in key_elements if e in X_enhanced.columns]
        
        print("Creating polynomial features...")
        for elem in existing_key:
            X_enhanced[f'{elem}^2'] = X_enhanced[elem] ** 2
            X_enhanced[f'{elem}^3'] = X_enhanced[elem] ** 3
            X_enhanced[f'sqrt_{elem}'] = np.sqrt(X_enhanced[elem])
            X_enhanced[f'log_{elem}'] = np.log1p(X_enhanced[elem])
        
        # 3. Composite features
        print("Creating composite features...")
        
        # Total transition metals
        transition_metals = ['Fe', 'Co', 'Ni', 'Cr', 'Mn', 'Mo', 'Ti', 'V']
        existing_tm = [tm for tm in transition_metals if tm in X_enhanced.columns]
        if existing_tm:
            X_enhanced['Total_Transition_Metals'] = X_enhanced[existing_tm].sum(axis=1)
        
        # Hardness-based features
        if 'Hardness (HVN)' in X_enhanced.columns:
            X_enhanced['Hardness_log'] = np.log1p(X_enhanced['Hardness (HVN)'])
            X_enhanced['Hardness_sqrt'] = np.sqrt(X_enhanced['Hardness (HVN)'])
        
        # Density-based features
        if 'Density (g/cm3)' in X_enhanced.columns:
            X_enhanced['Density_inv'] = 1 / (X_enhanced['Density (g/cm3)'] + 0.1)
        
        # Element complexity features
        if 'Num_Elements' in X_enhanced.columns:
            X_enhanced['Element_Complexity'] = X_enhanced['Num_Elements'] ** 2
            X_enhanced['Element_Diversity'] = X_enhanced['Num_Elements'] / X_enhanced['Total_Composition']
        
        # 4. Phase-based interactions
        phase_cols = ['FCC', 'BCC', 'HCP', 'IM']
        existing_phases = [p for p in phase_cols if p in X_enhanced.columns]
        
        if len(existing_phases) > 1:
            for i, phase1 in enumerate(existing_phases):
                for phase2 in existing_phases[i+1:]:
                    X_enhanced[f'{phase1}x{phase2}'] = X_enhanced[phase1] * X_enhanced[phase2]
        
        print(f"Enhanced features: {X_enhanced.shape[1]} (added {X_enhanced.shape[1] - self.X.shape[1]})")
        
        return X_enhanced
    
    def feature_selection(self, X_enhanced, method='combined', k=50):
        """Select the best features using multiple methods."""
        print(f"\n=== Feature Selection (top {k} features) ===")
        
        if method == 'univariate':
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X_enhanced, self.y)
            selected_features = X_enhanced.columns[selector.get_support()]
            
        elif method == 'rfe':
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            selector = RFE(estimator=rf, n_features_to_select=k)
            X_selected = selector.fit_transform(X_enhanced, self.y)
            selected_features = X_enhanced.columns[selector.get_support()]
            
        elif method == 'combined':
            # Use Random Forest feature importance + correlation
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_enhanced, self.y)
            
            # Get feature importance scores
            importance_scores = pd.Series(rf.feature_importances_, index=X_enhanced.columns)
            
            # Get correlation scores
            corr_scores = X_enhanced.corrwith(self.y).abs()
            
            # Combined score (weighted average)
            combined_scores = 0.7 * importance_scores + 0.3 * corr_scores
            selected_features = combined_scores.nlargest(k).index
            X_selected = X_enhanced[selected_features]
        
        print(f"Selected features: {list(selected_features[:10])}...")
        return X_selected, selected_features
    
    def prepare_train_test_split(self, X_selected, test_size=0.2):
        """Prepare train-test split with scaling."""
        print(f"\n=== Preparing Train-Test Split ===")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_selected, self.y, test_size=test_size, random_state=42, stratify=None
        )
        
        # Scale features
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def experiment_random_forest(self):
        """Experiment with Random Forest with hyperparameter tuning."""
        print("\n=== Random Forest Experiment ===")
        
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Randomized search for efficiency
        rf_search = RandomizedSearchCV(
            rf, param_grid, n_iter=50, cv=5, scoring='r2',
            random_state=42, n_jobs=-1, verbose=1
        )
        
        rf_search.fit(self.X_train, self.y_train)
        
        # Best model predictions
        y_pred = rf_search.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        self.results['Random Forest'] = {
            'model': rf_search.best_estimator_,
            'r2': r2,
            'rmse': rmse,
            'params': rf_search.best_params_
        }
        
        print(f"Random Forest R²: {r2:.4f}, RMSE: {rmse:.2f}")
        return rf_search.best_estimator_
    
    def experiment_xgboost(self):
        """Experiment with XGBoost with hyperparameter tuning."""
        print("\n=== XGBoost Experiment ===")
        
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        xgb_search = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=50, cv=5, scoring='r2',
            random_state=42, n_jobs=-1, verbose=1
        )
        
        xgb_search.fit(self.X_train_scaled, self.y_train)
        
        y_pred = xgb_search.predict(self.X_test_scaled)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        self.results['XGBoost'] = {
            'model': xgb_search.best_estimator_,
            'r2': r2,
            'rmse': rmse,
            'params': xgb_search.best_params_
        }
        
        print(f"XGBoost R²: {r2:.4f}, RMSE: {rmse:.2f}")
        return xgb_search.best_estimator_
    
    def experiment_lightgbm(self):
        """Experiment with LightGBM with hyperparameter tuning."""
        print("\n=== LightGBM Experiment ===")
        
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2],
            'num_leaves': [31, 50, 100]
        }
        
        lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        
        lgb_search = RandomizedSearchCV(
            lgb_model, param_grid, n_iter=50, cv=5, scoring='r2',
            random_state=42, n_jobs=-1, verbose=1
        )
        
        lgb_search.fit(self.X_train_scaled, self.y_train)
        
        y_pred = lgb_search.predict(self.X_test_scaled)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        self.results['LightGBM'] = {
            'model': lgb_search.best_estimator_,
            'r2': r2,
            'rmse': rmse,
            'params': lgb_search.best_params_
        }
        
        print(f"LightGBM R²: {r2:.4f}, RMSE: {rmse:.2f}")
        return lgb_search.best_estimator_
    
    def experiment_neural_network(self):
        """Experiment with Neural Network."""
        print("\n=== Neural Network Experiment ===")
        
        param_grid = {
            'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100), (300, 150, 75)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }
        
        mlp = MLPRegressor(max_iter=1000, random_state=42)
        
        mlp_search = RandomizedSearchCV(
            mlp, param_grid, n_iter=30, cv=5, scoring='r2',
            random_state=42, n_jobs=-1, verbose=1
        )
        
        mlp_search.fit(self.X_train_scaled, self.y_train)
        
        y_pred = mlp_search.predict(self.X_test_scaled)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        self.results['Neural Network'] = {
            'model': mlp_search.best_estimator_,
            'r2': r2,
            'rmse': rmse,
            'params': mlp_search.best_params_
        }
        
        print(f"Neural Network R²: {r2:.4f}, RMSE: {rmse:.2f}")
        return mlp_search.best_estimator_
    
    def experiment_svr(self):
        """Experiment with Support Vector Regression."""
        print("\n=== Support Vector Regression Experiment ===")
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'kernel': ['rbf', 'poly']
        }
        
        svr = SVR()
        
        svr_search = RandomizedSearchCV(
            svr, param_grid, n_iter=30, cv=5, scoring='r2',
            random_state=42, n_jobs=-1, verbose=1
        )
        
        svr_search.fit(self.X_train_scaled, self.y_train)
        
        y_pred = svr_search.predict(self.X_test_scaled)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        self.results['SVR'] = {
            'model': svr_search.best_estimator_,
            'r2': r2,
            'rmse': rmse,
            'params': svr_search.best_params_
        }
        
        print(f"SVR R²: {r2:.4f}, RMSE: {rmse:.2f}")
        return svr_search.best_estimator_
    
    def experiment_ensemble_methods(self):
        """Experiment with ensemble methods."""
        print("\n=== Ensemble Methods Experiment ===")
        
        # Get best models from previous experiments
        models = []
        weights = []
        
        for name, result in self.results.items():
            if result['r2'] > 0.6:  # Only include decent models
                models.append(result['model'])
                weights.append(result['r2'])  # Weight by performance
        
        if len(models) < 2:
            print("Not enough good models for ensemble")
            return None
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Create weighted ensemble predictions
        ensemble_preds = np.zeros(len(self.y_test))
        
        for i, model in enumerate(models):
            if hasattr(model, 'predict'):
                if 'Neural Network' in str(type(model)) or 'SVR' in str(type(model)):
                    pred = model.predict(self.X_test_scaled)
                else:
                    pred = model.predict(self.X_test)
                ensemble_preds += weights[i] * pred
        
        r2 = r2_score(self.y_test, ensemble_preds)
        rmse = np.sqrt(mean_squared_error(self.y_test, ensemble_preds))
        
        self.results['Weighted Ensemble'] = {
            'r2': r2,
            'rmse': rmse,
            'weights': weights,
            'models': [str(type(m).__name__) for m in models]
        }
        
        print(f"Weighted Ensemble R²: {r2:.4f}, RMSE: {rmse:.2f}")
        return ensemble_preds
    
    def cross_validation_analysis(self, X_selected):
        """Perform cross-validation analysis on best models."""
        print("\n=== Cross-Validation Analysis ===")
        
        cv_results = {}
        
        # Test top performing models with CV
        for name, result in self.results.items():
            if name != 'Weighted Ensemble' and result['r2'] > 0.7:
                model = result['model']
                
                if 'Neural Network' in str(type(model)) or 'SVR' in str(type(model)):
                    X_cv = self.scaler.fit_transform(X_selected)
                else:
                    X_cv = X_selected
                
                cv_scores = cross_val_score(model, X_cv, self.y, cv=10, scoring='r2')
                cv_results[name] = {
                    'mean_r2': cv_scores.mean(),
                    'std_r2': cv_scores.std(),
                    'scores': cv_scores
                }
                
                print(f"{name} CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return cv_results
    
    def plot_results(self):
        """Plot experimental results."""
        print("\n=== Plotting Results ===")
        
        # Extract R² scores
        models = []
        r2_scores = []
        
        for name, result in self.results.items():
            models.append(name)
            r2_scores.append(result['r2'])
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(models, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'orange'])
        plt.title('Model Performance Comparison (R² Score)', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add target line at R² = 0.85
        plt.axhline(y=0.85, color='red', linestyle='--', linewidth=2, label='Target R² = 0.85')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('data_processing/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def generate_report(self):
        """Generate comprehensive experiment report."""
        print("\n=== Generating Experiment Report ===")
        
        # Sort results by R² score
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        report = []
        report.append("# Compressive Strength Prediction - ML Experiment Results")
        report.append("=" * 60)
        report.append(f"\nDataset: {self.df.shape[0]} samples, {self.X.shape[1]} original features")
        report.append(f"Target: Compressive Strength (Range: {self.y.min():.1f} - {self.y.max():.1f} MPa)")
        report.append(f"Goal: Achieve R² ≥ 0.85")
        
        report.append("\n## Model Performance Ranking:")
        report.append("-" * 40)
        
        best_r2 = 0
        best_model = None
        
        for i, (name, result) in enumerate(sorted_results, 1):
            r2 = result['r2']
            rmse = result['rmse']
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = name
            
            status = "✅ TARGET ACHIEVED!" if r2 >= 0.85 else "❌ Below target"
            report.append(f"{i}. {name}")
            report.append(f"   R² Score: {r2:.4f} | RMSE: {rmse:.2f} MPa | {status}")
            
            if 'params' in result:
                report.append(f"   Best Parameters: {result['params']}")
            report.append("")
        
        # Summary
        report.append("\n## Summary:")
        report.append("-" * 20)
        report.append(f"🏆 Best Model: {best_model}")
        report.append(f"🎯 Best R² Score: {best_r2:.4f}")
        
        if best_r2 >= 0.85:
            report.append("🎉 TARGET ACHIEVED! R² ≥ 0.85")
        else:
            report.append(f"⚠️  Target not reached. Gap: {0.85 - best_r2:.4f}")
            report.append("\n## Recommendations for Improvement:")
            report.append("- Try more advanced feature engineering")
            report.append("- Collect more training data")
            report.append("- Experiment with deep learning models")
            report.append("- Consider domain-specific feature transformations")
        
        # Save report
        with open('data_processing/ml_experiment_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
        return '\n'.join(report)

def main():
    """Main experimental pipeline."""
    print("🚀 Starting Advanced ML Experiments for Compressive Strength Prediction")
    print("Goal: Achieve R² ≥ 0.85")
    print("=" * 80)
    
    # Initialize predictor
    predictor = CompressiveStrengthPredictor()
    
    # Load and prepare data
    X, y = predictor.load_and_prepare_data()
    
    # Advanced feature engineering
    X_enhanced = predictor.advanced_feature_engineering()
    
    # Feature selection
    X_selected, selected_features = predictor.feature_selection(X_enhanced, method='combined', k=60)
    
    # Prepare train-test split
    X_train_scaled, X_test_scaled, y_train, y_test = predictor.prepare_train_test_split(X_selected)
    
    # Run all experiments
    print("\n🔬 Running ML Experiments...")
    
    # 1. Random Forest
    predictor.experiment_random_forest()
    
    # 2. XGBoost
    predictor.experiment_xgboost()
    
    # 3. LightGBM
    predictor.experiment_lightgbm()
    
    # 4. Neural Network
    predictor.experiment_neural_network()
    
    # 5. Support Vector Regression
    predictor.experiment_svr()
    
    # 6. Ensemble Methods
    predictor.experiment_ensemble_methods()
    
    # Cross-validation analysis
    cv_results = predictor.cross_validation_analysis(X_selected)
    
    # Plot results
    predictor.plot_results()
    
    # Generate comprehensive report
    predictor.generate_report()
    
    print("\n🎯 Experiment Complete!")
    print("Check 'data_processing/ml_experiment_report.txt' for detailed results")
    print("Check 'data_processing/model_comparison.png' for performance visualization")

if __name__ == "__main__":
    main()