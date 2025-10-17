"""
Best ML Experiments for Compressive Strength Prediction
Focused approach to achieve R² >= 0.85

This version focuses on:
1. Proven feature engineering techniques
2. Optimal hyperparameter ranges
3. Robust model validation
4. Simple but effective ensemble methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class BestCompressiveStrengthPredictor:
    def __init__(self, data_path="data_processing/merged_database_cleaned.csv"):
        """Initialize the best predictor."""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset with careful preprocessing."""
        print("=== Loading and Preparing Data ===")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape}")
        
        # Prepare features and target
        exclude_cols = ['Compressive strength (MPa)', 'Source', 'Phases present', 'Unique ID', 'Composition']
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        self.X = self.df[feature_cols].copy()
        self.y = self.df['Compressive strength (MPa)'].copy()
        
        # Handle missing values carefully
        for col in self.X.columns:
            if self.X[col].isnull().sum() > 0:
                if col in ['Hardness (HVN)', 'Plasticity (%) - Compressive']:
                    # Use median for important features
                    self.X[col] = self.X[col].fillna(self.X[col].median())
                else:
                    # Use 0 for composition features
                    self.X[col] = self.X[col].fillna(0)
        
        # Remove constant and near-constant features
        for col in self.X.columns:
            if self.X[col].nunique() <= 1:
                self.X = self.X.drop(columns=[col])
                print(f"Removed constant feature: {col}")
        
        print(f"Final features: {self.X.shape[1]}")
        print(f"Target range: {self.y.min():.1f} - {self.y.max():.1f} MPa")
        
        return self.X, self.y
    
    def create_best_features(self):
        """Create the most effective engineered features."""
        print("\n=== Creating Best Features ===")
        
        X_best = self.X.copy()
        
        # Key elements based on materials science
        key_elements = ['Cu', 'Fe', 'Al', 'Ni', 'Co', 'Cr', 'Mo', 'Ti', 'Mn']
        existing_elements = [e for e in key_elements if e in X_best.columns]
        
        print(f"Working with elements: {existing_elements}")
        
        # 1. Most important element interactions
        important_pairs = [
            ('Cu', 'Fe'), ('Al', 'Ni'), ('Co', 'Cr'), ('Fe', 'Ni'), 
            ('Cu', 'Al'), ('Mo', 'Fe'), ('Ti', 'Al')
        ]
        
        for elem1, elem2 in important_pairs:
            if elem1 in X_best.columns and elem2 in X_best.columns:
                # Product interaction
                X_best[f'{elem1}_x_{elem2}'] = X_best[elem1] * X_best[elem2]
                # Ratio interaction (safe)
                X_best[f'{elem1}_div_{elem2}'] = np.where(
                    X_best[elem2] > 0.001,
                    X_best[elem1] / X_best[elem2],
                    0
                )
        
        # 2. Polynomial features for key elements
        key_poly_elements = ['Cu', 'Fe', 'Al', 'Ni', 'Mo']
        for elem in key_poly_elements:
            if elem in X_best.columns:
                X_best[f'{elem}_squared'] = X_best[elem] ** 2
                X_best[f'{elem}_cubed'] = X_best[elem] ** 3
                X_best[f'{elem}_sqrt'] = np.sqrt(X_best[elem])
                X_best[f'{elem}_log'] = np.log1p(X_best[elem])
        
        # 3. Hardness transformations (most important feature)
        if 'Hardness (HVN)' in X_best.columns:
            hardness = X_best['Hardness (HVN)']
            
            X_best['Hardness_log'] = np.log1p(hardness)
            X_best['Hardness_sqrt'] = np.sqrt(hardness)
            X_best['Hardness_squared'] = hardness ** 2
            X_best['Hardness_cubed'] = hardness ** 3
            
            # Hardness interactions with key elements
            for elem in ['Cu', 'Fe', 'Al', 'Mo', 'Ni']:
                if elem in X_best.columns:
                    X_best[f'Hardness_x_{elem}'] = hardness * X_best[elem]
        
        # 4. Composition complexity features
        if 'Num_Elements' in X_best.columns:
            num_elem = X_best['Num_Elements']
            X_best['Element_Complexity'] = num_elem ** 2
            X_best['Element_Entropy'] = num_elem * np.log1p(num_elem)
            
            if 'Total_Composition' in X_best.columns:
                X_best['Element_Diversity'] = num_elem / (X_best['Total_Composition'] + 0.001)
        
        # 5. Phase features
        phase_cols = ['FCC', 'BCC', 'HCP', 'IM']
        existing_phases = [p for p in phase_cols if p in X_best.columns]
        
        if len(existing_phases) >= 2:
            # Key phase interactions
            if 'FCC' in existing_phases and 'BCC' in existing_phases:
                X_best['FCC_x_BCC'] = X_best['FCC'] * X_best['BCC']
        
        print(f"Enhanced features: {X_best.shape[1]} (added {X_best.shape[1] - self.X.shape[1]})")
        
        return X_best
    
    def select_best_features(self, X_best, k=50):
        """Select the most predictive features."""
        print(f"\n=== Selecting Best {k} Features ===")
        
        # Method 1: Random Forest importance
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_best, self.y)
        rf_importance = pd.Series(rf.feature_importances_, index=X_best.columns)
        
        # Method 2: Correlation with target
        correlations = pd.Series(index=X_best.columns, dtype=float)
        for col in X_best.columns:
            correlations[col] = abs(X_best[col].corr(self.y))
        
        # Method 3: Mutual information
        mi_scores = mutual_info_regression(X_best, self.y, random_state=42)
        mi_importance = pd.Series(mi_scores, index=X_best.columns)
        mi_importance = mi_importance / mi_importance.max()
        
        # Combined scoring (equal weights for simplicity)
        combined_scores = (rf_importance + correlations + mi_importance) / 3
        
        # Select top k features
        selected_features = combined_scores.nlargest(k).index
        X_selected = X_best[selected_features]
        
        print(f"Top 10 selected features:")
        for i, feature in enumerate(selected_features[:10], 1):
            print(f"{i:2d}. {feature} (score: {combined_scores[feature]:.4f})")
        
        return X_selected, selected_features
    
    def prepare_data_splits(self, X_selected, test_size=0.2):
        """Prepare train-test splits with proper scaling."""
        print(f"\n=== Preparing Data Splits ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, self.y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def best_xgboost_experiment(self, X_train, X_test, y_train, y_test):
        """Best XGBoost experiment with focused hyperparameter tuning."""
        print("\n=== Best XGBoost Experiment ===")
        
        # Focused parameter grid
        param_grid = {
            'n_estimators': [1000, 1500, 2000],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        # Use GridSearchCV for thorough search
        xgb_search = GridSearchCV(
            xgb_model, param_grid, cv=5, scoring='r2',
            n_jobs=-1, verbose=1
        )
        
        xgb_search.fit(X_train, y_train)
        
        y_pred = xgb_search.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.results['Best XGBoost'] = {
            'model': xgb_search.best_estimator_,
            'r2': r2,
            'rmse': rmse,
            'params': xgb_search.best_params_,
            'predictions': y_pred
        }
        
        print(f"Best XGBoost R²: {r2:.4f}, RMSE: {rmse:.2f}")
        print(f"Best params: {xgb_search.best_params_}")
        return xgb_search.best_estimator_
    
    def best_lightgbm_experiment(self, X_train, X_test, y_train, y_test):
        """Best LightGBM experiment."""
        print("\n=== Best LightGBM Experiment ===")
        
        param_grid = {
            'n_estimators': [1000, 1500, 2000],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2],
            'num_leaves': [31, 50, 70]
        }
        
        lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        
        lgb_search = RandomizedSearchCV(
            lgb_model, param_grid, n_iter=50, cv=5, scoring='r2',
            random_state=42, n_jobs=-1, verbose=1
        )
        
        lgb_search.fit(X_train, y_train)
        
        y_pred = lgb_search.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.results['Best LightGBM'] = {
            'model': lgb_search.best_estimator_,
            'r2': r2,
            'rmse': rmse,
            'params': lgb_search.best_params_,
            'predictions': y_pred
        }
        
        print(f"Best LightGBM R²: {r2:.4f}, RMSE: {rmse:.2f}")
        return lgb_search.best_estimator_
    
    def best_random_forest_experiment(self, X_train, X_test, y_train, y_test):
        """Best Random Forest experiment."""
        print("\n=== Best Random Forest Experiment ===")
        
        param_grid = {
            'n_estimators': [1000, 1500, 2000],
            'max_depth': [None, 20, 25],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 0.3, 0.5],
            'bootstrap': [True]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        rf_search = RandomizedSearchCV(
            rf, param_grid, n_iter=40, cv=5, scoring='r2',
            random_state=42, n_jobs=-1, verbose=1
        )
        
        rf_search.fit(X_train, y_train)
        
        y_pred = rf_search.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.results['Best Random Forest'] = {
            'model': rf_search.best_estimator_,
            'r2': r2,
            'rmse': rmse,
            'params': rf_search.best_params_,
            'predictions': y_pred
        }
        
        print(f"Best Random Forest R²: {r2:.4f}, RMSE: {rmse:.2f}")
        return rf_search.best_estimator_
    
    def create_best_ensemble(self, y_test):
        """Create the best ensemble from top models."""
        print("\n=== Creating Best Ensemble ===")
        
        # Get models with R² > 0.7
        good_models = {name: result for name, result in self.results.items() 
                      if result['r2'] > 0.7}
        
        if len(good_models) < 2:
            print("Not enough good models for ensemble")
            return None
        
        # Simple average ensemble
        ensemble_pred = np.zeros(len(y_test))
        for name, result in good_models.items():
            ensemble_pred += result['predictions']
        ensemble_pred /= len(good_models)
        
        r2 = r2_score(y_test, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        
        self.results['Best Ensemble'] = {
            'r2': r2,
            'rmse': rmse,
            'models': list(good_models.keys()),
            'predictions': ensemble_pred
        }
        
        print(f"Best Ensemble R²: {r2:.4f}, RMSE: {rmse:.2f}")
        print(f"Models in ensemble: {list(good_models.keys())}")
        
        return ensemble_pred
    
    def cross_validation_analysis(self, X_selected):
        """Perform cross-validation on best models."""
        print("\n=== Cross-Validation Analysis ===")
        
        cv_results = {}
        
        for name, result in self.results.items():
            if 'model' in result:
                model = result['model']
                cv_scores = cross_val_score(model, X_selected, self.y, cv=10, scoring='r2', n_jobs=-1)
                cv_results[name] = {
                    'mean_r2': cv_scores.mean(),
                    'std_r2': cv_scores.std()
                }
                
                print(f"{name} CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return cv_results
    
    def plot_results(self):
        """Plot results."""
        print("\n=== Creating Visualizations ===")
        
        models = list(self.results.keys())
        r2_scores = [self.results[model]['r2'] for model in models]
        
        plt.figure(figsize=(12, 6))
        
        # Color code based on performance
        colors = ['gold' if r2 >= 0.85 else 'lightgreen' if r2 >= 0.8 else 'skyblue' for r2 in r2_scores]
        bars = plt.bar(models, r2_scores, color=colors)
        
        plt.title('Best Model Performance - R² Score', fontsize=16, fontweight='bold')
        plt.ylabel('R² Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add target lines
        plt.axhline(y=0.85, color='red', linestyle='--', linewidth=2, label='Target R² = 0.85')
        plt.axhline(y=0.80, color='orange', linestyle=':', linewidth=2, label='Good R² = 0.80')
        plt.legend()
        
        # Add value labels
        for bar, score in zip(bars, r2_scores):
            color = 'red' if score >= 0.85 else 'darkgreen' if score >= 0.8 else 'black'
            weight = 'bold' if score >= 0.8 else 'normal'
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight=weight, color=color)
        
        plt.tight_layout()
        plt.savefig('data_processing/best_ml_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def generate_report(self):
        """Generate comprehensive report."""
        print("\n=== Generating Report ===")
        
        # Sort by R² score
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        report = []
        report.append("BEST ML EXPERIMENTS - Compressive Strength Prediction")
        report.append("=" * 60)
        report.append(f"\nDataset: {self.df.shape[0]} samples")
        report.append(f"Target: Compressive Strength ({self.y.min():.1f} - {self.y.max():.1f} MPa)")
        report.append(f"Goal: R² >= 0.85")
        
        report.append("\nMODEL PERFORMANCE RANKING:")
        report.append("-" * 40)
        
        best_r2 = sorted_results[0][1]['r2']
        target_achieved = best_r2 >= 0.85
        
        for i, (name, result) in enumerate(sorted_results, 1):
            r2 = result['r2']
            rmse = result['rmse']
            
            if r2 >= 0.85:
                status = "TARGET ACHIEVED!"
            elif r2 >= 0.80:
                status = "EXCELLENT"
            elif r2 >= 0.75:
                status = "VERY GOOD"
            else:
                status = "NEEDS IMPROVEMENT"
            
            report.append(f"{i}. {name}")
            report.append(f"   R² = {r2:.4f} | RMSE = {rmse:.1f} MPa | {status}")
            report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 20)
        report.append(f"Best Model: {sorted_results[0][0]}")
        report.append(f"Best R² Score: {best_r2:.4f}")
        report.append(f"Best RMSE: {sorted_results[0][1]['rmse']:.1f} MPa")
        
        if target_achieved:
            report.append("\nMISSION ACCOMPLISHED!")
            report.append("Target R² >= 0.85 achieved!")
        else:
            gap = 0.85 - best_r2
            report.append(f"\nTarget not reached. Gap: {gap:.4f}")
            
            if best_r2 >= 0.80:
                report.append("Very close to target! Consider:")
                report.append("- More data collection")
                report.append("- Advanced feature engineering")
                report.append("- Deep learning approaches")
        
        # Save report
        report_text = '\n'.join(report)
        with open('data_processing/best_ml_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text

def main():
    """Main execution pipeline."""
    print("BEST ML EXPERIMENTS FOR COMPRESSIVE STRENGTH PREDICTION")
    print("MISSION: ACHIEVE R² >= 0.85")
    print("=" * 60)
    
    # Initialize predictor
    predictor = BestCompressiveStrengthPredictor()
    
    # Load and prepare data
    X, y = predictor.load_and_prepare_data()
    
    # Create best features
    X_best = predictor.create_best_features()
    
    # Select best features
    X_selected, selected_features = predictor.select_best_features(X_best, k=50)
    
    # Prepare data splits
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = predictor.prepare_data_splits(X_selected)
    
    # Run experiments
    print("\nRUNNING BEST ML EXPERIMENTS...")
    
    # 1. Best XGBoost
    predictor.best_xgboost_experiment(X_train, X_test, y_train, y_test)
    
    # 2. Best LightGBM
    predictor.best_lightgbm_experiment(X_train, X_test, y_train, y_test)
    
    # 3. Best Random Forest
    predictor.best_random_forest_experiment(X_train, X_test, y_train, y_test)
    
    # 4. Best Ensemble
    predictor.create_best_ensemble(y_test)
    
    # Cross-validation analysis
    cv_results = predictor.cross_validation_analysis(X_selected)
    
    # Visualizations
    predictor.plot_results()
    
    # Generate report
    predictor.generate_report()
    
    print("\nBEST EXPERIMENTS COMPLETE!")
    print("Check 'data_processing/best_ml_report.txt' for results")
    print("Check 'data_processing/best_ml_results.png' for visualizations")

if __name__ == "__main__":
    main()