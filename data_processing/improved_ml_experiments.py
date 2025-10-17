"""
Improved ML Experiments for Compressive Strength Prediction
Addresses identified issues from diagnostic analysis:
1. High target variability
2. Overfitting
3. Weak feature correlations
4. Source-based differences
5. Multicollinearity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ImprovedCompressiveStrengthPredictor:
    def __init__(self, data_path="data_processing/merged_database_cleaned.csv"):
        """Initialize improved predictor."""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess data addressing identified issues."""
        print("=== Loading and Preprocessing Data ===")
        
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
        
        # Remove constant and near-constant features
        variance_selector = VarianceThreshold(threshold=0.01)
        X_variance_filtered = variance_selector.fit_transform(self.X)
        selected_features = self.X.columns[variance_selector.get_support()]
        self.X = pd.DataFrame(X_variance_filtered, columns=selected_features, index=self.X.index)
        
        print(f"Removed {len(feature_cols) - len(selected_features)} low-variance features")
        
        # Remove highly correlated features (multicollinearity)
        corr_matrix = self.X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation > 0.95
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        self.X = self.X.drop(columns=high_corr_features)
        
        print(f"Removed {len(high_corr_features)} highly correlated features")
        print(f"Final features: {self.X.shape[1]}")
        
        return self.X, self.y
    
    def transform_target_variable(self):
        """Transform target variable to reduce variability."""
        print("\n=== Target Variable Transformation ===")
        
        # Test different transformations
        transformations = {
            'original': self.y.copy(),
            'log': np.log1p(self.y),
            'sqrt': np.sqrt(self.y),
            'power': None
        }
        
        # Power transformation
        try:
            pt = PowerTransformer(method='yeo-johnson')
            y_power = pt.fit_transform(self.y.values.reshape(-1, 1)).flatten()
            transformations['power'] = pd.Series(y_power, index=self.y.index)
        except:
            pass
        
        # Test which transformation gives best normality
        best_transform = 'original'
        best_stat = 0
        
        for name, y_trans in transformations.items():
            if y_trans is not None:
                # Test normality
                sample_size = min(len(y_trans), 5000)
                sample = y_trans.sample(sample_size) if hasattr(y_trans, 'sample') else pd.Series(y_trans).sample(sample_size)
                stat, p_value = stats.shapiro(sample)
                
                print(f"{name}: Shapiro-Wilk stat = {stat:.4f}")
                
                if stat > best_stat:
                    best_stat = stat
                    best_transform = name
        
        print(f"Best transformation: {best_transform}")
        
        if best_transform != 'original':
            self.y_transformed = transformations[best_transform]
            self.transform_type = best_transform
            
            # Store transformation parameters for inverse transform
            if best_transform == 'power':
                self.transformer = pt
        else:
            self.y_transformed = self.y.copy()
            self.transform_type = 'original'
        
        return self.y_transformed
    
    def create_robust_features(self):
        """Create robust features focusing on strongest predictors."""
        print("\n=== Creating Robust Features ===")
        
        X_robust = self.X.copy()
        
        # Focus on elements with strongest correlations
        key_elements = ['Cu', 'Fe', 'Al', 'Ni', 'Co', 'Cr', 'Mo']
        existing_elements = [e for e in key_elements if e in X_robust.columns]
        
        print(f"Creating features for: {existing_elements}")
        
        # 1. Only most important interactions (based on materials science)
        critical_pairs = [('Cu', 'Fe'), ('Al', 'Ni'), ('Co', 'Cr'), ('Mo', 'Fe')]
        
        for elem1, elem2 in critical_pairs:
            if elem1 in X_robust.columns and elem2 in X_robust.columns:
                X_robust[f'{elem1}_x_{elem2}'] = X_robust[elem1] * X_robust[elem2]
        
        # 2. Hardness transformations (strongest single predictor)
        if 'Hardness (HVN)' in X_robust.columns:
            hardness = X_robust['Hardness (HVN)']
            X_robust['Hardness_log'] = np.log1p(hardness)
            X_robust['Hardness_sqrt'] = np.sqrt(hardness)
            X_robust['Hardness_squared'] = hardness ** 2
        
        # 3. Simple composition features
        if 'Num_Elements' in X_robust.columns:
            X_robust['Element_Complexity'] = X_robust['Num_Elements'] ** 2
        
        print(f"Robust features: {X_robust.shape[1]} (added {X_robust.shape[1] - self.X.shape[1]})")
        
        return X_robust
    
    def select_robust_features(self, X_robust, k=30):
        """Select most robust features using conservative approach."""
        print(f"\n=== Selecting {k} Most Robust Features ===")
        
        # Use only statistical feature selection (most robust)
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X_robust, self.y_transformed)
        selected_features = X_robust.columns[selector.get_support()]
        
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X_robust.index)
        
        print(f"Selected features: {list(selected_features)}")
        
        return X_selected_df, selected_features
    
    def create_stratified_splits(self, X_selected):
        """Create stratified splits to handle source differences."""
        print("\n=== Creating Stratified Splits ===")
        
        # Create stratification based on target quantiles
        y_quantiles = pd.qcut(self.y_transformed, q=5, labels=False)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, self.y_transformed, test_size=0.2, 
            random_state=42, stratify=y_quantiles
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def regularized_xgboost_experiment(self, X_train, X_test, y_train, y_test):
        """XGBoost with strong regularization to prevent overfitting."""
        print("\n=== Regularized XGBoost Experiment ===")
        
        # Conservative parameters to prevent overfitting
        param_grid = {
            'n_estimators': [300, 500],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
            'reg_alpha': [1, 5, 10],
            'reg_lambda': [1, 5, 10],
            'min_child_weight': [3, 5, 10]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        # Use cross-validation for model selection
        best_score = -np.inf
        best_params = None
        best_model = None
        
        # Manual grid search with cross-validation
        for n_est in param_grid['n_estimators']:
            for max_d in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    for reg_a in param_grid['reg_alpha']:
                        for reg_l in param_grid['reg_lambda']:
                            
                            model = xgb.XGBRegressor(
                                n_estimators=n_est,
                                max_depth=max_d,
                                learning_rate=lr,
                                reg_alpha=reg_a,
                                reg_lambda=reg_l,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                min_child_weight=5,
                                random_state=42,
                                n_jobs=-1
                            )
                            
                            # Cross-validation
                            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                            mean_score = cv_scores.mean()
                            
                            if mean_score > best_score:
                                best_score = mean_score
                                best_params = {
                                    'n_estimators': n_est, 'max_depth': max_d, 'learning_rate': lr,
                                    'reg_alpha': reg_a, 'reg_lambda': reg_l
                                }
                                best_model = model
        
        # Train best model and evaluate
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.results['Regularized XGBoost'] = {
            'model': best_model,
            'r2': r2,
            'rmse': rmse,
            'cv_score': best_score,
            'params': best_params,
            'predictions': y_pred
        }
        
        print(f"Regularized XGBoost R²: {r2:.4f}, RMSE: {rmse:.2f}")
        print(f"CV Score: {best_score:.4f}")
        print(f"Best params: {best_params}")
        
        return best_model
    
    def regularized_random_forest_experiment(self, X_train, X_test, y_train, y_test):
        """Random Forest with regularization."""
        print("\n=== Regularized Random Forest Experiment ===")
        
        # Conservative parameters
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'max_features': [0.3, 0.5, 'sqrt']
        }
        
        best_score = -np.inf
        best_params = None
        best_model = None
        
        for n_est in param_grid['n_estimators']:
            for max_d in param_grid['max_depth']:
                for min_split in param_grid['min_samples_split']:
                    for min_leaf in param_grid['min_samples_leaf']:
                        for max_feat in param_grid['max_features']:
                            
                            model = RandomForestRegressor(
                                n_estimators=n_est,
                                max_depth=max_d,
                                min_samples_split=min_split,
                                min_samples_leaf=min_leaf,
                                max_features=max_feat,
                                random_state=42,
                                n_jobs=-1
                            )
                            
                            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                            mean_score = cv_scores.mean()
                            
                            if mean_score > best_score:
                                best_score = mean_score
                                best_params = {
                                    'n_estimators': n_est, 'max_depth': max_d,
                                    'min_samples_split': min_split, 'min_samples_leaf': min_leaf,
                                    'max_features': max_feat
                                }
                                best_model = model
        
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.results['Regularized Random Forest'] = {
            'model': best_model,
            'r2': r2,
            'rmse': rmse,
            'cv_score': best_score,
            'params': best_params,
            'predictions': y_pred
        }
        
        print(f"Regularized Random Forest R²: {r2:.4f}, RMSE: {rmse:.2f}")
        print(f"CV Score: {best_score:.4f}")
        
        return best_model
    
    def ridge_regression_experiment(self, X_train_scaled, X_test_scaled, y_train, y_test):
        """Ridge regression for comparison."""
        print("\n=== Ridge Regression Experiment ===")
        
        alphas = [0.1, 1, 10, 100, 1000]
        best_score = -np.inf
        best_alpha = None
        best_model = None
        
        for alpha in alphas:
            model = Ridge(alpha=alpha, random_state=42)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            mean_score = cv_scores.mean()
            
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha
                best_model = model
        
        best_model.fit(X_train_scaled, y_train)
        y_pred = best_model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.results['Ridge Regression'] = {
            'model': best_model,
            'r2': r2,
            'rmse': rmse,
            'cv_score': best_score,
            'params': {'alpha': best_alpha},
            'predictions': y_pred
        }
        
        print(f"Ridge Regression R²: {r2:.4f}, RMSE: {rmse:.2f}")
        print(f"CV Score: {best_score:.4f}")
        
        return best_model
    
    def create_conservative_ensemble(self, y_test):
        """Create conservative ensemble."""
        print("\n=== Creating Conservative Ensemble ===")
        
        # Only use models with positive CV scores
        good_models = {name: result for name, result in self.results.items() 
                      if result.get('cv_score', 0) > 0}
        
        if len(good_models) < 2:
            print("Not enough good models for ensemble")
            return None
        
        # Weight by CV score
        total_cv_score = sum(result['cv_score'] for result in good_models.values())
        
        ensemble_pred = np.zeros(len(y_test))
        for name, result in good_models.items():
            weight = result['cv_score'] / total_cv_score
            ensemble_pred += weight * result['predictions']
        
        r2 = r2_score(y_test, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        
        # Calculate ensemble CV score (approximate)
        ensemble_cv = sum(result['cv_score'] * result['cv_score'] / total_cv_score 
                         for result in good_models.values())
        
        self.results['Conservative Ensemble'] = {
            'r2': r2,
            'rmse': rmse,
            'cv_score': ensemble_cv,
            'models': list(good_models.keys()),
            'predictions': ensemble_pred
        }
        
        print(f"Conservative Ensemble R²: {r2:.4f}, RMSE: {rmse:.2f}")
        print(f"Ensemble CV Score: {ensemble_cv:.4f}")
        
        return ensemble_pred
    
    def comprehensive_cross_validation(self, X_selected):
        """Comprehensive cross-validation analysis."""
        print("\n=== Comprehensive Cross-Validation ===")
        
        cv_results = {}
        
        for name, result in self.results.items():
            if 'model' in result:
                model = result['model']
                
                # 10-fold CV
                cv_scores = cross_val_score(model, X_selected, self.y_transformed, cv=10, scoring='r2')
                
                cv_results[name] = {
                    'mean_r2': cv_scores.mean(),
                    'std_r2': cv_scores.std(),
                    'min_r2': cv_scores.min(),
                    'max_r2': cv_scores.max()
                }
                
                print(f"{name}:")
                print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"  Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
        
        return cv_results
    
    def generate_improved_report(self):
        """Generate improved analysis report."""
        print("\n=== Generating Improved Report ===")
        
        # Sort by CV score (more reliable than test score)
        sorted_results = sorted(
            [(name, result) for name, result in self.results.items() if 'cv_score' in result],
            key=lambda x: x[1]['cv_score'], reverse=True
        )
        
        report = []
        report.append("IMPROVED ML EXPERIMENTS - Compressive Strength Prediction")
        report.append("Addressing overfitting and data quality issues")
        report.append("=" * 65)
        
        report.append(f"\nDataset: {self.df.shape[0]} samples")
        report.append(f"Target transformation: {self.transform_type}")
        report.append(f"Features used: {self.X.shape[1]}")
        
        report.append("\nMODEL PERFORMANCE (sorted by CV score):")
        report.append("-" * 50)
        
        best_cv_score = sorted_results[0][1]['cv_score'] if sorted_results else 0
        
        for i, (name, result) in enumerate(sorted_results, 1):
            cv_score = result['cv_score']
            test_r2 = result['r2']
            rmse = result['rmse']
            
            # Status based on CV score (more reliable)
            if cv_score >= 0.7:
                status = "EXCELLENT"
            elif cv_score >= 0.6:
                status = "GOOD"
            elif cv_score >= 0.4:
                status = "FAIR"
            else:
                status = "POOR"
            
            report.append(f"{i}. {name}")
            report.append(f"   CV R²: {cv_score:.4f} | Test R²: {test_r2:.4f} | RMSE: {rmse:.1f} | {status}")
            report.append("")
        
        # Analysis
        report.append("ANALYSIS:")
        report.append("-" * 20)
        
        if best_cv_score >= 0.7:
            report.append("Excellent cross-validation performance achieved!")
        elif best_cv_score >= 0.6:
            report.append("Good cross-validation performance. Close to target.")
        elif best_cv_score >= 0.4:
            report.append("Fair performance. Significant improvements made over baseline.")
        else:
            report.append("Performance still needs improvement.")
        
        report.append("\nIMPROVEMENTS MADE:")
        report.append("- Removed multicollinear features")
        report.append("- Applied target transformation")
        report.append("- Used stratified sampling")
        report.append("- Applied strong regularization")
        report.append("- Conservative feature selection")
        report.append("- Cross-validation based model selection")
        
        if best_cv_score < 0.7:
            report.append("\nFURTHER RECOMMENDATIONS:")
            report.append("- Collect more high-quality data")
            report.append("- Investigate domain-specific features")
            report.append("- Consider advanced ensemble methods")
            report.append("- Explore deep learning approaches")
        
        # Save report
        report_text = '\n'.join(report)
        with open('data_processing/improved_ml_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text

def main():
    """Main improved ML pipeline."""
    print("IMPROVED ML EXPERIMENTS FOR COMPRESSIVE STRENGTH PREDICTION")
    print("Addressing overfitting and data quality issues")
    print("=" * 70)
    
    predictor = ImprovedCompressiveStrengthPredictor()
    
    # Load and preprocess data
    X, y = predictor.load_and_preprocess_data()
    
    # Transform target variable
    y_transformed = predictor.transform_target_variable()
    
    # Create robust features
    X_robust = predictor.create_robust_features()
    
    # Select robust features
    X_selected, selected_features = predictor.select_robust_features(X_robust, k=30)
    
    # Create stratified splits
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = predictor.create_stratified_splits(X_selected)
    
    # Run improved experiments
    print("\nRUNNING IMPROVED ML EXPERIMENTS...")
    
    # 1. Regularized XGBoost
    predictor.regularized_xgboost_experiment(X_train, X_test, y_train, y_test)
    
    # 2. Regularized Random Forest
    predictor.regularized_random_forest_experiment(X_train, X_test, y_train, y_test)
    
    # 3. Ridge Regression
    predictor.ridge_regression_experiment(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 4. Conservative Ensemble
    predictor.create_conservative_ensemble(y_test)
    
    # Comprehensive cross-validation
    cv_results = predictor.comprehensive_cross_validation(X_selected)
    
    # Generate report
    predictor.generate_improved_report()
    
    print("\nIMPROVED EXPERIMENTS COMPLETE!")
    print("Check 'data_processing/improved_ml_report.txt' for results")

if __name__ == "__main__":
    main()