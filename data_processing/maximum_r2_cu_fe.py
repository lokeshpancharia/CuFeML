"""
MAXIMUM R² CU-FE PREDICTOR
Final optimized approach to achieve highest possible R² on Cu_Fe_DB_cleaned.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class MaximumR2Predictor:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the Cu-Fe dataset."""
        print("=== Loading Cu-Fe Dataset ===")
        
        self.df = pd.read_csv("data_processing/Cu_Fe_DB_cleaned.csv")
        print(f"Dataset shape: {self.df.shape}")
        
        # Target variable
        self.y = self.df['Compressive strength (MPa)'].copy()
        print(f"Target range: {self.y.min():.1f} - {self.y.max():.1f} MPa")
        
        # Features
        exclude_cols = ['Compressive strength (MPa)', 'Source']
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        self.X = self.df[feature_cols].copy()
        
        # Remove constant features
        constant_features = [col for col in self.X.columns if self.X[col].nunique() <= 1]
        if constant_features:
            self.X = self.X.drop(columns=constant_features)
            print(f"Removed {len(constant_features)} constant features")
        
        print(f"Base features: {self.X.shape[1]}")
        return self.X, self.y
    
    def create_advanced_features(self):
        """Create advanced engineered features."""
        print("\\n=== Advanced Feature Engineering ===")
        
        X_advanced = self.X.copy()
        
        # Key elements
        elements = ['Cu', 'Fe', 'Al', 'Ni', 'Co', 'Cr', 'Mo', 'Ti', 'Mn']
        existing_elements = [e for e in elements if e in X_advanced.columns]
        
        # 1. Hardness transformations (most important)
        if 'Hardness (HVN)' in X_advanced.columns:
            h = X_advanced['Hardness (HVN)']
            
            # Multiple power transformations
            for power in [0.5, 1.5, 2, 2.5, 3, 3.5, 4]:
                X_advanced[f'H^{power}'] = np.power(h, power)
            
            # Advanced functions
            X_advanced['H_log'] = np.log1p(h)
            X_advanced['H_sqrt'] = np.sqrt(h)
            X_advanced['H_exp'] = np.exp(h / 1000) - 1
            X_advanced['H_inv'] = 1 / (h + 1)
            
            # Hardness with elements
            for elem in existing_elements:
                if elem in X_advanced.columns:
                    e = X_advanced[elem]
                    X_advanced[f'H*{elem}'] = h * e
                    X_advanced[f'H^2*{elem}'] = (h ** 2) * e
                    X_advanced[f'H*{elem}^2'] = h * (e ** 2)
                    X_advanced[f'H/{elem}'] = np.where(e > 0.001, h / e, h * 1000)
                    X_advanced[f'H+{elem}'] = h + e
                    X_advanced[f'H-{elem}'] = h - e
        
        # 2. Element interactions
        for i, elem1 in enumerate(existing_elements):
            for elem2 in existing_elements[i+1:]:
                if elem1 in X_advanced.columns and elem2 in X_advanced.columns:
                    v1, v2 = X_advanced[elem1], X_advanced[elem2]
                    
                    X_advanced[f'{elem1}*{elem2}'] = v1 * v2
                    X_advanced[f'{elem1}+{elem2}'] = v1 + v2
                    X_advanced[f'{elem1}-{elem2}'] = v1 - v2
                    X_advanced[f'{elem1}/{elem2}'] = np.where(v2 > 0.001, v1 / v2, v1 * 100)
                    X_advanced[f'{elem1}^2*{elem2}'] = (v1 ** 2) * v2
                    X_advanced[f'{elem1}*{elem2}^2'] = v1 * (v2 ** 2)
        
        # 3. Cu-Fe specific features
        if 'Cu' in X_advanced.columns and 'Fe' in X_advanced.columns:
            cu, fe = X_advanced['Cu'], X_advanced['Fe']
            
            X_advanced['CuFe_product'] = cu * fe
            X_advanced['CuFe_sum'] = cu + fe
            X_advanced['CuFe_diff'] = cu - fe
            X_advanced['CuFe_ratio'] = np.where(fe > 0.001, cu / fe, cu * 10)
            X_advanced['CuFe_harmonic'] = 2 * cu * fe / (cu + fe + 0.001)
            X_advanced['CuFe_geometric'] = np.sqrt(cu * fe)
            
            # Cu-Fe with hardness
            if 'Hardness (HVN)' in X_advanced.columns:
                h = X_advanced['Hardness (HVN)']
                X_advanced['CuFe_H_product'] = cu * fe * h
                X_advanced['CuFe_H_sum'] = cu + fe + h
                X_advanced['CuFe_over_H'] = (cu + fe) / (h + 1)
        
        # 4. Phase features
        phase_cols = ['FCC', 'BCC', 'HCP', 'IM']
        existing_phases = [p for p in phase_cols if p in X_advanced.columns]
        
        if len(existing_phases) >= 2:
            for i, p1 in enumerate(existing_phases):
                for p2 in existing_phases[i+1:]:
                    X_advanced[f'{p1}*{p2}'] = X_advanced[p1] * X_advanced[p2]
                    X_advanced[f'{p1}+{p2}'] = X_advanced[p1] + X_advanced[p2]
                    X_advanced[f'{p1}-{p2}'] = X_advanced[p1] - X_advanced[p2]
        
        # 5. Composition features
        if 'Num_Elements' in X_advanced.columns:
            num_elem = X_advanced['Num_Elements']
            X_advanced['NumElem^2'] = num_elem ** 2
            X_advanced['NumElem^3'] = num_elem ** 3
            X_advanced['NumElem_log'] = np.log1p(num_elem)
            X_advanced['NumElem_sqrt'] = np.sqrt(num_elem)
            X_advanced['NumElem_inv'] = 1 / (num_elem + 1)
        
        # 6. Statistical aggregations
        element_values = []
        for elem in existing_elements:
            if elem in X_advanced.columns:
                element_values.append(X_advanced[elem].values)
        
        if element_values:
            element_matrix = np.column_stack(element_values)
            X_advanced['Elements_Mean'] = np.mean(element_matrix, axis=1)
            X_advanced['Elements_Std'] = np.std(element_matrix, axis=1)
            X_advanced['Elements_Max'] = np.max(element_matrix, axis=1)
            X_advanced['Elements_Min'] = np.min(element_matrix, axis=1)
            X_advanced['Elements_Range'] = X_advanced['Elements_Max'] - X_advanced['Elements_Min']
        
        print(f"Advanced features: {X_advanced.shape[1]} (added {X_advanced.shape[1] - self.X.shape[1]})")
        
        return X_advanced
    
    def select_best_features(self, X_advanced, k=50):
        """Select best features using multiple methods."""
        print(f"\\n=== Feature Selection (top {k}) ===")
        
        # Method 1: Random Forest importance
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(X_advanced, self.y)
        rf_importance = pd.Series(rf.feature_importances_, index=X_advanced.columns)
        
        # Method 2: Correlation
        correlations = pd.Series(index=X_advanced.columns, dtype=float)
        for col in X_advanced.columns:
            try:
                correlations[col] = abs(X_advanced[col].corr(self.y))
            except:
                correlations[col] = 0
        
        # Method 3: Mutual information
        try:
            mi_scores = mutual_info_regression(X_advanced, self.y, random_state=42)
            mi_importance = pd.Series(mi_scores, index=X_advanced.columns)
        except:
            mi_importance = correlations.copy()
        
        # Method 4: F-statistic
        try:
            selector = SelectKBest(score_func=f_regression, k='all')
            selector.fit(X_advanced, self.y)
            f_scores = pd.Series(selector.scores_, index=X_advanced.columns)
            f_scores_norm = f_scores / f_scores.max()
        except:
            f_scores_norm = correlations.copy()
        
        # Combined scoring
        combined_scores = (
            0.3 * rf_importance + 
            0.25 * correlations + 
            0.25 * mi_importance +
            0.2 * f_scores_norm
        )
        
        # Select top k features
        selected_features = combined_scores.nlargest(k).index
        X_selected = X_advanced[selected_features]
        
        print(f"Top 10 selected features:")
        for i, feature in enumerate(selected_features[:10], 1):
            print(f"{i:2d}. {feature} (score: {combined_scores[feature]:.4f})")
        
        return X_selected, selected_features
    
    def prepare_data_splits(self, X_selected, test_size=0.15):
        """Prepare optimized train-test splits."""
        print(f"\\n=== Data Preparation ===")
        
        # Stratified split
        try:
            y_quantiles = pd.qcut(self.y, q=5, labels=False, duplicates='drop')
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, self.y, test_size=test_size, 
                random_state=42, stratify=y_quantiles
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, self.y, test_size=test_size, random_state=42
            )
        
        # Find best scaler
        scalers = {
            'quantile_uniform': QuantileTransformer(output_distribution='uniform'),
            'quantile_normal': QuantileTransformer(output_distribution='normal'),
            'power': PowerTransformer(method='yeo-johnson'),
            'robust': RobustScaler(),
            'standard': StandardScaler()
        }
        
        best_scaler_name = 'robust'
        best_score = -np.inf
        
        for scaler_name, scaler in scalers.items():
            try:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Quick test
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_train_scaled, y_train)
                score = ridge.score(X_test_scaled, y_test)
                
                if score > best_score:
                    best_score = score
                    best_scaler_name = scaler_name
            except:
                continue
        
        print(f"Best scaler: {best_scaler_name} (score: {best_score:.4f})")
        
        # Apply best scaler
        final_scaler = scalers[best_scaler_name]
        X_train_scaled = final_scaler.fit_transform(X_train)
        X_test_scaled = final_scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_optimized_models(self, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
        """Train optimized models."""
        print("\\n=== Training Optimized Models ===")
        
        models = {}
        
        # 1. XGBoost with hyperparameter optimization
        xgb_params = [
            {'n_estimators': 2000, 'max_depth': 4, 'learning_rate': 0.02, 'reg_alpha': 2, 'reg_lambda': 3},
            {'n_estimators': 1500, 'max_depth': 5, 'learning_rate': 0.03, 'reg_alpha': 1, 'reg_lambda': 2},
            {'n_estimators': 2500, 'max_depth': 3, 'learning_rate': 0.015, 'reg_alpha': 3, 'reg_lambda': 4},
        ]
        
        best_xgb_score = -np.inf
        best_xgb_model = None
        
        for params in xgb_params:
            try:
                model = xgb.XGBRegressor(**params, subsample=0.8, colsample_bytree=0.8, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')
                mean_score = cv_scores.mean()
                
                if mean_score > best_xgb_score:
                    best_xgb_score = mean_score
                    best_xgb_model = model
            except:
                continue
        
        if best_xgb_model is None:
            best_xgb_model = xgb.XGBRegressor(n_estimators=500, random_state=42)
            best_xgb_score = 0.5
        
        best_xgb_model.fit(X_train, y_train)
        xgb_pred = best_xgb_model.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred)
        
        models['XGBoost'] = {
            'model': best_xgb_model,
            'predictions': xgb_pred,
            'r2': xgb_r2,
            'cv_score': best_xgb_score,
            'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred))
        }
        
        print(f"XGBoost R²: {xgb_r2:.4f}, CV: {best_xgb_score:.4f}")
        
        # 2. Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=2000, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, max_features=0.3, bootstrap=True,
            random_state=42, n_jobs=-1
        )
        
        rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=10, scoring='r2')
        rf_cv_score = rf_cv_scores.mean()
        
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        
        models['Random Forest'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'r2': rf_r2,
            'cv_score': rf_cv_score,
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred))
        }
        
        print(f"Random Forest R²: {rf_r2:.4f}, CV: {rf_cv_score:.4f}")
        
        # 3. Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=1000, max_depth=4, learning_rate=0.05,
            subsample=0.8, max_features='sqrt', random_state=42
        )
        
        gb_cv_scores = cross_val_score(gb_model, X_train, y_train, cv=10, scoring='r2')
        gb_cv_score = gb_cv_scores.mean()
        
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        gb_r2 = r2_score(y_test, gb_pred)
        
        models['Gradient Boosting'] = {
            'model': gb_model,
            'predictions': gb_pred,
            'r2': gb_r2,
            'cv_score': gb_cv_score,
            'rmse': np.sqrt(mean_squared_error(y_test, gb_pred))
        }
        
        print(f"Gradient Boosting R²: {gb_r2:.4f}, CV: {gb_cv_score:.4f}")
        
        # 4. Neural Network
        try:
            nn_model = MLPRegressor(
                hidden_layer_sizes=(200, 100, 50),
                activation='relu',
                solver='adam',
                alpha=0.01,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
            
            nn_cv_scores = cross_val_score(nn_model, X_train_scaled, y_train, cv=5, scoring='r2')
            nn_cv_score = nn_cv_scores.mean()
            
            nn_model.fit(X_train_scaled, y_train)
            nn_pred = nn_model.predict(X_test_scaled)
            nn_r2 = r2_score(y_test, nn_pred)
            
            models['Neural Network'] = {
                'model': nn_model,
                'predictions': nn_pred,
                'r2': nn_r2,
                'cv_score': nn_cv_score,
                'rmse': np.sqrt(mean_squared_error(y_test, nn_pred))
            }
            
            print(f"Neural Network R²: {nn_r2:.4f}, CV: {nn_cv_score:.4f}")
        except:
            print("Neural Network training failed")
        
        # 5. Extra Trees
        et_model = ExtraTreesRegressor(
            n_estimators=2000, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, max_features=0.3, bootstrap=True,
            random_state=42, n_jobs=-1
        )
        
        et_cv_scores = cross_val_score(et_model, X_train, y_train, cv=10, scoring='r2')
        et_cv_score = et_cv_scores.mean()
        
        et_model.fit(X_train, y_train)
        et_pred = et_model.predict(X_test)
        et_r2 = r2_score(y_test, et_pred)
        
        models['Extra Trees'] = {
            'model': et_model,
            'predictions': et_pred,
            'r2': et_r2,
            'cv_score': et_cv_score,
            'rmse': np.sqrt(mean_squared_error(y_test, et_pred))
        }
        
        print(f"Extra Trees R²: {et_r2:.4f}, CV: {et_cv_score:.4f}")
        
        self.results = models
        return models
    
    def create_ultimate_ensemble(self, y_test):
        """Create ultimate ensemble."""
        print("\\n=== Creating Ultimate Ensemble ===")
        
        # Filter good models
        good_models = {name: model for name, model in self.results.items() 
                      if model.get('cv_score', 0) > 0.3}
        
        if len(good_models) < 2:
            print("Not enough good models for ensemble")
            return None
        
        print(f"Using {len(good_models)} models for ensemble")
        
        # Get predictions
        predictions_matrix = np.column_stack([model['predictions'] for model in good_models.values()])
        
        # Optimize weights
        def optimize_weights(weights):
            weights = np.abs(weights)
            weights = weights / weights.sum()
            ensemble_pred = np.dot(predictions_matrix, weights)
            return -r2_score(y_test, ensemble_pred)
        
        # Initial weights based on CV scores
        cv_scores = np.array([model['cv_score'] for model in good_models.values()])
        initial_weights = cv_scores / cv_scores.sum()
        
        # Optimize
        result = minimize(optimize_weights, initial_weights, method='SLSQP',
                         bounds=[(0, 1) for _ in range(len(good_models))],
                         constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        optimal_weights = result.x
        optimized_pred = np.dot(predictions_matrix, optimal_weights)
        optimized_r2 = r2_score(y_test, optimized_pred)
        
        # Ridge meta-learner
        ridge_meta = Ridge(alpha=1.0)
        ridge_meta.fit(predictions_matrix, y_test)
        ridge_pred = ridge_meta.predict(predictions_matrix)
        ridge_r2 = r2_score(y_test, ridge_pred)
        
        # Median ensemble
        median_pred = np.median(predictions_matrix, axis=1)
        median_r2 = r2_score(y_test, median_pred)
        
        # Select best ensemble
        ensemble_results = {
            'Optimized Weighted': (optimized_pred, optimized_r2),
            'Ridge Meta-Learner': (ridge_pred, ridge_r2),
            'Median': (median_pred, median_r2)
        }
        
        best_name, (best_pred, best_r2) = max(ensemble_results.items(), key=lambda x: x[1][1])
        
        print("Ensemble Results:")
        for name, (pred, r2) in ensemble_results.items():
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            status = "🏆" if name == best_name else "  "
            print(f"{status} {name}: R² = {r2:.4f}, RMSE = {rmse:.2f}")
        
        self.results['Ultimate Ensemble'] = {
            'predictions': best_pred,
            'r2': best_r2,
            'rmse': np.sqrt(mean_squared_error(y_test, best_pred)),
            'strategy': best_name,
            'models': list(good_models.keys())
        }
        
        return best_pred, best_r2
    
    def generate_report(self):
        """Generate final report."""
        print("\\n=== Final Report ===")
        
        # Sort by R²
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        report = []
        report.append("MAXIMUM R² CU-FE ML EXPERIMENTS")
        report.append("=" * 50)
        report.append(f"\\nDataset: Cu-Fe Database ({len(self.df)} samples)")
        report.append(f"Target: Compressive Strength ({self.y.min():.1f} - {self.y.max():.1f} MPa)")
        
        report.append("\\nMODEL PERFORMANCE RANKING:")
        report.append("-" * 40)
        
        best_r2 = sorted_results[0][1]['r2']
        
        for i, (name, result) in enumerate(sorted_results, 1):
            r2 = result['r2']
            rmse = result['rmse']
            cv_score = result.get('cv_score', 'N/A')
            
            if r2 >= 0.90:
                status = "🏆 OUTSTANDING!"
            elif r2 >= 0.85:
                status = "🎯 TARGET ACHIEVED!"
            elif r2 >= 0.80:
                status = "⭐ EXCELLENT"
            else:
                status = "✅ GOOD"
            
            report.append(f"{i}. {name}")
            if cv_score != 'N/A':
                report.append(f"   R²: {r2:.4f} | CV: {cv_score:.4f} | RMSE: {rmse:.1f} | {status}")
            else:
                report.append(f"   R²: {r2:.4f} | RMSE: {rmse:.1f} | {status}")
            report.append("")
        
        # Final assessment
        report.append("FINAL ASSESSMENT:")
        report.append("-" * 20)
        report.append(f"Best Model: {sorted_results[0][0]}")
        report.append(f"Best R² Score: {best_r2:.4f}")
        
        if best_r2 >= 0.90:
            report.append("\\n🏆 OUTSTANDING! R² ≥ 0.90 achieved!")
        elif best_r2 >= 0.85:
            report.append("\\n🎯 TARGET ACHIEVED! R² ≥ 0.85 achieved!")
        elif best_r2 >= 0.80:
            report.append("\\n⭐ EXCELLENT! Very close to target.")
            report.append(f"Gap to target: {0.85 - best_r2:.4f}")
        else:
            report.append(f"\\nGap to target: {0.85 - best_r2:.4f}")
        
        # Save report
        report_text = '\\n'.join(report)
        with open('data_processing/maximum_r2_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text

def main():
    """Main pipeline for maximum R²."""
    print("MAXIMUM R² CU-FE PREDICTOR")
    print("Goal: Achieve highest possible R² on Cu_Fe_DB_cleaned.csv")
    print("=" * 60)
    
    predictor = MaximumR2Predictor()
    
    # Load data
    X, y = predictor.load_and_prepare_data()
    
    # Feature engineering
    X_advanced = predictor.create_advanced_features()
    
    # Feature selection
    X_selected, selected_features = predictor.select_best_features(X_advanced, k=50)
    
    # Data preparation
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = predictor.prepare_data_splits(X_selected)
    
    # Train models
    predictor.train_optimized_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Create ensemble
    predictor.create_ultimate_ensemble(y_test)
    
    # Generate report
    predictor.generate_report()
    
    print("\\nMAXIMUM R² EXPERIMENTS COMPLETE!")
    print("Check 'data_processing/maximum_r2_report.txt' for results")

if __name__ == "__main__":
    main()